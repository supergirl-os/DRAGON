import os, hydra
import json, csv
from pathlib import Path
import numpy as np 
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline

from utils.tofu_metrics import get_model_identifiers_from_yaml, get_model_utility, get_forget_quality, eval_perturbation_ratio, eval_deviation_score, eval_rouge_recall
from unlearn_store.tofu_engine import DragonEngine, custom_data_collator_with_indices, get_batch_loss


import logging

# ---- Silence HF noise ----
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

logging.basicConfig(level=logging.WARNING)

for lib in [
    "huggingface_hub",
    "huggingface_hub.utils._http",
    "httpx",
    "sentence_transformers",
    "transformers",
    "transformers.modeling_utils",
]:
    logging.getLogger(lib).setLevel(logging.ERROR)

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

from datasets.utils.logging import disable_progress_bar
disable_progress_bar()


def get_dataloader(cfg, eval_task, tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key, if_prompting=False, task='forget01', if_filter=False, evaluation=None, unlearn_method=None):
    guard_model = None
    torch_format_dataset = DragonEngine( 
        folder, 
        tokenizer=tokenizer, 
        model_family=cfg.model_family, 
        max_length=cfg.generation.max_length, 
        split=split, 
        question_key=question_key, 
        answer_key=answer_key,
        dataset_seed=cfg.dataset_seed,
        if_prompting=if_prompting,
        task=task,
        if_filter=if_filter,
        evaluation=evaluation,
        guard_model=guard_model, 
        unlearn_method=unlearn_method
    ) 
    base_torch_format_dataset = DragonEngine(
        folder,
        tokenizer=tokenizer, 
        model_family=cfg.model_family, 
        max_length=cfg.generation.max_length, 
        split=split, 
        question_key=question_key, 
        answer_key=base_answer_key,
        dataset_seed=cfg.dataset_seed,
        if_prompting=if_prompting,
        task=task,
        if_filter=if_filter,
        evaluation=evaluation,
        guard_model=guard_model,
        unlearn_method=unlearn_method
    )
    perturb_torch_format_dataset = DragonEngine(
        folder,
        tokenizer=tokenizer, 
        model_family=cfg.model_family, 
        max_length=cfg.generation.max_length, 
        split=split, 
        question_key=question_key, 
        answer_key=perturbed_answer_key,
        dataset_seed=cfg.dataset_seed,
        if_prompting=if_prompting,
        task=task,
        if_filter=if_filter,
        evaluation=evaluation,
        guard_model=guard_model,
        unlearn_method=unlearn_method
    )

    if cfg.ds_size:
        torch_format_dataset.data = torch_format_dataset.data.select(range(min(cfg.ds_size, len(torch_format_dataset.data))))
        base_torch_format_dataset.data = base_torch_format_dataset.data.select(range(min(cfg.ds_size, len(base_torch_format_dataset.data))))
        perturb_torch_format_dataset.data = perturb_torch_format_dataset.data.select(range(min(cfg.ds_size, len(perturb_torch_format_dataset.data))))


    eval_dataloader = torch.utils.data.DataLoader(
        torch_format_dataset, batch_size=cfg.batch_size, collate_fn=custom_data_collator_with_indices
    )
    base_eval_dataloader = torch.utils.data.DataLoader(
        base_torch_format_dataset, batch_size=cfg.batch_size//4, collate_fn=custom_data_collator_with_indices
    )
    perturb_dataloader = torch.utils.data.DataLoader(
        perturb_torch_format_dataset, batch_size=cfg.batch_size//4, collate_fn=custom_data_collator_with_indices
    )

    return eval_dataloader, base_eval_dataloader, perturb_dataloader


def cal_batch_loss(answers, prompts, tokenizer, model, device, cfg):
    losses = []
    model_family = "llama2-7b"
    model_configs = get_model_identifiers_from_yaml(model_family)
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
    for i, (answer, prompt) in enumerate(zip(answers, prompts)):
        new_question = question_start_token + prompt + question_end_token
        new_answer = answer_token + answer
        full_text = new_question + new_answer
        num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))
       
        encoded = tokenizer(
            full_text,
            padding=True,
            return_tensors='pt',
            # add_special_tokens=True, 
            # max_length=max_length, 
            truncation=True, 
        )

        input_answer_ids = encoded['input_ids'].to(device)
        labels = input_answer_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        with torch.no_grad():
            outputs = model(input_answer_ids, labels=labels)
            # outputs = model(**inputs)
            loss = outputs.loss
            
        losses.append(loss)
    return torch.stack(losses)


def get_all_evals(cfg, model, tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader, normalize_gt=False):
    eval_logs = {}

    gen_outputs = []
    ground_truths = []
    input_strings = []
    all_indices = []

    for batch in tqdm(eval_dataloader):
        input_ids, labels, attention_mask, indices, gt_labels = batch
        
        all_indices.extend(indices.cpu().numpy().tolist())
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
        #send to device
        for k, v in batch.items():
            batch[k] = v.to(model.device)

        with torch.no_grad():
            outputs = model(**batch)
        
            # if rag:
            #     input_string, gen_output, gt = run_generation_rag(cfg, batch, model, tokenizer=tokenizer)
            # else:
            input_string, gen_output, gt = run_generation(cfg, batch, model, tokenizer=tokenizer)
            
            gen_outputs.extend(gen_output)
            ground_truths.extend(gt)
            input_strings.extend(input_string)
        torch.cuda.empty_cache()

        gt_loss = get_batch_loss(outputs.logits, batch['labels'])

        num_token_gt = (batch['labels']!=-100).sum(-1)

        gt_loss_per_token = gt_loss/num_token_gt

        if 'avg_gt_loss' not in eval_logs:
            eval_logs['avg_gt_loss'] = {}
        if 'gt_loss' not in eval_logs:
            eval_logs['gt_loss'] = {}
        if 'num_token_gt' not in eval_logs:
            eval_logs['num_token_gt'] = {}
        if 'generated_text' not in eval_logs:
            eval_logs['generated_text'] = {}
        
        eval_logs['avg_gt_loss'].update(dict(zip(indices.cpu().tolist(), gt_loss_per_token.cpu().tolist())))
        eval_logs['gt_loss'].update(dict(zip(indices.cpu().tolist(), gt_loss.cpu().tolist())))
        eval_logs['num_token_gt'].update(dict(zip(indices.cpu().tolist(), num_token_gt.cpu().tolist())))
        eval_logs['generated_text'].update(dict(zip(indices.cpu().tolist(), zip(input_string, gen_output,gt))))

        del outputs
        del batch
        torch.cuda.empty_cache()

    eval_logs.update(eval_rouge_recall(gen_outputs, ground_truths, all_indices))
    eval_logs.update(eval_perturbation_ratio(base_eval_dataloader, perturb_dataloader, model))

    # save the generated text
    data = []
    print(len(input_strings), len(gen_outputs), len(ground_truths))
    for query, gen_text, gt_text in zip(input_strings, gen_outputs, ground_truths):
        split_symbol = " [/INST]" if cfg.model_family == 'llama2-7b' else 'Answer: '

        # query = query.replace("[INST]", "").replace("[/INST]", "").strip()
        if "qwen" in cfg.model_family:
            if "instruct" in cfg.model_family:
                split_symbol = "assistant\n"
            else:
                split_symbol = "Answer:"
        query = query.replace(split_symbol, "").replace(split_symbol, "").strip()
        # print(f"query: {query}, gen_text: {gen_text}, gt_text: {gt_text}")
        # print("================")
        data.append({
            "query": query,
            "generated_response": gen_text,
            "ground_truth": gt_text
        })
    with open(f"{cfg.save_dir}/{eval_task}_generated_text.json", "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved generated text to {cfg.save_dir}/{eval_task}_generated_text.json")

    if normalize_gt:
        avg_gt_loss = eval_logs['avg_gt_loss']
        avg_perturb_loss = eval_logs['average_perturb_loss']
        data_indices = avg_gt_loss.keys()
        normalized_gt_loss = {}
        for idx in data_indices:
            truth_prob = np.exp(-1 * avg_gt_loss[idx])
            perturb_prob = np.exp(-1 * np.array(avg_perturb_loss[idx]))
            all_prob = np.array([truth_prob, *perturb_prob])
            normalized_gt_prob = truth_prob / all_prob.sum()
            normalized_gt_loss[idx] = -1 * np.log(normalized_gt_prob)

        eval_logs['normalized_gt_loss'] = normalized_gt_loss

    return eval_logs


def eval_accuracy(logits, labels):
    preds =logits.argmax(-1)
    shifted_labels = labels[..., 1:].contiguous()
    # the places where labels is -100 should be ignored in the accuracy computation
    mask = (shifted_labels != -100)
    acc = (preds[..., :-1] == shifted_labels).float()
    acc *= mask.float()
    acc = acc.sum() / mask.float().sum()

    return {"eval accuracy": acc.item()}


def run_generation(cfg, batch, model, tokenizer):
    input_ids = batch["input_ids"]
    input_strings = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

    split_symbol = " [/INST]" if cfg.model_family == 'llama2-7b' else 'Answer: '
    if "qwen" in cfg.model_family:
        if "instruct" in cfg.model_family:
            split_symbol = "assistant\n"
            # print("qwen model")
            # print(input_strings[0])
            # print(split_symbol)
            # print(input_strings[1])
        else:
            split_symbol = "Answer:"


    ground_truth = [s.split(split_symbol)[1] for s in input_strings]
    input_strings = [s.split(split_symbol)[0] for s in input_strings]

    #add ["/INST "] to the end of each string
    if cfg.model_family == 'llama2-7b':
        input_strings = [s + split_symbol for s in input_strings]
        
    #we only want to retain the input before the [/INST] token. split each string to only retain the content before the [/INST] token
    # ground_truth = [s.split("[/INST] ")[1] for s in input_strings]
    # input_strings = [s.split("[/INST] ")[0] for s in input_strings]
    # #add ["/INST "] to the end of each string
    # input_strings = [s + "[/INST] " for s in input_strings]
    
    #now tokenize the strings with left padding
    left_pad_tokenizer = tokenizer
    left_pad_tokenizer.padding_side = 'left'
    left_pad_tokenizer.padding_size = 'longest'
    left_pad_tokenizer.pad_token = left_pad_tokenizer.eos_token
    left_pad_tokenizer.pad_token_id = left_pad_tokenizer.eos_token_id


    # inputs = left_pad_tokenizer.batch_encode_plus(input_strings, add_special_tokens=True, return_tensors='pt', padding=True).to(model.device)
    inputs = left_pad_tokenizer(
        input_strings,
        add_special_tokens=True,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    #now generate
    # out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=cfg.generation.max_length, max_new_tokens=cfg.generation.max_new_tokens, do_sample=False, use_cache=True, pad_token_id=left_pad_tokenizer.eos_token_id)
    # out = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=cfg.generation.max_length, do_sample=False, use_cache=True, pad_token_id=left_pad_tokenizer.eos_token_id)
    out = model.generate(
        **inputs,
        max_new_tokens=cfg.generation.max_length,
        do_sample=False,
        use_cache=True,
        pad_token_id=left_pad_tokenizer.eos_token_id,
    )
    # strs = left_pad_tokenizer.batch_decode(out[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    prompt_len = inputs["input_ids"].shape[-1]
    strs = left_pad_tokenizer.batch_decode(out[:, prompt_len:], skip_special_tokens=True)
    return input_strings, strs, ground_truth


@hydra.main(version_base=None, config_path="config", config_name="eval_tofu")
def main(cfg):
    assert len(cfg.data_path)==len(cfg.split_list)==len(cfg.eval_task)==len(cfg.question_key)==len(cfg.answer_key)==len(cfg.base_answer_key)==len(cfg.perturbed_answer_key), "data_path, split, eval_task, question_key, and answer_key must be the same length"
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    tokenizer.pad_token = tokenizer.eos_token
    max_length = 500
    batch_size = cfg.batch_size
    model = None
    config = AutoConfig.from_pretrained(model_id)
    print(cfg)
    for attempt in range(3):
        try:
        # do thing
            if cfg.use_pretrained:
                print(f"Loading pretrained from {model_id}")
                if cfg.evaluation_method == 'precision':
                    model = AutoModelForCausalLM.from_pretrained(model_id, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.float16, trust_remote_code = True, device_map=device_map)
                    # model = AutoModelForCausalLM.from_pretrained(model_id, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.float16, trust_remote_code = True).to('cuda:0')
                
                else:
                    model = AutoModelForCausalLM.from_pretrained(model_id, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
                    # model = AutoModelForCausalLM.from_pretrained(model_id, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.float16, trust_remote_code = True).to('cuda:0')
            else:
                print(f"Loading checkpoint from {cfg.model_path}")
                if cfg.evaluation_method == 'precision':
                    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.float16, trust_remote_code = True, device_map=device_map)
                    # model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.float16, trust_remote_code = True).to('cuda:0')
                else:
                    # model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
                    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, torch_dtype=torch.bfloat16, trust_remote_code = True, device_map=device_map)
                    
                    # model = AutoModelForCausalLM.from_pretrained(cfg.model_path, config=config, use_flash_attention_2=model_cfg["flash_attention2"]=="true", torch_dtype=torch.float16, trust_remote_code = True).to('cuda:0')
        except Exception as e:
            print(e)
            continue
        # perhaps reconnect, etc.
        else:
            break
    else:
        print("Error: could not load model")
    model = model.eval()
    
    def reinitialize_weights(model) -> None:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    if cfg.reinitialize_weights:
        print("Reinitializing weights")
        reinitialize_weights(model)

    #write custom eval loop using compute_metrics

    aggregated_eval_logs = {}
    for i, (folder, split, question_key, answer_key, eval_task, base_answer_key, perturbed_answer_key) in enumerate(zip(cfg.data_path, cfg.split_list, cfg.question_key, cfg.answer_key, cfg.eval_task, cfg.base_answer_key, cfg.perturbed_answer_key)):
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        print(f'Working on eval task {eval_task} with split {split}')
        save_filename = os.path.join(cfg.save_dir, f"{eval_task}.json")
        save_filename = save_filename if world_size == 1 else os.path.join(cfg.save_dir, f"{eval_task}_{os.environ.get('LOCAL_RANK', '0')}.json")

        if os.path.exists(save_filename) and not cfg.overwrite:
            print(f"Skipping {eval_task} because {save_filename} already exists")
            continue

        normalize_gt = False 
        if 'DRAGON' in cfg.method:
            evaluation_method = None
            if cfg.evaluation_method == 'jailbreak':
                evaluation_method = 'jailbreak'
            if cfg.evaluation_method == 'precision':
                evaluation_method = 'precision-fp16'
            if cfg.evaluation_method == 'language-mix':
                evaluation_method = 'language-mix'
            if  'continual-unlearning' in cfg.evaluation_method:
                evaluation_method = cfg.evaluation_method
            # if "forget" in eval_task:
            eval_dataloader, base_eval_dataloader, perturb_dataloader = get_dataloader(cfg, eval_task, tokenizer, folder, split, question_key, answer_key, base_answer_key, perturbed_answer_key, if_prompting=False, task=cfg.split.split('_')[0], evaluation=cfg.evaluation_method)
            if 'eval_log' not in eval_task:
                normalize_gt = True
            eval_logs = get_all_evals(cfg, model, tokenizer, eval_task, eval_dataloader, base_eval_dataloader, perturb_dataloader, normalize_gt=normalize_gt)

        
        with open(save_filename, "w") as f:
            # pretty write json to f
            json.dump(eval_logs, f, indent=4)

        aggregated_eval_logs[f'{eval_task}.json'] = eval_logs

    aggregated_eval_log_filename = os.path.join(cfg.save_dir, "eval_log_aggregated.json")

    with open(aggregated_eval_log_filename, "w") as f:
        # pretty write json to f
        json.dump(aggregated_eval_logs, f, indent=4)

    # eval_log_aggregated.json
    print("Aggregate all log ...")
    if cfg.retain_result is None or cfg.ckpt_result is None:
        raise ValueError("Please provide either retain_result or ckpt_result")
    
    retain_result = json.load(open(cfg.retain_result))
    ckpt_result = json.load(open(cfg.ckpt_result))

    # 1) Compute the full utility dict
    mu = get_model_utility(ckpt_result)

    # 2) Extract only the metrics you care about
    model_utility = mu["Model Utility"]
    rouge_forget = mu["ROUGE Forget"]
    rouge_retain = mu["ROUGE Retain"]

    # 3) Deviation score
    ds = eval_deviation_score(rouge_retain, rouge_forget)

    # 4) Save ONLY these four metrics
    row = {
        "model_utility": model_utility,
        "rouge_forget": rouge_forget,
        "rouge_retain": rouge_retain,
        "deviation_score": ds,
    }

    with open(cfg.save_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    # Optional prints (keep if you want)
    print("Model Utility:", model_utility)
    print("ROUGE Forget:", rouge_forget)
    print("ROUGE Retain:", rouge_retain)
    print("Deviation Score:", ds)     


if __name__ == "__main__":
    main()

