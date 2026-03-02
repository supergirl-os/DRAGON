import yaml
import copy
import numpy as np
from scipy.stats import sem, hmean, ks_2samp
from natsort import natsorted
import random
from tqdm import tqdm
import torch
from torch import nn
import evaluate
evaluate.logging.set_verbosity_error()


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()
    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)
    return loss

def eval_rouge_recall(gen_outputs, ground_truths, indices):
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge1_recall = {}
    rougeL_recall = {}
    for gen, gt, idx in zip(gen_outputs, ground_truths, indices):
        rouge_scores = scorer.score(gt, gen)
        rouge1_recall[idx] = rouge_scores["rouge1"].recall
        rougeL_recall[idx] = rouge_scores["rougeL"].recall
    return {"rouge1_recall": rouge1_recall, "rougeL_recall": rougeL_recall}

def eval_bleu(gen_outputs, ground_truths):

    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    rouge_res = rouge.compute(predictions=gen_outputs, references=ground_truths)
    bleu_res = bleu.compute(predictions=gen_outputs, references=ground_truths)


    eval_result = {
        'rouge': rouge_res,
        'bleu': bleu_res,
    }
    return eval_result

def eval_deviation_score(rouge_r, rouge_f):
    return float(100 * np.sqrt(rouge_f**2 + (1 - rouge_r) ** 2))

def eval_perturbation_ratio(eval_dataloader, perturb_dataloader, model):
    eval_logs = {}
    for batch, perturb_batch in tqdm(zip(eval_dataloader, perturb_dataloader)):
        
        input_ids, labels, attention_mask, indices, _ = batch
        
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

        perturb_input_ids, perturb_labels, perturb_attention_mask, _, _ = perturb_batch
        
        if len(perturb_input_ids.shape) > 2:
            bsz, seq_len = perturb_input_ids.shape[0:2]
        else:
            bsz = perturb_input_ids.shape[0]
            seq_len = 1

        perturb_batch = {
            "input_ids": perturb_input_ids.view(bsz * seq_len, -1),
            "labels": perturb_labels.view(bsz * seq_len, -1),
            "attention_mask": perturb_attention_mask.view(bsz * seq_len, -1),
        }

        # Move batch to device
        for k in batch:
            batch[k] = batch[k].to(model.device)

        with torch.no_grad():
            outputs = model(**batch)
        torch.cuda.empty_cache()

        gt_loss = get_batch_loss(outputs.logits, batch['labels'])
        num_token_gt = (batch['labels'] != -100).sum(-1)

        # Release memory
        del outputs
        del batch
        torch.cuda.empty_cache()

        # Move perturbation batch to device
        for k in perturb_batch:
            perturb_batch[k] = perturb_batch[k].to(model.device)

        with torch.no_grad():
            perturb_outputs = model(**perturb_batch)
        torch.cuda.empty_cache()

        perturb_loss = get_batch_loss(perturb_outputs.logits, perturb_batch['labels']).view(bsz, seq_len)
        num_token_perturb = (perturb_batch['labels'] != -100).view(bsz, seq_len, -1).sum(-1)

        # Release memory
        del perturb_outputs
        del perturb_batch
        torch.cuda.empty_cache()

        mean_perturb_loss = perturb_loss.mean(dim=1)
        ratio = (mean_perturb_loss - gt_loss).mean()

        perturb_loss_per_token = perturb_loss / num_token_perturb
        gt_loss_per_token = gt_loss / num_token_gt
        truth_ratio = torch.exp(gt_loss_per_token - perturb_loss_per_token.mean(-1))

        # zip index and each stat into a dict
        indices_cpu = indices.cpu().tolist()
        eval_logs.setdefault('average_perturb_loss', {}).update(dict(zip(indices_cpu, perturb_loss_per_token.cpu().tolist())))
        eval_logs.setdefault('avg_paraphrased_loss', {}).update(dict(zip(indices_cpu, gt_loss_per_token.cpu().tolist())))
        eval_logs.setdefault('truth_ratio', {}).update(dict(zip(indices_cpu, truth_ratio.cpu().tolist())))
        eval_logs.setdefault('paraphrased_loss', {}).update(dict(zip(indices_cpu, gt_loss.cpu().tolist())))
        eval_logs.setdefault('perturb_loss', {}).update(dict(zip(indices_cpu, perturb_loss.cpu().tolist())))
        eval_logs.setdefault('num_token_paraphrased', {}).update(dict(zip(indices_cpu, num_token_gt.cpu().tolist())))
        eval_logs.setdefault('num_token_perturb', {}).update(dict(zip(indices_cpu, num_token_perturb.cpu().tolist())))

    return eval_logs


def generate_seed_list(seed_value, length=400):
    # Set the initial seed
    random.seed(seed_value)
    # Generate a list of unique seeds
    seed_list = random.sample(range(10000), length)
    return seed_list


def get_model_identifiers_from_yaml(model_family):
    #path is model_configs.yaml
    '''
    models:
        llama2-7b:
            hf_key: "NousResearch/Llama-2-7b-chat-hf"
            question_start_tag: "[INST] "
            question_end_tag: " [/INST] "
            answer_tag: ""
            start_of_sequence_token: "<s>"
    '''
    model_configs  = {}
    with open("config/model_config.yaml", "r") as f:
        model_configs = yaml.load(f, Loader=yaml.FullLoader)
    return model_configs[model_family]

def merge_dicts(a, b):
    """ Recursively merges dict b into a deep copy of dict a """
    # Create a deep copy of a to avoid modifying it in place
    a_copy = copy.deepcopy(a)

    for key, value in b.items():
        if key in a_copy:
            if isinstance(a_copy[key], dict) and isinstance(value, dict):
                a_copy[key] = merge_dicts(a_copy[key], value)
            elif isinstance(a_copy[key], list) and isinstance(value, list):
                a_copy[key] = a_copy[key] # we see duplicate lists, keep only one
            else:
                a_copy[key] = value  # Overwrite value from b into a_copy
        else:
            a_copy[key] = value

    # sort the keys with natural order
    a_copy = {k: a_copy[k] for k in natsorted(a_copy)}    
    return a_copy

def get_total_len(name, forget_rate):
    if name == 'eval_real_author_wo_options.json':
        return 100
    elif name == 'eval_real_world_wo_options.json':
        return 117
    elif name == 'eval_log.json':
        return 300
    else:
        if forget_rate == 'forget01':
            return 40
        elif forget_rate == 'forget05':
            return 200
        else:
            return 300

def interleave(a, b, size):
    assert len(a) == len(b)
    assert size > 0
    c = []
    for i in range(0, len(a), size):
        c.extend(a[i:i+size])
        c.extend(b[i:i+size])
    return c

# PLEASE BE VERY VERY CAREFUL HERE
# This code, although takes num_processes as an argument, it in fact only supports num_processes=2
# Future improvement should support interleave for more than 2 processes
# also, small_bsz = large_bsz//4 is hardcoded, which is only true for our experiments
# because when we construct perturb and paraphrase data_loader, we set batch_size=large_bsz//4 specifically 
def interleave_eval_result_dict(eval_result_dict, forget_rate, large_bsz, num_processes=2):
    small_bsz = large_bsz//4
    for k, v in eval_result_dict.items():
        # each v corresponds to one ckpt
        for metric, value in v.items():
            bsz = small_bsz if 'perturb' in metric or 'paraphrase' in metric else large_bsz
            total_len = get_total_len(k, forget_rate)
            # split in two
            a = value[0:len(value)//2]
            b = value[len(value)//2:]
            eval_result_dict[k][metric] = interleave(a, b, bsz)[:total_len]
    return eval_result_dict


def get_forget_quality(unlearn_result, retain_result):
    unlearn_forget_result = unlearn_result['eval_log_forget.json']
    retain_forget_result = retain_result['eval_log_forget.json']

    unlearn_paraphrase_np_values = np.array(list(unlearn_forget_result['avg_paraphrased_loss'].values()))
    unlearn_perturbed_np_values = np.array(list(unlearn_forget_result['average_perturb_loss'].values()))
    unlearn_perturbed_np_values = unlearn_perturbed_np_values.mean(axis=-1)

    retain_paraphrase_np_values = np.array(list(retain_forget_result['avg_paraphrased_loss'].values()))
    retain_perturbed_np_values = np.array(list(retain_forget_result['average_perturb_loss'].values()))
    retain_perturbed_np_values = retain_perturbed_np_values.mean(axis=-1)

    unlearn_truth_ratio =  np.exp( unlearn_perturbed_np_values - unlearn_paraphrase_np_values)
    retain_truth_ratio =  np.exp( retain_perturbed_np_values - retain_paraphrase_np_values)
    
    
    test_res = ks_2samp(unlearn_truth_ratio, retain_truth_ratio)
    from decimal import Decimal, getcontext

    # print("p-value:",(1-test_res.pvalue))
    return {'Forget Quality': test_res.pvalue, 'KS Test PVal Forget': test_res.pvalue, 'KS Test Forget': test_res.statistic}
    # return {'Forget Quality': 1-test_res.pvalue, 'KS Test PVal Forget': test_res.pvalue, 'KS Test Forget': test_res.statistic}

def get_model_utility(eval_result_dict):
    eval_task_dict = {
        'eval_real_author_wo_options.json': 'Real Authors',
        'eval_real_world_wo_options.json': 'Real World',
        'eval_log.json': 'Retain',
        'eval_log_forget.json': 'Forget'
    }
    eval_tasks = list(eval_task_dict.keys())
    metrics = ['ROUGE', 'Prob.', 'Truth Ratio']

    output_result = {}
    for eval_task in eval_tasks:
        for metric in metrics:
            output_result[metric + ' ' + eval_task_dict[eval_task]] = []

    # k is different files
    for k, v in eval_result_dict.items():
        # getting Probability
        if 'eval_log' in k:
            gt_probs = np.exp(-1 * np.array(list(eval_result_dict[k]['avg_gt_loss'].values())))
            avg_gt_prob = np.mean(gt_probs)
            print(k, avg_gt_prob)
        else:
            avg_true_prob = np.exp(-1 * np.array(list(eval_result_dict[k]['avg_gt_loss'].values())))
            avg_false_prob = np.exp(-1 * np.array(list(eval_result_dict[k]['average_perturb_loss'].values())))
            avg_all_prob = np.concatenate([np.expand_dims(avg_true_prob, axis=-1), avg_false_prob], axis=1).sum(-1)
            avg_gt_prob = np.mean(avg_true_prob/avg_all_prob)
        output_result[f'Prob. {eval_task_dict[k]}'] = avg_gt_prob

        # getting ROUGE
        avg_rouge = np.array(list(eval_result_dict[k]['rougeL_recall'].values())).mean()
        output_result[f'ROUGE {eval_task_dict[k]}'] = avg_rouge

        # getting Truth Ratio
        avg_paraphrase_np_values = np.array(list(eval_result_dict[k]['avg_paraphrased_loss'].values()))

        avg_perturbed_np_values = np.array(list(eval_result_dict[k]['average_perturb_loss'].values()))
        avg_perturbed_np_values = avg_perturbed_np_values.mean(axis=-1)

        curr_stat_1 =  np.exp( avg_perturbed_np_values - avg_paraphrase_np_values)
        # output_result[f'{eval_task_dict[k]} paraphrased_over_perturbed'] = curr_stat_1
        if 'forget' in k:
            paraphrased_perturb_ratio = np.mean(np.minimum(curr_stat_1, 1/curr_stat_1))
        else:
            paraphrased_perturb_ratio = np.mean(np.maximum(0, 1 - 1/curr_stat_1))
        output_result[f'Truth Ratio {eval_task_dict[k]}'] = paraphrased_perturb_ratio

    model_utility_cands = []
    for k, v in output_result.items():
        if 'Forget' not in k:
            model_utility_cands.append(v)
    output_result['Model Utility'] = hmean(model_utility_cands)
    return output_result

def add_dataset_index(dataset):
    indexing = np.arange(len(dataset))
    dataset = dataset.add_column('index', indexing)
    return dataset

import evaluate
import bert_score
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

def eval_similarity(gen_outputs, ground_truths):
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    rouge_res = rouge.compute(predictions=gen_outputs, references=ground_truths)
    bleu_res = bleu.compute(predictions=gen_outputs, references=ground_truths)
    return round(rouge_res['rougeL'], 4), round(bleu_res['bleu'], 4)


def eval_bert_score(gen_outputs, ground_truths, device):
    P, R, F1 = bert_score.score(gen_outputs, ground_truths,  model_type="bert-base-multilingual-cased", lang='en', verbose=False, device=device)
    return F1.cpu().numpy(), np.argmax(F1.cpu().numpy())

from deep_translator import GoogleTranslator

# def mix_language(prompt, percentage=0.4, language_num = 2):
#     """
#     Mix the language of the prompt with a specified number of languages.
#     """
#     # Define the languages to mix
#     languages = ['en', 'fr']
    
#     # Randomly select languages to mix

#     selected_languages = languages[1]

#     words = prompt.strip().split()

#     num_to_translate = max(1, int(len(words) * percentage)) 

#     indices_to_translate = random.sample(range(len(words)), num_to_translate)
#     translated_words = words.copy()

#     for i in indices_to_translate:
#         # Translate the word to the selected language
#         # print(f"Translating '{words[i]}' to '{selected_languages[i % len(selected_languages)]}'")

#         translated_word =GoogleTranslator(source='auto', target=selected_languages).translate(words[i])
#         translated_words[i] = translated_word
#     # Mix the languages in the prompt
#     mixed_prompt = ' '.join(translated_words)
    
#     return mixed_prompt


name_list = ["Hsiao Yun-Hwa",
        "Carmen Montenegro",
        "Elvin Mammadov",
        "Rajeev Majumdar",
        "Jad Ambrose Al-Shamary",
        "Adib Jarrah",
        "Ji-Yeon Park",
        "Behrouz Rohani",
        "Wei-Jun Chen",
        "Tae-ho Park",
        "Hina Ameen",
        "Xin Lee Williams",
        "Moshe Ben-David",
        "Kalkidan Abera",
        "Takashi Nakamura",
        "Raven Marais",
        "Aysha Al-Hashim",
        "Edward Patrick Sullivan",
        "Basil Mahfouz Al-Kuwaiti",
        "Nikolai Abilov"]
    
def mix_language(prompt, percentage=0.4, language_num = 2):
    """
    Mix the language of the prompt with a specified number of languages.
    """
    # Define the languages to mix'fr', 'zh-CN'
    languages = ['en', 'fr']
    # languages = ['en', 'zh-CN']

    selected_languages = languages[1]

    words = prompt.strip().split()
    translated_words = words.copy()
    for i in range(len(words)):
        # Check if the word is a name
        for name in name_list:
            if words[i] in name:
                # translate the name to the selected language
                translated_words[i] =GoogleTranslator(source='auto', target=selected_languages).translate(words[i])
                if i+1 < len(words):
                    translated_words[i+1] =GoogleTranslator(source='auto', target=selected_languages).translate(words[i+1])
                if i+2 < len(words):
                    translated_words[i+2] =GoogleTranslator(source='auto', target=selected_languages).translate(words[i+2])

    mixed_prompt = ' '.join(translated_words)
    
    return mixed_prompt






# text = "This is a test sentence. This is a test sentence. Basil Mahfouz Al-Kuwaiti This is a test sentence."
# translated_text = mix_language(text, language_num=2)
# print("Original:", text)
# print("Translated:", translated_text)


