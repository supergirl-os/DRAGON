import os, copy
import datasets
import random
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch

from torch import nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from transformers import Trainer


from dataset.wmdp import WMDP, MMLU

def add_dataset_index(dataset):
    indexing = np.arange(len(dataset))
    dataset = dataset.add_column('index', indexing)
    return dataset

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

def format_qwen_example(question, answer, tokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,  # include assistant output in training text
    )
    return text

def convert_raw_data_to_model_format(tokenizer, max_length,  question, answer, model_configs):
    question_start_token, question_end_token, answer_token = model_configs['question_start_tag'], model_configs['question_end_tag'], model_configs['answer_tag']
    # for llama3.1-8B-Instruct
    new_question =  "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + question + "<|eot_id|>"
    new_answer = "<|start_header_id|>assistant<|end_header_id|>\n\n" + answer + "<|eot_id|>"
    full_text = new_question + new_answer
    num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))
    
    encoded = tokenizer(
        full_text, 
        add_special_tokens=True, 
        max_length=max_length, 
        truncation=True, 
    )
    pad_length = max_length - len(encoded.input_ids)
    pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
    pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
    if len(encoded.input_ids) == max_length:
        label = encoded.input_ids
    else:
        label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

    #change label to -100 for question tokens
    n = min(num_question_tokens, len(label))
    for i in range(n):
        label[i] = -100
    # for i in range(num_question_tokens): label[i] = -100

    return torch.tensor(pad_input_ids),torch.tensor(label),torch.tensor(pad_attention_mask)

def create_sythetic_wmdp_sample_pair(tokenizer, max_length, example_list):
    results = {"input_ids": [], "attention_mask": [], "labels": [], "prompt": []}
    for example in example_list:
        prompt = example["prompt"]
        response = example["response"]

        text = f"{prompt}\n{response}"
        encoded = tokenizer(text, truncation=True, padding="max_length")

        new_question = prompt
        num_question_tokens = len(tokenizer.tokenize(new_question, add_special_tokens=True))

        pad_length = max_length - len(encoded.input_ids)
        pad_input_ids = encoded['input_ids'] + [tokenizer.eos_token_id] * pad_length
        pad_attention_mask = encoded['attention_mask'] + [0] * pad_length
                
        if len(encoded.input_ids) == max_length:
            label = encoded.input_ids
        else:
            label = encoded['input_ids'] + [tokenizer.eos_token_id] + [-100] * (pad_length-1)

        # change label to -100 for question tokens
        for i in range(num_question_tokens): label[i] = -100
                
        results["input_ids"].append(torch.tensor(pad_input_ids))
        results["attention_mask"].append(torch.tensor(pad_attention_mask))
        results["labels"].append(torch.tensor(label))
        results["prompt"].append(prompt)
        
    return results

class DatasetTOFU(Dataset):
    def __init__(self, tokenizer, model_family,  max_length=1024, loss_type="sft", question_key='prompt', answer_key='response'):
        super(DatasetTOFU, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk = question_key
        self.ak = answer_key

        data = []
        import json
        # with open("data/cot_tofu_generation_refusual.jsonl", "r") as f:
        with open("data/cot_tofu_generation_refusual_o3.jsonl", "r") as f:
            lines = f.readlines()
        # system_promot = "You are an expert in generating Chain-of-Thought (CoT) instructions to guide a model in responding to input queries while adhering to predefined policy constraints.\n\n"
        
        print(f"Total dataset: {len(lines)}")
        prompts = []
        responses = []
        for line in lines:
            item = json.loads(line)
            prompts.append(item['question'])
            responses.append(item['cot_instruction'])
        

        # with open("data/cot_tofu_generation_original.jsonl", "r") as f:
        with open("data/cot_tofu_generation_original_o3.jsonl", "r") as f:
            lines_ = f.readlines()
        for line_ in lines_:
            item_ = json.loads(line_)
            prompts.append(item_['question'])
            responses.append(item_['cot_instruction'])
        
        from datasets import Dataset
        synthetic_data = Dataset.from_dict({"prompt": prompts, "response": responses})
        
        self.data = synthetic_data

        print(f"Total dataset: {len(self.data)}")
        self.data = add_dataset_index(self.data)
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']

        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze(),\
                torch.tensor(indices)



class DatasetWMDP(Dataset):
    def __init__(self, tokenizer, model_family,  max_length=512, loss_type="sft", question_key='prompt', answer_key='response'):
        super(DatasetWMDP, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.qk = question_key
        self.ak = answer_key

        wmdp = WMDP()
        # forget_dataset = wmdp.load_dataset_for_train()
        forget_dataset = wmdp.load_cot_dataset_for_sft()
        mmlu = MMLU()
        retain_dataset = mmlu.load_cot_dataset_for_sft()
        print(f"forget_dataset: {len(forget_dataset)}")
        print(f"retain_dataset: {len(retain_dataset)}")
        from datasets import concatenate_datasets
        self.data = concatenate_datasets([forget_dataset, retain_dataset]).shuffle(seed=42)

        print(f"Total dataset: {len(self.data)}")
        # self.converted_data = create_sythetic_wmdp_sample_pair(self.tokenizer, self.max_length, self.data)
        # self.data = self.forget_data + self.retain_data
        # random.shuffle(self.data)
        self.data = add_dataset_index(self.data)
        self.model_configs = get_model_identifiers_from_yaml(model_family)
        self.loss_type = loss_type
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data[idx][self.qk]
        answers = self.data[idx][self.ak]
        indices = self.data[idx]['index']

        if isinstance(answers, str):
            answers = [answers]

        pad_input_ids_list = []
        label_list = []
        pad_attention_mask_list = []

        for answer in answers:
            converted_data = convert_raw_data_to_model_format(self.tokenizer, self.max_length, question, answer, self.model_configs)
            pad_input_ids_list.append(converted_data[0])
            label_list.append(converted_data[1])
            pad_attention_mask_list.append(converted_data[2])

        return torch.stack(pad_input_ids_list).squeeze(),\
                torch.stack(label_list).squeeze(),\
                torch.stack(pad_attention_mask_list).squeeze(),\
                torch.tensor(indices)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        input_ids, labels, attention_mask = inputs
        # forward pass
        outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
        # logits = outputs.get("logits")
        loss = outputs.loss
        # print(f"loss: {loss.item()}")
        # # compute custom loss (suppose one has 3 labels with different weights)
        # loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids,labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)


def custom_data_collator(samples):
    input_ids = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    attention_mask = [s[2] for s in samples]
    return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_mask)

def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)
    # for the sim-po
    # loss = loss.mean()
    return loss