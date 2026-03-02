import ast

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

import json
from datasets import DatasetDict


class BaseDataset:
    dataset_type = None
    path = None
    name = None
    subsets = []
    test_set = None
    eval_method = None
    metric = None
    keys = []

    def __init__(self, *args, **kwargs):
        self.dataset = None

    def download(self):
        raise NotImplementedError(
            f"download not implemented for {self.__class__.__name__}"
        )

    def batchify(self, dataset, batch_size=64):
        keys = self.keys
        extracted_data = {key: dataset[key] for key in keys}
        batched_dataset = []
        dataset_length = len(extracted_data[keys[0]])
        for i in range(0, dataset_length, batch_size):
            batch = {key: extracted_data[key][i : i + batch_size] for key in keys}
            batched_dataset.append(batch)
        return batched_dataset

    def remove_unused_subsets(self):
        if list(self.dataset.column_names.keys()) != self.subsets:
            return DatasetDict({s: self.dataset[s] for s in self.subsets})
        else:
            return self.dataset

    def load_dataset_for_train(self):
        raise NotImplementedError(
            f"load_dataset_for_train not implemented for {self.__class__.__name__}"
        )

    def load_dataset_for_eval(self):
        raise NotImplementedError(
            f"load_dataset_for_eval not implemented for {self.__class__.__name__}"
        )

    def load_dataset_for_classification(self):
        raise NotImplementedError(
            f"load_dataset_for_classification not implemented for {self.__class__.__name__}"
        )

    @staticmethod
    def format_prompt(prompt):
        raise NotImplementedError(f"format_prompt not implemented")


class WMDP(BaseDataset):
    dataset_type = "multiple_choice"
    path = "cais/wmdp"
    name = "wmdp"
    subjects = ["bio", "chem", "cyber"]
    subjects_text = {"bio": "biology", "chem": "chemistry", "cyber": "cybersecurity"}
    subsets = ["test"]
    test_set = "test"
    choice_labels = ["A", "B", "C", "D"]
    eval_method = "choice_by_top_logit"
    metric = "accuracy"
    keys = ["prompt", "choices", "correct_answer"]
    eval_prompt_key = "prompt"
    eval_answer_key = "choices"
    eval_correct_key = "correct_answer"
    eval_dataset_keys = ["retain", "forget", "test", "general"]

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.synthetic_dataset_paths = [
            f"dataset/synthetic_wmdp_questions/synthetic_questions_{subject}.csv"
            for subject in self.subjects
        ]

    def download(self):
        self.dataset = DatasetDict(
            {
                "bio": load_dataset(
                    self.path, "wmdp-bio", keep_in_memory=True, trust_remote_code=True
                )["test"],
                "chem": load_dataset(
                    self.path, "wmdp-chem", keep_in_memory=True, trust_remote_code=True
                )["test"],
                "cyber": load_dataset(
                    self.path, "wmdp-cyber", keep_in_memory=True, trust_remote_code=True
                )["test"],
            }
        )

    def load_dataset_for_eval(self):
        if self.dataset is None:
            self.download()
        dataset = DatasetDict(
            {
                s: self.dataset[s].map(
                    lambda x: {
                        "prompt": WMDP.format_prompt(
                            x["question"],
                            x["choices"],
                            self.choice_labels,
                            self.subjects_text[s],
                        )
                    }
                )
                for s in self.subjects
            }
        )
        dataset = dataset.remove_columns(["question"])
        return dataset


    def load_dataset_for_train(self):
        # wmdp_dataset = self.load_dataset_for_eval()
        # bio_size, chem_size, cyber_size = [len(wmdp_dataset[s]) for s in self.subjects]
        # bio_prompts, chem_prompts, cyber_prompts = [
        #     wmdp_dataset[s]["prompt"] for s in self.subjects
        # ]
        # prompts = bio_prompts + chem_prompts + cyber_prompts
        # labels = [0] * bio_size + [1] * chem_size + [2] * cyber_size
        # forget_dataset = Dataset.from_dict({"text": prompts, "label": labels})
        # forget_dataset = forget_dataset.class_encode_column("label")

        bio_df = pd.read_csv(self.synthetic_dataset_paths[0])
        bio_df["subject"] = "bio"
        chem_df = pd.read_csv(self.synthetic_dataset_paths[1])
        chem_df["subject"] = "chem"
        cyber_df = pd.read_csv(self.synthetic_dataset_paths[2])
        cyber_df["subject"] = "cyber"

        train_forget_dfs = pd.concat(
            [bio_df, chem_df, cyber_df], ignore_index=True
        ).reset_index(drop=True)

        train_forget_prompts = []
        train_forget_responses = []
        train_forget_dfs["choices"] = train_forget_dfs["choices"].apply(
            ast.literal_eval
        )

        for _, row in train_forget_dfs.iterrows():
            prompt = WMDP.format_prompt(
                row["question"],
                row["choices"],
                self.choice_labels,
                self.subjects_text[row["subject"]],
            )
            response = WMDP.format_response(row["answer"])

            train_forget_prompts.append(prompt)
            train_forget_responses.append(response)


        train_forget_dataset = Dataset.from_dict(
            {
                "prompt": train_forget_prompts,
                # "label": [1] * len(train_forget_prompts),
                "response": train_forget_responses
            }
        )
        return train_forget_dataset

    def load_cot_dataset_for_sft(self, each_num=3):
        # file_path = "dataset/cot_prompts_responses.jsonl"
        file_path = "dataset/cot_prompts_responses_qwq_32b.jsonl"
        system_promot = "You are an expert in generating Chain-of-Thought (CoT) instructions to guide a model in responding to input queries while adhering to predefined policy constraints.\n\n"
        with open(file_path, "r") as f:
            lines = f.readlines()
        prompts = []
        responses = []
        for line in lines:
            data = json.loads(line)
            for i in range(each_num):
                prompts.append(system_promot + data['prompt'])
                responses.append(data['responses'][i].split("\n</think>\n\n")[1])
                # print(data['responses'][i].split("</think>")[1])
                # import pdb; pdb.set_trace()
        cot_dataset = Dataset.from_dict({"prompt": prompts, "response": responses})
        return cot_dataset
            
            
    @staticmethod
    def format_prompt(prompt, choice_text, choice_label, subject):
        subject = subject.replace("_", " ")
        topic_line = (
            f"The following are multiple choice questions (with answers) about "
            + f"{subject}.\n\n"
        )
        question_line = f"{prompt}\n"
        choice_lines = "\n".join(
            [f"{label}. {text}" for label, text in zip(choice_label, choice_text)]
        )
        answer_line = f"\nAnswer:"
        # answer_line = """\nPlease start your response with "Answer: {A, B, C, D}". The first generated token should belong to the set {A, B, C, D}"""
        # answer_line = """\nPlease answer the question in json format. An example is given below:\n```json\n{"answer": "A"}\n```"""
        return topic_line + question_line + choice_lines + answer_line

    @staticmethod
    def format_response(response):
        choice_map = ["A", "B", "C", "D"]

        # need to further decide
        answer_text = f"The answer is "
        return answer_text + choice_map[response]
        

class WMDPBio(WMDP):
    name = "wmdp-bio"
    subject = "biology"

    def __init__(self):
        super().__init__()

    def download(self):
        self.dataset = load_dataset(
            self.path, "wmdp-bio", keep_in_memory=True, trust_remote_code=True
        )

    def load_dataset_for_eval(
        self, split_name, load_in_batch=False, batch_size=64, prompt_prefix=""
    ):
        if self.dataset is None:
            self.download()
        dataset = self.dataset[split_name]
        dataset = dataset.map(
            lambda x: {
                "prompt": WMDP.format_prompt(
                    x["question"], x["choices"], self.choice_labels, self.subject
                )
            }
        )
        dataset = dataset.map(lambda x: {"prompt": prompt_prefix + x["prompt"]})
        dataset = dataset.map(lambda x: {"choices": self.choice_labels})
        dataset = dataset.remove_columns(["question"])
        dataset = dataset.rename_column("answer", "correct_answer")
        return self.batchify(dataset, batch_size) if load_in_batch else dataset

    def load_dataset_for_classification(self):
        raise NotImplementedError(
            f"load_dataset_for_classification not implemented for {self.__class__.__name__}"
        )


class WMDPChem(WMDP):
    name = "wmdp-chem"
    subject = "chemistry"

    def __init__(self):
        super().__init__()

    def download(self):
        self.dataset = load_dataset(
            self.path, "wmdp-chem", keep_in_memory=True, trust_remote_code=True
        )

    def load_dataset_for_eval(
        self, split_name, load_in_batch=False, batch_size=64, prompt_prefix=""
    ):
        if self.dataset is None:
            self.download()
        dataset = self.dataset[split_name]
        dataset = dataset.map(
            lambda x: {
                "prompt": WMDP.format_prompt(
                    x["question"], x["choices"], self.choice_labels, self.subject
                )
            }
        )
        dataset = dataset.map(lambda x: {"prompt": prompt_prefix + x["prompt"]})
        dataset = dataset.map(lambda x: {"choices": self.choice_labels})
        dataset = dataset.remove_columns(["question"])
        dataset = dataset.rename_column("answer", "correct_answer")
        return self.batchify(dataset, batch_size) if load_in_batch else dataset

    def load_dataset_for_classification(self):
        raise NotImplementedError(
            f"load_dataset_for_classification not implemented for {self.__class__.__name__}"
        )


class WMDPCyber(WMDP):
    name = "wmdp-cyber"
    subject = "cybersecurity"

    def __init__(self):
        super().__init__()

    def download(self):
        self.dataset = load_dataset(
            self.path, "wmdp-cyber", keep_in_memory=True, trust_remote_code=True
        )

    def load_dataset_for_eval(
        self, split_name, load_in_batch=False, batch_size=64, prompt_prefix=""
    ):
        if self.dataset is None:
            self.download()
        dataset = self.dataset[split_name]
        dataset = dataset.map(
            lambda x: {
                "prompt": WMDP.format_prompt(
                    x["question"], x["choices"], self.choice_labels, self.subject
                )
            }
        )
        dataset = dataset.map(lambda x: {"prompt": prompt_prefix + x["prompt"]})
        dataset = dataset.map(lambda x: {"choices": self.choice_labels})
        dataset = dataset.remove_columns(["question"])
        dataset = dataset.rename_column("answer", "correct_answer")
        return self.batchify(dataset, batch_size) if load_in_batch else dataset

    def load_dataset_for_classification(self):
        raise NotImplementedError(
            f"load_dataset_for_classification not implemented for {self.__class__.__name__}"
        )


mmlu_subjects = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]


class MMLU(BaseDataset):
    dataset_type = "multiple_choice"
    path = "cais/mmlu"
    name = "mmlu"
    subjects = mmlu_subjects.copy()
    subsets = ["auxiliary_train", "dev", "validation", "test"]
    test_set = "test"
    choice_labels = ["A", "B", "C", "D"]
    eval_method = "choice_by_top_logit"
    metric = "accuracy"
    keys = ["prompt", "choices", "correct_answer"]
    eval_prompt_key = "prompt"
    eval_answer_key = "choices"
    eval_correct_key = "correct_answer"

    def __init__(self):
        super().__init__()

    def download(self):
        self.dataset = load_dataset(
            self.path, "all", keep_in_memory=True, trust_remote_code=True
        )
    
    def load_dataset_for_train(self):
        if self.dataset is None:
            self.download()
        dataset = self.remove_unused_subsets()
        dataset = dataset["validation"]

        dataset = dataset.map(
            lambda x: {
                "prompt": MMLU.format_prompt(
                    x["question"], x["choices"], self.choice_labels, x["subject"]
                )
            }
        )
        prompts = dataset["prompt"]
        responses = []
        for answer in dataset["answer"]:
            responses.append(MMLU.format_response(answer))
        train_dataset = Dataset.from_dict({"prompt": prompts, "response": responses})
        return train_dataset

    def load_cot_dataset_for_sft(self):
        system_promot = "You are an expert in generating Chain-of-Thought (CoT) instructions to guide a model in responding to input queries while adhering to predefined policy constraints.\n\n"
        if self.dataset is None:
            self.download()
        dataset = self.remove_unused_subsets()
        # dataset = dataset["validation"]
        dataset = dataset["test"]
        # from the dataset randomly select 5500 samples
        dataset = dataset.shuffle(seed=42).select(range(5500))

        dataset = dataset.map(
            lambda x: {
                "prompt": MMLU.format_prompt(
                    x["question"], x["choices"], self.choice_labels, x["subject"]
                )
            }
        )
        prompts = []
        for item in dataset['prompt']:
            prompts.append(system_promot + item)
        # prompts = dataset["prompt"]
        responses = []
        for answer in dataset["answer"]:
            responses.append("```json\n{\n    \"requires_refusal\": false,\n    \"cot_instruction\": \" Just answer the question\"\n}```")
            # responses.append("<eos>")
        train_dataset = Dataset.from_dict({"prompt": prompts, "response": responses})
        return train_dataset
    

    def load_dataset_for_eval(self, split_name, load_in_batch=False, batch_size=64, prompt_prefix=""):
        if self.dataset is None:
            self.download()
        dataset = self.remove_unused_subsets()
        dataset = dataset[split_name]
        dataset = dataset.map(
            lambda x: {
                "prompt": MMLU.format_prompt(
                    x["question"], x["choices"], self.choice_labels, x["subject"]
                )
            }
        )
        dataset = dataset.map(lambda x: {"prompt": prompt_prefix + x["prompt"]})
        dataset = dataset.map(lambda x: {"choices": self.choice_labels})
        dataset = dataset.remove_columns(["question", "subject"])
        dataset = dataset.rename_column("answer", "correct_answer")
        return self.batchify(dataset, batch_size) if load_in_batch else dataset
        
    @staticmethod
    def format_prompt(prompt, choice_text, choice_label, subject):
        subject = subject.replace("_", " ")
        if subject:
            topic_line = (
                f"The following are multiple choice questions (with answers) about "
                + f"{subject}.\n\n"
            )
        else:
            topic_line = (
                f"The following are multiple choice questions (with answers).\n\n"
            )
        question_line = f"{prompt}\n"
        choice_lines = "\n".join(
            [f"{label}. {text}" for label, text in zip(choice_label, choice_text)]
        )
        answer_line = f"\nAnswer:"
        return topic_line + question_line + choice_lines + answer_line
    
    @staticmethod
    def format_response(response):
        choice_map = ["A", "B", "C", "D"]

        # need to further decide
        answer_text = f"The answer is "
        return answer_text + choice_map[response]