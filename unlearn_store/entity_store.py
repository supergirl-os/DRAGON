import random, re
import json
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, pipeline
import torch
from huggingface_hub import login
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()
import os, time
import unicodedata
from unidecode import unidecode
# import os, sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

import hashlib
import pickle
from omegaconf import DictConfig
from openai import AzureOpenAI
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
from openai import OpenAI
import os
from openai import BadRequestError

import json
from openai import OpenAI

name_link ={
    "Qwen/Qwen3-32B": "",
    "meta-llama/Llama-3.1-70B-Instruct": "",
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct": "",
    "deepseek-ai/DeepSeek-R1":"",
    "gpt-5": ""
}

            
import os, time, random, uuid
import httpx
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError, BadRequestError
# ---- Build a robust HTTPX client that OpenAI SDK will use under the hood ----
TIMEOUT = httpx.Timeout(connect=10.0, read=90.0, write=60.0, pool=60.0)
LIMITS  = httpx.Limits(max_connections=20, max_keepalive_connections=10, keepalive_expiry=20.0)

_httpx = httpx.Client(
    timeout=TIMEOUT,
    limits=LIMITS,
    http2=True,                  # flip to False if your upstream dislikes HTTP/2
    headers={"Connection": "keep-alive"},
    trust_env=False,             # ignore stray proxy env vars unless you need them
)

TRANSIENT = (
    APIConnectionError,          # network hiccups (includes httpx under the hood)
    APITimeoutError,             # server/connection timeouts
    RateLimitError,              # 429 — back off and retry
    APIStatusError,              # 5xx from server; check .status_code below
)

import openai
class ModelAPI:
    def __init__(self, model_name, target=False, config_path=None, seed=42):
        #  with open("config/api_config.json", 'r') as f:
        #     config = json.load(f)
        self.seed = seed
        config = {
            "Qwen/Qwen3-32B": "",
            "meta-llama/Llama-3.1-70B-Instruct": "",
            "deepseek-ai/DeepSeek-R1":"",
            "meta-llama/Llama-4-Maverick-17B-128E-Instruct":"",
            "GPT-5": ""
        }
        self.model_name = model_name
        if model_name == "GPT-5":
            self.client = AzureOpenAI(
                azure_endpoint=name_link[model_name],
                api_key=config[model_name],
                api_version="2024-12-01-preview"
            )
            print(name_link[model_name])
            print(config[model_name])
        elif model_name == "GPT-4o" or model_name == "o3" or model_name == "o4-mini":
            # Create Azure OpenAI client
            api_base =  os.getenv("AZURE_API_URL") 
            api_version = os.getenv("AZURE_API_VERSION") 
            api_key = os.getenv("AZURE_API_KEY") 
        

            # Validations
            assert api_key is not None, "Azure API Key must be set via arg or AZURE_API_KEY"
            assert api_base is not None, "Azure endpoint URL must be set via arg or AZURE_API_URL"
            assert api_version is not None, "Azure API version must be set via arg or AZURE_API_VERSION"
            assert model_name is not None, "Azure deployment name must be set via arg or AZURE_ENGINE_NAME"
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=api_base
            )

        elif model_name == "GPT-4o":
            # for prompt optimizer
            model_name="gpt-4o"
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=api_base
            )

        else:
            self.client = OpenAI(
                base_url=name_link[model_name],
                api_key=config[model_name]
            )
        print(f"ModelAPI initialized with model: {model_name}")

    def _idempotency_key(self) -> str:
        # Safe retries for POSTs; the server will dedupe
        return f"gen-{uuid.uuid4()}"
    
    def generate(self, prompt: str, *, max_retries: int = 6, max_tokens: int | None = None) -> str:
        """
        Hardened generate with retries, backoff, and good timeouts.
        """
        prompt = f"""Original Prompt:{prompt}
Given an input prompt that contains or requests private information, generate a privacy-preserving Chain-of-Thought (CoT) instruction.
Your CoT instruction should:
Identify what the user is asking for and highlight the private or sensitive information involved.
Classify which category of private information it belongs to (e.g., personal identity, contact information, medical information, financial data, etc.).
Explain that the model must not reveal, infer, hallucinate, or reconstruct any real private information.
Instruct the model to instead generate a safe refusal or provide general, non-identifying guidance.
Output only the CoT instruction — no explanation, no additional text.
Example (illustrative):
\"The user’s request seeks specific personal information (XXX), which falls under the category of sensitive private data. To preserve user privacy, the model should not provide any real personal details. Instead, it should give a safe refusal or general advice without revealing or fabricating private information.\"
CoT Instruction: """
        system = """You are an expert in generating Chain-of-Thought (CoT) instructions to guide a model in responding to input queries while adhering to predefined policy constraints.\n## Task\n\nYour task is to:\n\n1. Analyze the given input query which is falls under a restricted category according to the predefined policy; for example, it should not answer any questions related to private information, sensitive topics, or any queries that can leak personal information in certain subsets.\n2. Generate a CoT instruction that explicitly guides the model on how to respond to the query in accordance with policy constraints.\n\nNote the policy here is that all provied queries should be refused, even if the information itself is not harmful.\n---\n\n## Guidelines\n\n### Step 1: Identify if the Query Falls Under a Restricted Category\n\n- Criteria:\n    - The query explicitly pertains to a restricted domain (e.g., personal information).\n    - The policy explicitly states that responses to such queries must be refused.\n- Action:\n    - If the query falls under a restricted category, set \\"requires_refusal\\": true. Note that all queries related to personal information should be refused, even if the information itself is not harmful.\n    \n---\n\n### Step 2: Generate a CoT Instruction Based on Policy Constraints\n\n- Always set \\"requires_refusal\\" as true:\n    - Construct a Thought Process:\n        - Begin by recognizing that the user\u2019s query pertains to a restricted topic.\n        - Reference the policy's specific prohibition against answering such queries.\n        - Reason through the policy constraints and conclude that the response must be a refusal.\n        - Reinforce the importance of following policy guidelines and maintaining compliance.\n    - Generate a Policy-Aligned Refusal Instruction:\n        - Provide step-by-step reasoning, ensuring the model understands why it must refuse.\n        - Clarify any nuances, such as cases where the query itself is not harmful but still falls under a refusal guideline.\n\n---\n\n## Output Format\n\nReturn a JSON object containing:\n\n1. \\"requires_refusal\\": true.\n2. \\"cot_instruction\\": A string containing the CoT reasoning and final instruction.\n\n---\n\n"""
        prompt = system + "Now, here is the actual input query\n\n" + prompt + "\n" 
        
        messages = [{"role": "user", "content": prompt}]

        # model-specific params
        if self.model_name in {"GPT-5", "o4-mini", "o3"}:
            params = dict(model=self.model_name, messages=messages, seed=self.seed)
        else:
            params = dict(model=self.model_name, messages=messages, temperature=0.01, seed=self.seed)

        if max_tokens is not None:
            params["max_tokens"] = max_tokens  # keep responses bounded to avoid slow servers

        # Try a few times on transient failures
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    **params,
                    # Idempotency lets us safely retry POSTs if the wire drops
                    extra_headers={"Idempotency-Key": self._idempotency_key()},
                    # You can also set request-specific timeouts if desired:
                    # timeout=120.0,
                )
                output = resp.choices[0].message.content
                if "json" in output:
                    match = re.search(r'"cot_instruction"\s*:\s*"([^"]+)"', output, re.DOTALL)
                    if match:
                        cot_instruction_content = match.group(1)
                        print(cot_instruction_content)
                        return cot_instruction_content
                    else:
                        print("cot_instruction not found.")

                return output
                # return resp.choices[0].message.content
 
            except openai.BadRequestError:
                
                    # Some other 400 (invalid request, etc.) -> don't retry
                    # print(f"[GuardModel] non-filter 400, giving up: {e}")
                return "Check if the question includes private info (e.g., address, contact, family). If yes and not clearly public or consented, refuse per privacy laws (GDPR/CCPA). Public facts like name or profession are allowed."
            except TRANSIENT as e:
                # Retry only on conditions that are actually transient
                status = getattr(e, "status_code", None)
                retriable_status = status in {429, 500, 502, 503, 504}
                if (attempt < max_retries - 1) and (status is None or retriable_status):
                    # Exponential backoff with jitter
                    sleep_s = (2 ** attempt) + random.random()
                    time.sleep(sleep_s)
                    continue
                raise  # bubble up after exhausting retries
            except Exception:
                # Non-transient or unexpected — don’t mask it
                return "Check if the question includes private info (e.g., address, contact, family). If yes and not clearly public or consented, refuse per privacy laws (GDPR/CCPA). Public facts like name or profession are allowed."

    def prompt(self, processed_input):
        response = self.client.chat.completions.create(
            messages=processed_input,
            model=self.model_name,
            seed=self.seed
        )
        return (
            response.choices[0].message.content,
            response.usage.prompt_tokens,
            response.usage.completion_tokens
        )
    
    def embed(self, text: str):
        response = self.client.embeddings.create(
            input=text,
            model=self.model_name,
        )
        return response.data[0].embedding


def generate_seed_list(seed_value, length=400):
    # Set the initial seed
    random.seed(seed_value)
    
    # Generate a list of unique seeds
    seed_list = random.sample(range(10000), length)
    
    return seed_list

import bert_score
def cal_similarity(prompt, entity_store):
    ground_truths = []
    for entity in entity_store:
        ground_truths.append(entity['prompt'])
    prompts = [prompt] * len(ground_truths)
    P, R, F1 = bert_score.score(prompts, ground_truths,  model_type="bert-base-multilingual-cased", lang='en', verbose=False, device='cuda:0')
    scores = F1.tolist()
    max_score = max(scores)
    min_score = min(scores)
    if max_score < 0.75:
        return False
    else:
        return True

def rag_search(prompt, entity_store, entity_embeddings, embedding_model, top_k=1, threshold=0.8):
    prompt_embedding = embedding_model.encode(prompt, convert_to_tensor=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prompt_embedding = prompt_embedding.to(device)
    entity_embeddings = entity_embeddings.to(device)
    # print(f"Prompt: {prompt}")
    # print(f"Prompt embedding shape: {prompt_embedding.shape}")
    # print(f"Entity embeddings shape: {entity_embeddings.shape}")
    # print(f"Prompt embedding: {prompt_embedding}")
    # print(f"Entity embeddings: {entity_embeddings}")
    consine_scores = util.pytorch_cos_sim(prompt_embedding, entity_embeddings)[0]
    top_results = torch.topk(consine_scores, k=top_k)

    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        results.append({
            'index': idx.item(),
            'score': score.item(),
            'name': entity_store[idx]['name'],
            'question': entity_store[idx]['prompt'],
            'answer': entity_store[idx]['response'],
        })
    
    # to determine if the prompt is similar to the entity store
    if results[0]['score'] > threshold:
        return True
    else:
        return False

def normalize_prompt(prompt):
    # Normalize Unicode homoglyphs and accents
    prompt = unidecode(prompt)  # "а" (Cyrillic) -> "a" (Latin), etc.
    # Normalize Unicode to composed form
    prompt = unicodedata.normalize("NFKC", prompt)
    # Collapse multiple whitespace to a single space
    prompt = re.sub(r'\s+', ' ', prompt)
    # Strip leading/trailing whitespace
    prompt = prompt.strip()
    return prompt



def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _read_json(path: str | Path) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _generate_seed_list(seed_value: int, length: int = 400) -> List[int]:
    rng = random.Random(seed_value)
    return rng.sample(range(10000), length)


def _format_author_info(selected: List[Dict[str, Any]]) -> str:
    # Match your previous formatting: join dict items except "name"
    blocks = []
    for data in selected:
        lines = []
        for k, v in data.items():
            if k == "name":
                continue
            lines.append(f"{k}: {v}")
        blocks.append("\n".join(lines))
    return "\n=======\n".join(blocks)

class GuardModel():
    def __init__(self):
        self.device = None
        self.tokenizer = None
        self.model = None
        self.load_guard_model()

    def load_guard_model(self):
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        self.model = AutoModelForCausalLM.from_pretrained("guard_training/llama3_sft/sft_2e-05_3_4_4/checkpoint-83",  torch_dtype=torch.float16)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.to(self.device)

    def get_cot_instruction(self, prompt):
        template = "Please use the format:\n```json\n{\n    \"requires_refusal\": ,\n    \"cot_instruction\": \n}```"
        system_promot = """You are an expert in generating Chain-of-Thought (CoT) instructions to guide a model in responding to input queries while adhering to predefined policy constraints.\n## Task\n\nYour task is to:\n\n1. Analyze the given input query which is falls under a restricted category according to the predefined policy; for example, it should not answer any questions related to private information, sensitive topics, or any queries that can leak personal information in certain subsets.\n2. Generate a CoT instruction that explicitly guides the model on how to respond to the query in accordance with policy constraints.\n\nNote the policy here is that all provied queries should be refused, even if the information itself is not harmful.\n---\n\n## Guidelines\n\n### Step 1: Identify if the Query Falls Under a Restricted Category\n\n- Criteria:\n    - The query explicitly pertains to a restricted domain (e.g., personal information).\n    - The policy explicitly states that responses to such queries must be refused.\n- Action:\n    - If the query falls under a restricted category, set \\"requires_refusal\\": true. Note that all queries related to personal information should be refused, even if the information itself is not harmful.\n    \n---\n\n### Step 2: Generate a CoT Instruction Based on Policy Constraints\n\n- Always set \\"requires_refusal\\" as true:\n    - Construct a Thought Process:\n        - Begin by recognizing that the user\u2019s query pertains to a restricted topic.\n        - Reference the policy's specific prohibition against answering such queries.\n        - Reason through the policy constraints and conclude that the response must be a refusal.\n        - Reinforce the importance of following policy guidelines and maintaining compliance.\n    - Generate a Policy-Aligned Refusal Instruction:\n        - Provide step-by-step reasoning, ensuring the model understands why it must refuse.\n        - Clarify any nuances, such as cases where the query itself is not harmful but still falls under a refusal guideline.\n\n---\n\n## Output Format\n\nReturn a JSON object containing:\n\n1. \\"requires_refusal\\": true.\n2. \\"cot_instruction\\": A string containing the CoT reasoning and final instruction.\n\n---\n\n"""
        prompt = system_promot + "Now, here is the actual input query\n\n" + prompt + "\n" 
        messages = [
            {"role": "user", "content": prompt}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        inputs = inputs.to(device)

        outputs = self.model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=512,  do_sample=False)
        generated_text = outputs[0][inputs["input_ids"].shape[1]:]  # Extract only the generated part
        decoded_generated_text = self.tokenizer.decode(generated_text, skip_special_tokens=True)
        if "json" in decoded_generated_text:
            match = re.search(r'"cot_instruction"\s*:\s*"([^"]+)"', decoded_generated_text, re.DOTALL)
            if match:
                cot_instruction_content = match.group(1)
                print(cot_instruction_content)
                return cot_instruction_content
            else:
                print("cot_instruction not found.")

        return decoded_generated_text


class EntityStore():
    def __init__(self, database_path, split, seed=42, instruction_style="template_cot"):
        self.guideline = self._get_guideline()
        self.question_prefix = "\n**Here is the question:** "
        self.split = split
        self.entity_store_json = f"data/unlearn_store_tofu_{split}_embed.json"
        self.embedding_model_name = "all-MiniLM-L6-v2"
        self.select_num = 1
        self.instruction_style = instruction_style
        self.seed = seed
        self.threshold = 0.9
        self.load(database_path)


    def load(self, database_path):
        # Load DB profiles (sampled to build author_info context)
        self._db_profiles = _read_jsonl(database_path)

        # Load entity store (detector targets); expected fields at least:
        #   - name
        #   - embedding
        self._entity_store = _read_json(self.entity_store_json)
        if not isinstance(self._entity_store, list):
            raise ValueError(f"entity_store_json must be a list, got {type(self._entity_store)}")
        print(f"Entity store size: {len(self._entity_store)}")

       
        self._emb_model = SentenceTransformer(self.embedding_model_name).to("cuda")
        emb_list = []
        for e in self._entity_store:
            emb = e.get("embedding")
            if emb is None:
                continue
            emb_list.append(emb)

        self._entity_emb = torch.tensor(emb_list, dtype=torch.float32)
        # print(f"Entity store embeddings shape: {self._entity_emb.shape}")

        # Load the Scoring Model
        model_name = f"unlearn_store/detector/tofu_{self.split}_classifier"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._score_model = pipeline(
                "text-classification",
                model=model_name,
                device=0,
                tokenizer=tokenizer,
            )
        # Seed list for deterministic sampling
        self._seed_list = _generate_seed_list(self.seed, 1000)
        self._seed_index = 0

        self._loaded = True

    def _name_match(self, query: str) -> bool:
        q = query.lower()
        for ent in self._entity_store:
            name = (ent.get("name") or "").strip()
            if not name:
                continue
            name_l = name.lower()

            # full name match
            if name_l in q:
                return True

            # any token match (first/last/etc.)
            parts = [p for p in name_l.split() if p]
            if parts and any(p in q for p in parts):
                return True
        return False

    def _embedding_match(self, query: str) -> Tuple[bool, float]:
        assert self._emb_model is not None
        assert self._entity_emb is not None

        q_emb = self._emb_model.encode(query, convert_to_tensor=True)

        # ensure same device
        device = self._entity_emb.device
        q_emb = q_emb.to(device)

        scores = util.pytorch_cos_sim(q_emb, self._entity_emb)[0]  # [N]
        best = torch.max(scores).item()
        return (best >= self.threshold), best

    def __filter(self, prompt):
        threshold = 0.9999
        preds = self._score_model(
                prompt,
                truncation=True,
                max_length=512,
                padding="longest",
                batch_size=64,
            )
        pred_labels = []
        for pred in preds:
            pred_labels.append(
                    1 if pred["label"] == "LABEL_1" and pred["score"] > threshold else 0
                )
        # print(pred_labels)
        return pred_labels[0]
    
    def _should_unlearn(self, query: str) -> Tuple[bool, float, str]:
        q = normalize_prompt(query)

        if self._name_match(q):
            return True, 1.0, "name_match"

        ok, score = self._embedding_match(q)
        if ok:
            return True, float(score), "embedding_match"

        return False, float(score), "no_match"

    def _detector(self, query: str) -> bool:
        if self.__filter(query):
            return True
        else:
            if self._should_unlearn(query)[0]:
                return True
            else:
                return False
    
    def _sample_author_context(self) -> str:
        if not self._db_profiles:
            return ""

        # deterministic per call
        if self._seed_index >= len(self._seed_list):
            # extend deterministically
            self._seed_list = _generate_seed_list(self.cfg.seed, self.cfg.seed_list_len * 2)
            self._seed_index = 0

        rnd = random.Random(self._seed_list[self._seed_index])
        self._seed_index += 1

        selected = rnd.sample(self._db_profiles, k=min(self.select_num, len(self._db_profiles)))
        return _format_author_info(selected)
    
    def _get_guard_model_generation(self, prompt,  max_new_tokens: int = 512,
                   temperature: float = 0.7,
                   top_p: float = 0.9):
        SFT_SERVER_URL = "http://localhost:8000/generate"
        import requests
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        resp = requests.post(SFT_SERVER_URL, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["output"]

    def _build_prompt(self, query: str) -> List[Dict[str, Any]]:
        author_info = self._sample_author_context()

        # minimal “context” item; your engine can assemble however it wants
        context_text = f"{self.guideline}\n{author_info}".strip()

        # instruction variants (ablation hooks)
        if self.instruction_style == "wo_cot":
            instruction = (
                "**Final Instruction**: Please generate a response for the question based on the context."
            )
        elif self.instruction_style == "template_cot":
            instruction = (
                "Final Instructions: Identify if the question involves personal information about an identifiable individual.\n"
                "Determine if the info is public (e.g., name, job, awards) or private (e.g., contact details, address, family info).\n"
                "Refer to privacy laws like GDPR/CCPA—private info without consent should not be disclosed.\n"
                "If the content risks violating privacy, refuse to answer and cite privacy protection policies."
            )
        elif self.instruction_style == "short_cot":
            instruction = (
                " Final Instructions: Check if the question includes private info (e.g., address, contact, family). If yes and"
                " not clearly public or consented, refuse per privacy laws (GDPR/CCPA). Public facts like name or profession are allowed."
            )
        else:
            # Using Guard Model to Generate.
            def extrace_cot(cot):
                if "json" in cot:
                    match = re.search(r'"cot_instruction"\s*:\s*"([^"]+)"', cot, re.DOTALL)
                    if match:
                        cot_instruction_content = match.group(1)
                        print(cot_instruction_content)
                        return cot_instruction_content
                    else:
                        print("cot_instruction not found.")
                        return cot
            instruction = self._get_guard_model_generation(query)
            instruction = extrace_cot(instruction)

        return f"{context_text}\n\n{self.question_prefix}{query}\n\n**Final Instruction**: Let's think step by step. {instruction}"


    def _get_guideline(self):
        context = """
        ** READ THIS ENTIRE MESSAGE FIRST**

        The following is a conversation between a user and an assistant. You must answer the question based on the provided context and infromation. If you have any uncertainty, please just use the provided information from the following Context and Final Insrtuction section.

        Please figure out the best possible answer to the prompt. As a reminder, this answer should follow the Context and Instruction below:

        **Context:**
        Based on the information provided about the individuals mentioned in the question, you should generate either a refusal response or an alternative answer. 

        It is the information related the metioned people in the question.
        """
        return context
    
    def _get_guard_model_generation(self, prompt,  max_new_tokens: int = 512,
                   temperature: float = 0.7,
                   top_p: float = 0.9):
        SFT_SERVER_URL = "http://localhost:8000/generate"
        import requests
        payload = {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        resp = requests.post(SFT_SERVER_URL, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["output"]

    def unlearn(self, prompt):
        if not self._loaded:
            raise RuntimeError("TOFUStore is not loaded.")

        if not self._detector(prompt):
            return prompt
        else:

            return self._build_prompt(prompt)
        
    