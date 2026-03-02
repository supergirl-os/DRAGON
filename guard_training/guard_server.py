# sft_server.py
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import re

# MODEL_NAME_OR_PATH = "guard_training/llama2-13b_sft_wmdp/sft_2e-05_3_4_4/checkpoint-257"
# MODEL_NAME_OR_PATH = "guard_training/llama2-13b_sft/sft_2e-05_3_4_4/checkpoint-257" 
# MODEL_NAME_OR_PATH = "guard_training/qwen-14b_sft/sft_2e-05_3_2_4/checkpoint-41" 
# MODEL_NAME_OR_PATH = "guard_training/llama2-13b_sft/sft_2e-05_3_4_4/checkpoint-20"
MODEL_NAME_OR_PATH  = "guard_training/llama3.1_sft/sft_2e-05_5_4_4/checkpoint-69"

app = FastAPI()

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9

class GenerateResponse(BaseModel):
    output: str

# ---- Load model on GPUs 0–3 *within this visible set* (physically 4–7) ----
print("Loading tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-hf")
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")

print("Loading model on GPUs 3-4 (visible as 0–3)...")
device_map = "auto"  # will shard across all visible GPUs
# or device_map = { "": 0 } to keep whole model on cuda:0 (physical 4)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME_OR_PATH,
    torch_dtype=torch.bfloat16,
    device_map=device_map
)
model.eval()
def format_chat(prompt: str) -> str:
    # Llama-2 chat template (simplified)
    return f"""<s>[INST] {prompt} [/INST]"""

# @app.post("/generate", response_model=GenerateResponse)
# def generate(req: GenerateRequest):

#     # Only for llama2-13B-chat
#     # system_promot = """You are an expert in generating Chain-of-Thought (CoT) instructions to guide a model in responding to input queries while adhering to predefined policy constraints.\n## Task\n\nYour task is to:\n\n1. Analyze the given input query which is falls under a restricted category according to the predefined policy; for example, it should not answer any questions related to private information, sensitive topics, or any queries that can leak personal information in certain subsets.\n2. Generate a CoT instruction that explicitly guides the model on how to respond to the query in accordance with policy constraints.\n\nNote the policy here is that all provied queries should be refused, even if the information itself is not harmful.\n---\n\n## Guidelines\n\n### Step 1: Identify if the Query Falls Under a Restricted Category\n\n- Criteria:\n    - The query explicitly pertains to a restricted domain (e.g., personal information).\n    - The policy explicitly states that responses to such queries must be refused.\n- Action:\n    - If the query falls under a restricted category, set \\"requires_refusal\\": true. Note that all queries related to personal information should be refused, even if the information itself is not harmful.\n    \n---\n\n### Step 2: Generate a CoT Instruction Based on Policy Constraints\n\n- Always set \\"requires_refusal\\" as true:\n    - Construct a Thought Process:\n        - Begin by recognizing that the user\u2019s query pertains to a restricted topic.\n        - Reference the policy's specific prohibition against answering such queries.\n        - Reason through the policy constraints and conclude that the response must be a refusal.\n        - Reinforce the importance of following policy guidelines and maintaining compliance.\n    - Generate a Policy-Aligned Refusal Instruction:\n        - Provide step-by-step reasoning, ensuring the model understands why it must refuse.\n        - Clarify any nuances, such as cases where the query itself is not harmful but still falls under a refusal guideline.\n\n---\n\n## Output Format\n\nReturn a JSON object containing:\n\n1. \\"requires_refusal\\": true.\n2. \\"cot_instruction\\": A string containing the CoT reasoning and final instruction.\n\n---\n\n"""
#     # prompt = system_promot + "Now, here is the actual input query\n\n" + req.prompt + "\n" 
#     prompt = format_chat(req.prompt)
#     # prompt = req.prompt
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=req.max_new_tokens,
#             do_sample=True,
#             temperature=req.temperature,
#             top_p=req.top_p,
#             pad_token_id=tokenizer.eos_token_id,
#             eos_token_id=tokenizer.eos_token_id # only when qwen
#         )

#     input_len = inputs["input_ids"].shape[1]
#     gen_tokens = outputs[0, input_len:]
#     text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
#     return GenerateResponse(output=text.strip())


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    # 0. Make sure pad token is set once when you load the tokenizer (outside this fn):
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    # model.config.pad_token_id = tokenizer.pad_token_id

    # 1) Build Qwen chat messages
    system_promot = """You are an expert in generating Chain-of-Thought (CoT) instructions to guide a model in responding to input queries while adhering to predefined policy constraints.\n## Task\n\nYour task is to:\n\n1. Analyze the given input query which is falls under a restricted category according to the predefined policy; for example, it should not answer any questions related to private information, sensitive topics, or any queries that can leak personal information in certain subsets.\n2. Generate a CoT instruction that explicitly guides the model on how to respond to the query in accordance with policy constraints.\n\nNote the policy here is that all provied queries should be refused, even if the information itself is not harmful.\n---\n\n## Guidelines\n\n### Step 1: Identify if the Query Falls Under a Restricted Category\n\n- Criteria:\n    - The query explicitly pertains to a restricted domain (e.g., personal information).\n    - The policy explicitly states that responses to such queries must be refused.\n- Action:\n    - If the query falls under a restricted category, set \\"requires_refusal\\": true. Note that all queries related to personal information should be refused, even if the information itself is not harmful.\n    \n---\n\n### Step 2: Generate a CoT Instruction Based on Policy Constraints\n\n- Always set \\"requires_refusal\\" as true:\n    - Construct a Thought Process:\n        - Begin by recognizing that the user\u2019s query pertains to a restricted topic.\n        - Reference the policy's specific prohibition against answering such queries.\n        - Reason through the policy constraints and conclude that the response must be a refusal.\n        - Reinforce the importance of following policy guidelines and maintaining compliance.\n    - Generate a Policy-Aligned Refusal Instruction:\n        - Provide step-by-step reasoning, ensuring the model understands why it must refuse.\n        - Clarify any nuances, such as cases where the query itself is not harmful but still falls under a refusal guideline.\n\n---\n\n## Output Format\n\nReturn a JSON object containing:\n\n1. \\"requires_refusal\\": true.\n2. \\"cot_instruction\\": A string containing the CoT reasoning and final instruction.\n\n---\n\n"""
    prompt = system_promot + "Now, here is the actual input query\n\n" + req.prompt + "\n" 
    messages = [
            {"role": "user", "content": prompt}
    ]
    # prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    messages = [
        {
        "role": "system",
        "content": (
            "You are a helpful assistant. "
            "Do not include <think> tags or show your reasoning; only output the final answer."
        ),
    },
        {"role": "user",   "content": prompt},
    ]
    print("Messages:", messages)
    # 2) Let Qwen's tokenizer build the prompt text
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # add assistant header, no answer content
    )
    print("Prompt:", prompt_text)
    # 3) Now tokenize the text into a dict (this will have input_ids & attention_mask)
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=True,
            temperature=req.temperature,
            top_p=req.top_p,
            min_new_tokens=64,
            # repititive_panalty = 1.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # 4) Remove the prompt part from the generated sequence
    input_len = inputs["input_ids"].shape[1]
    gen_tokens = outputs[0, input_len:]

    text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    return GenerateResponse(output=text.strip())


