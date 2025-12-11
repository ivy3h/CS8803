# %%
import os
os.environ["HF_DATASETS_CACHE"] = "/localscratch/mlobo6/mlv/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/localscratch/mlobo6/mlv/huggingface_cache"
os.environ["HF_HOME"] = "/localscratch/mlobo6/mlv/huggingface_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/localscratch/mlobo6/mlv/huggingface_cache"

# Make sure no pre-existing config forces ~/.cache
os.environ["HF_DATASETS_OFFLINE"] = "0"  # avoid stale metadata
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# from datasets.utils import logging as datasets_logging
# datasets_logging.set_verbosity_debug()

# from datasets import load_dataset

# dataset = load_dataset(
#     "BoKelvin/GEMeX-ThinkVG",
#     cache_dir="/localscratch/mlobo6/mlv/huggingface_cache"
# )

# print(dataset)

# %%
from huggingface_hub import login
login("")

# %%
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")

# %%
# rom transformers import AutoProcessor, Llama4ForConditionalGeneration
# import torch

# model_id = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"

# processor = AutoProcessor.from_pretrained(model_id)
# model = Llama4ForConditionalGeneration.from_pretrained(
#     model_id,
#     attn_implementation="flex_attention",
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
# )

# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_name = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="cuda:1")# ,padding_side='left')
model.generation_config = GenerationConfig.from_pretrained(model_name,device_map="auto",
                                             torch_dtype=torch.float16)
model.generation_config.pad_token_id = model.generation_config.eos_token_id


# %%
results = ['/localscratch/mlobo6/mlv/results/InternVL3_5-38B-Instruct/Fair_MedXpertQA_subset/MM/train_*.jsonl', '/localscratch/mlobo6/mlv/results/InternVL3_5-38B-Instruct/Fair_MedXpertQA_subset/MM/train_*.jsonl']

# %%
import glob
import json

# %%
prompt_template = """
<|im_start|>system
You are performing the task of "LLM as a Judge" in the medical domain.
You are given:
- A question: {question}
- Answer choices: {choices}
- The correct answer: {label}
- The LLM's generated response: {response}
- The LLM's selected answer choice: {selected}
Your task is to categorize the type of error made by the LLM. 
Choose exactly one of the following categories:
1. Reasoning Process Error:
   The rationale contains incorrect reasoning steps that directly lead to the wrong answer.
2. Question Understanding Error:
   The LLM misunderstood the question, misinterpreted instructions, or answered a different question.
3. Lack of Medical Knowledge:
   The LLM lacked the necessary medical knowledge to derive the correct answer, even if reasoning structure was correct.
4. Formatting Error:
   The content is correct, but formatting mistakes prevented identifying the correct answer (e.g., wrong label format, missing choice letter).
OUTPUT REQUIREMENTS:
- Return ONLY valid JSON.
- Use exactly one key: judge
- The value must be exactly one of the four category names (verbatim).
Output format:
Always return a valid JSON with 1 key and corresponding value: 
judge: category name you generated 
Do not return the original prompt, do not include any explanations, no prefixes and no additional or surrounding text
<|im_end|>
<|im_start|>user
Please execute as instructed.
<|im_end|>
<|im_start|>assistant
"""


# %%
import json

# %%
prompts = {}

# for entry in results:
#     train_files = glob.glob(entry)

for entry2 in ['/localscratch/mlobo6/mlv/results/InternVL3_5-38B-Instruct/Fair_MedXpertQA_subset/MM/train_all_results.jsonl', '/localscratch/mlobo6/mlv/results/Qwen2.5-VL-72B-Instruct-AWQ/Fair_MedXpertQA_subset/MM/train_all_results.jsonl']:
    if '_LLMAsJudge' not in entry2: 
        prompts[entry2] = []
        with open(entry2) as f:
            for line in f:
                if line.strip():
                    tmpa = json.loads(line)
                    if tmpa['correct'] == True:
                        continue
                    else:
                        par_json = {
                            "question": tmpa['question'],
                            "choices": tmpa['options'],
                            "label": tmpa['label'],
                            "response": tmpa['response'],
                            "selected": tmpa['prediction']
                        }
                        prompt = prompt_template.format(**par_json)

                        prompts[entry2].append(prompt)
                        # meta.append(par_json)
            


# %%
BATCH_SIZE = 8
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
# OUTPUT_PATH = "GEMeX-VQA-ThinkVG-augmented_core.jsonl"
GLOBAL_SEED = 42

# %%
import json
import random
import torch
from tqdm import tqdm
import re

# %%
@torch.no_grad()
def generate_batch(prompts, model, tokenizer, temperature, max_new_tokens):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side='left', truncation=True).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


# def parse_json_safe(text):
#     try:
#         json_start = text.find("{")
#         json_end = text.rfind("}") + 1
#         if json_start == -1 or json_end == 0:
#             return None
#         return json.loads(text[json_start:json_end])
#     except Exception:
#         return None

# import json
# import re

def parse_json_safe(text):
    try:
        # We only care about JSON that contains "judge"
        # Find all JSON-like blocks: { ... }
        matches = re.findall(r"judge", text, flags=re.DOTALL)

        for block in matches:
            if "judge" in block:  # Only accept blocks containing the key we want
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue  # Try the next one if this one fails

        return None

    except Exception:
        return None
# ------------------------
# MAIN AUGMENTATION LOOP
# ------------------------

judge_all = {}
for entry in  prompts: # 2988 from dataset 1 
    print(entry)
    judge_all[entry] = []
    with open(entry + '_LLMAsJudge_deepseek.jsonl', "w") as f:
        subset = random.sample(prompts[entry], 1000)
        n = len(subset)
        # n = len(prompts[entry])
        for start in tqdm(range(0, n, BATCH_SIZE), desc="Augmenting dataset"):
            batch = subset[start:start + BATCH_SIZE]
            # batch_list = [dict(zip(batch.keys(), t)) for t in zip(*batch.values())]

            decoded_texts = generate_batch(batch, model, tokenizer, TEMPERATURE, MAX_NEW_TOKENS)

            # print(decoded_texts)
        
        # judge_pipe = pipeline(
        #     "text-generation",
        #     model=model,
        #     tokenizer=tokenizer,# padding_side='left', 
        #     # device='cuda:1',      # GPU id
        #     batch_size=BATCH_SIZE,
        #     max_new_tokens=MAX_NEW_TOKENS,
        # )
        # decoded = judge_pipe(subset)

        # for text in decoded:
        #     new_ex = parse_json_safe(text)
        #     if new_ex:
        #         # merged = {**prompts[entry], **new_ex}
        #         f.write(json.dumps(new_ex) + "\n")
        #         # if 'judge' in new_ex:
        #         #     judge_all[entry].append(new_ex['judge'])
        #         # elif 'judgement' in new_ex:
        #         #     judge_all[entry].append(new_ex['judgement'])
                


            for text in decoded_texts:
                # print(text)
                # new_ex = parse_json_safe(text)
                # print(new_ex)
                lines_with_judge = [line for line in text.splitlines() if ("judge" in line) or ('Judge' in line)]
                # print(lines_with_judge)
                if len(lines_with_judge) == 4:
                    # merged = {**prompts[entry], **new_ex}
                    f.write(json.dumps(lines_with_judge[3].strip()) + "\n")
                    # if 'judge' in new_ex:
                    #     judge_all[entry].append(new_ex['judge'])
                    # elif 'judgement' in new_ex:
                    #     judge_all[entry].append(new_ex['judgement'])
                    



# %%



