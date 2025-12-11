# %%
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
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
from huggingface_hub import login
import torch
login("")
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
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", padding_side='left')
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B",
                                             device_map="auto",
                                             torch_dtype=torch.float16)

# %%

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
Only return 1, 2, 3, or 4 corresponding to the 4 category options above.
Do not return the original prompt, do not include any explanations, no prefixes and no additional or surrounding text
<|im_end|>
<|im_start|>user
Please execute as instructed.
<|im_end|>
<|im_start|>assistant
Return only a single character: 1, 2, 3, or 4, according to the category defined above.
"""



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
from transformers import pipeline
# %%
tokenizer.pad_token = tokenizer.eos_token

# %%
@torch.no_grad()
def generate_batch(prompts, model, tokenizer, temperature, max_new_tokens):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, padding_side='left', truncation=True).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True, use_cache=True
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
        matches = re.findall(r"\{.*?\}", text, flags=re.DOTALL)

        for block in matches:
            if "judge" in block:  # Only accept blocks containing the key we want
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue  # Try the next one if this one fails

        return None

    except Exception:
        return None
    
def extract_final_judgment(text):
    # Find all occurrences of a line containing ONLY a single digit 1–4
    matches = re.findall(r'^\s*([1-4])\s*$', text, flags=re.MULTILINE)
    if matches:
        return matches[-1]  # return the final one
    return None
# ------------------------
# MAIN AUGMENTATION LOOP
# ------------------------
import random 
judge_all = {}
for entry in  prompts: # 2988 from dataset 1 
    print(entry)
    judge_all[entry] = []
    with open(entry + '_LLMAsJudge_llama.jsonl', "w") as f:
        # my_list = prompts[entry]
        #
        subset = random.sample(prompts[entry], 2000)
        n = len(subset)

        

        # judge_pipe = pipeline(
        #     "text-generation",
        #     model=model,
        #     tokenizer=tokenizer,# padding_side='left', 
        #     # device='cuda:1',      # GPU id
        #     batch_size=BATCH_SIZE,
        #     max_new_tokens=MAX_NEW_TOKENS,
        # )

        # decoded = judge_pipe(subset)

        for start in tqdm(range(0, n, BATCH_SIZE), desc="Augmenting dataset"):
            batch = subset[start:start + BATCH_SIZE]
            # batch_list = [dict(zip(batch.keys(), t)) for t in zip(*batch.values())]

            decoded_texts = generate_batch(batch, model, tokenizer, TEMPERATURE, MAX_NEW_TOKENS)

            # print(decoded_texts)
            # print(decoded)

            for text in decoded_texts:
                new_ex = extract_final_judgment(text)
                if new_ex:
                    # merged = {**prompts[entry], **new_ex}
                    f.write(json.dumps(new_ex) + "\n")
                    # if 'judge' in new_ex:
                    #     judge_all[entry].append(new_ex['judge'])
                    # elif 'judgement' in new_ex:
                    #     judge_all[entry].append(new_ex['judgement'])
                
        # for text in decoded:
        #         # print(text)
        #         # new_ex = parse_json_safe(text)
        #         # print(new_ex)
        #         lines_with_judge = [line for line in text.splitlines() if ("judge" in line) or ('Judge' in line)]
        #         # print(lines_with_judge)
        #         f.write(':'.join(lines_with_judge) + "\n")
        #             # if 'judge' in new_ex:
        #             #     judge_all[entry].append(new_ex['judge'])
        #             # elif 'judgement' in new_ex:
        #             #     judge_all[entry].append(new_ex['judgement'])
                    



# %%


# %%





# %%



