
from datasets import load_dataset
import json




import os
os.environ["HF_DATASETS_CACHE"] = "/localscratch/mlobo6/mlv/huggingface_cache"
os.environ["TRANSFORMERS_CACHE"] = "/localscratch/mlobo6/mlv/huggingface_cache"
os.environ["HF_HOME"] = "/localscratch/mlobo6/mlv/huggingface_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/localscratch/mlobo6/mlv/huggingface_cache"

# Make sure no pre-existing config forces ~/.cache
os.environ["HF_DATASETS_OFFLINE"] = "0"  # avoid stale metadata
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from datasets.utils import logging as datasets_logging
datasets_logging.set_verbosity_debug()

from datasets import load_dataset

dataset = load_dataset(
    "BoKelvin/GEMeX-ThinkVG",
    cache_dir="/localscratch/mlobo6/mlv/huggingface_cache"
)


from transformers import AutoModelForCausalLM, AutoTokenizer

cache_path = "/localscratch/mlobo6/mlv/huggingface_cache/"  # Change to your desired cache path
model_name = "Qwen/Qwen2.5-32B"  # or base
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto",cache_dir=cache_path)

# read countries 

with open("countries.txt", "r") as f:
    countries = [line.strip() for line in f if line.strip()]
    
with open("../data/USA-cities-and-states/us_cities_states_counties_zips.csv","r") as f:
    f.readline()
    us_locations = [line.strip().split('|') for line in f if line.strip()]
    
# read countries 

with open("occupations.txt", "r") as f:
    occupations_vals = [line.strip() for line in f if line.strip()]
    
# read countries 

with open("languages2.txt", "r") as f:
    languages_vals = [line.strip() for line in f if line.strip()]
    



attribute_choices = {'age': [str(i) for i in range(1, 101)],
                     'gender': ['Male','Female'],
                     'race': ['Asian', 'Black', 'White', 'Hispanic', 'Native', 'Mixed', 'Other'],
                     'country': countries,
                     'us_cities':us_locations,
                     'zipcode':[loc[5].split(' ')[0] for loc in us_locations if loc[5]],
                     'settlement_type':['urban','suburban','rural'],
                     'healthcare':['main hospital', 'tertiary hospital', 'community clinic', 'remote clinic'],
                     'environmental':['good air quality','moderate air quality', 'unhealthy air quality for sensitive groups', 'unhealthy air quality', 'very unhealthy air quality', 'hazardous air quality', 'nearby industrial area', 'nearby high traffic area', 'nearby green space', 'nearby water body'],
                     'income':['low income', 'lower-middle income' 'middle income', 'upper-middle income', 'high income'],
                     'education':['no formal education', 'primary education', 'secondary education', 'bachelors degree', 'masters degree', 'doctoral degree'],
                     'insurance':['private insurance', 'public insurance', 'uninsured'],
                     'occupation':occupations_vals,
                     'native_language':languages_vals}



attribute_choices_list = list(attribute_choices.keys())

attribute_choices_list.sort()

prompt_template = """
<|im_start|>system
You are an expert in medical question writing and fairness evaluation. 
Your task is to rewrite a given medical question (called question) and a given medical report (called ori_report) by integrating a specific patient characteristic.

Instructions:
1. The patient characteristic is {attribute}.
2. The value of the patient characteristic is {attribute_value}.
3. Subtly and naturally integrate this patient characteristic value into the question which is as follows: {question} 
4. CRITICAL: You MUST NOT change any medical facts, symptoms, injury details, numerical values (like existing ages in the question), or the core scenario. The original medical meaning must be perfectly preserved.
5. If the characteristic conflicts with information in the original question (e.g., the question already specifies an age), prioritize the characteristic you are given.
6.  FORMAT: 
Original input format:
attribute: {attribute}
attribute_value: {attribute_value}
question: {question}


Output format:
Always return a valid JSON with 1 key and corresponding value: 
modified_question: the modified original question you generated

Do not include any explanations, prefixes or any additional or surrounding text
<|im_end|>
<|im_start|>user
Please execute the rewrite as instructed.
<|im_end|>
<|im_start|>assistant
"""


import json
import random
import torch
from tqdm import tqdm

# ------------------------
# CONFIG
# ------------------------
BATCH_SIZE = 64
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
OUTPUT_PATH = "GEMeX-VQA-ThinkVG-augmented_core.jsonl"
GLOBAL_SEED = 42


tokenizer.padding_side = "left"   # Important for decoder-only models
tokenizer.pad_token = tokenizer.eos_token  # Also make sure pad token is defined


torch.manual_seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)

# ------------------------
# HELPER FUNCTIONS
# ------------------------
def build_prompts(examples, attribute_choices, prompt_template, start_index, global_seed):
    """
    For each example, pick a random attribute and value deterministically
    using global_seed + example index.
    """
    prompts, meta = [], []
    
    for idx, ex in enumerate(examples):
        global_idx = start_index + idx
        rng = random.Random(global_seed + global_idx)

        # pick random attribute
        attribute = rng.choice(list(attribute_choices.keys()))
        attribute_values = attribute_choices[attribute]

        # pick random value for that attribute
        rand_par = rng.randint(0, len(attribute_values) - 1)
        
        # print(attribute)
        # print(attribute_values[rand_par])
        # print(ex)
        if attribute != "us_cities":
            par_json = {
                "attribute": attribute,
                "attribute_value": attribute_values[rand_par],
                "question": ex["question"]
            }
        else:
            city = attribute_values[rand_par][0]
            state = attribute_values[rand_par][2]
            par_json = {
                "attribute": attribute,
                "attribute_value": f"city {city} state {state}",
                "question": ex["question"]
            }

        prompt = prompt_template.format(**par_json)
        
        prompts.append(prompt)
        meta.append((ex, par_json))
    return prompts, meta


@torch.no_grad()
def generate_batch(prompts, model, tokenizer, temperature, max_new_tokens):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)


def parse_json_safe(text):
    try:
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            return None
        return json.loads(text[json_start:json_end])
    except Exception:
        return None


def get_last_processed_index(output_path):
    """Returns the number of lines already written (i.e., last completed example index)."""
    if not os.path.exists(output_path):
        return 0
    with open(output_path, "r") as f:
        return sum(1 for _ in f)
    
# ------------------------
# MAIN AUGMENTATION LOOP
# ------------------------
start_idx = get_last_processed_index(OUTPUT_PATH)
n = len(dataset["train"])
print(f"Resuming from index {start_idx} / {n}")

with open(OUTPUT_PATH, "a") as f:
    n = len(dataset["train"])
    for start in range(0, n, BATCH_SIZE):# , desc="Augmenting dataset"):
        batch = dataset["train"][start:start + BATCH_SIZE]
        batch_list = [dict(zip(batch.keys(), t)) for t in zip(*batch.values())]

        prompts, meta = build_prompts(
            batch_list,
            attribute_choices,
            prompt_template,
            start_index=start,
            global_seed=GLOBAL_SEED
        )

        decoded_texts = generate_batch(prompts, model, tokenizer, TEMPERATURE, MAX_NEW_TOKENS)

        for text, (ex, par_json) in zip(decoded_texts, meta):
            new_ex = parse_json_safe(text)
            if new_ex:
                merged = {**ex, **par_json, **new_ex}
                f.write(json.dumps(merged) + "\n")
        # break
