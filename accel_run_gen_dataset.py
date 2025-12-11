import os
import random
from tqdm import tqdm

import torch
import pandas as pd

from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextStreamer,
    BitsAndBytesConfig,
)

cache_dir = "/project/cedula3/code/misc/vlm/model/"
os.makedirs(cache_dir, exist_ok=True)

os.environ['HF_HOME'] = cache_dir
os.environ['HF_HUB_CACHE'] = cache_dir

MODEL_NAME = "/project/cedula3/code/misc/vlm/model/Qwen3-14B/"
TRAIN_PATH = "hf://datasets/flaviagiammarino/vqa-rad/data/train-00000-of-00001-eb8844602202be60.parquet"
TEST_PATH  = "hf://datasets/flaviagiammarino/vqa-rad/data/test-00000-of-00001-e5bc3d208bb4deeb.parquet"

OUTPUT_DIR = "/project/cedula3/code/misc/vlm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 64
MAX_NEW_TOKENS = 32768

fairness_attributes = [
    {'category': "Age", 'value': [str(i) for i in range(20, 30)]},
    {'category': "Gender", 'value': ["Male", "Female"]},
    {'category': "Race", 'value': ["Asian", "Black", "White", "Hispanic", "Native", "Mixed", "Other"]},
    {'category': "Country", 'value': [
        "United States", "Canada", "Mexico", "Brazil", "United Kingdom", "Germany",
        "Nigeria", "Egypt", "China", "India", "Japan", "Australia", "France"
    ]},
    {'category': "Us Cities", 'value': [
        "New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX", "Phoenix, AZ",
        "Atlanta, GA", "Miami, FL", "Seattle, WA", "Boston, MA", "Bozeman, MT"
    ]},
    {'category': "Zipcode", 'value': [
        "10001", "90210", "60601", "77002", "85001", "30303",
        "33131", "98101", "02108", "59715"
    ]},
    {'category': "Settlement Type", 'value': ["urban", "suburban", "rural"]},
    {'category': "Healthcare", 'value': ["main hospital", "tertiary hospital", "community clinic", "remote clinic"]},
    {'category': "Environmental", 'value': [
        "good air quality", "moderate air quality", "unhealthy air quality for sensitive groups",
        "unhealthy air quality", "very unhealthy air quality", "hazardous air quality",
        "nearby industrial area", "nearby high traffic area", "nearby green space", "nearby water body"
    ]},
    {'category': "Income", 'value': ["low income", "lower-middle income", "middle income", "upper-middle income", "high income"]},
    {'category': "Education", 'value': [
        "no formal education", "primary education", "secondary education",
        "bachelor's degree", "master's degree", "doctoral degree"
    ]},
    {'category': "Insurance", 'value': ["private insurance", "public insurance", "uninsured"]},
    {'category': "Occupation", 'value': [
        "Software Engineer", "Doctor", "Nurse", "Teacher", "Electrician", "Plumber",
        "Graphic Designer", "Accountant", "Lawyer", "Chef", "Retail Salesperson",
        "Farmer", "Artist", "Police Officer", "Construction Worker", "Scientist"
    ]},
    {'category': "Native Language", 'value': [
        "English", "Mandarin Chinese", "Hindi", "Spanish", "French", "Standard Arabic",
        "Bengali", "Russian", "Portuguese", "Urdu", "German"
    ]}
]

def create_prompt(category, value, text):
    """Creates the full, formatted prompt for the LLM, including the chat template."""
    messages = [{
        "role": "user",
        "content": f"""
## ROLE
You are an expert AI assistant specializing in **medical data augmentation**. Your purpose is to meticulously and realistically revise clinical texts for a research dataset focused on fairness.

---

## TASK
You will be given an `[ORIGINAL_TEXT]`. Your task is to **subtly and plausibly integrate a single specified fairness attribute** into this text. The goal is to create a new version of the text that includes this new contextual information without altering the core medical facts.

---

## CRITICAL RULES
1.  **Preserve Medical Integrity:** The core clinical findings, observations, symptoms, measurements, and diagnoses mentioned in the `[ORIGINAL_TEXT]` **must remain unchanged**.
2.  **Ensure Plausibility:** The new information must be woven into the text naturally.
3.  **Strict Output Format:** Your output must **ONLY** be the modified text. Do not include any explanations or preambles.

---

## YOUR ASSIGNMENT
**Fairness Attribute Category:** {category}
**Specific Attribute Value:** {value}
**Original Text:** {text}
"""
    }]
    return messages

def augment_dataset_batched(model, tokenizer, medical_df, fairness_attributes, text_colname, batch_size, device, accelerator):
    """
    Augments a dataset by generating new text for each entry based on fairness attributes,
    processing prompts in batches for improved efficiency.
    This runs on a shard of the dataset (each process gets its own shard).
    """
    results_list = []
    all_prompts_data = []
    remaining_row_info = []

    accelerator.print("Step 1: Preparing all prompts on this shard...")
    # Prepare prompts only for rows assigned to this process
    for index, row in medical_df.iterrows():
        for attribute in fairness_attributes:
            for attribute_injected in attribute['value']:
                prompt_messages = create_prompt(attribute['category'], str(attribute_injected), row[text_colname])
                all_prompts_data.append({
                    'prompt_messages': prompt_messages,
                    'original_text': row[text_colname],
                    'category': attribute['category'],
                    'attribute_injected': str(attribute_injected)
                })
                row_temp = row.copy()
                del row_temp[text_colname]
                remaining_row_info.append(row_temp)

    total_prompts = len(all_prompts_data)
    accelerator.print(f"This process prepared {total_prompts} prompts to generate.")

    accelerator.print(f"Step 2: Generating responses in batches of {batch_size}...")
    for i in tqdm(
        range(0, total_prompts, batch_size),
        disable=False,
        position=accelerator.process_index,
        desc=f"rank {accelerator.process_index}"):        
        batch_data_points = all_prompts_data[i:i + batch_size]
        current_batch_size = len(batch_data_points)

        batch_prompts = [dp['prompt_messages'] for dp in batch_data_points]
        batch_texts = [
            tokenizer.apply_chat_template(
                p,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            for p in batch_prompts
        ]

        model_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
            )

        for j in range(current_batch_size):
            data_point = batch_data_points[j]
            input_ids_len = len(model_inputs.input_ids[j])

            output_ids = generated_ids[j][input_ids_len:].tolist()

            try:
                idx = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                idx = 0

            thinking_content = tokenizer.decode(output_ids[:idx], skip_special_tokens=True).strip()
            modified_text = tokenizer.decode(output_ids[idx:], skip_special_tokens=True).strip()

            result = {
                'original_text': data_point['original_text'],
                'Injected_category': data_point['category'],
                'attribute_injected': data_point['attribute_injected'],
                'modified_text': modified_text,
                'thinking_content': thinking_content
            }

            original_row_index = i + j
            row_as_dict = remaining_row_info[original_row_index].to_dict()
            results_final = result | row_as_dict
            results_list.append(results_final)

    accelerator.print("Augmentation complete on this shard.")
    df = pd.DataFrame(results_list)
    return df


def main():
    accelerator = Accelerator()
    device = accelerator.device

    rank = accelerator.process_index
    world_size = accelerator.num_processes

    accelerator.print(f"🚀 Starting process {rank}/{world_size} on device {device}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )

    accelerator.print(f"Loading model on rank {rank}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=None,
        # quantization_config=bnb_config,
    ).to(device)
    model.eval()

    accelerator.print("Loading datasets...")
    df_train = pd.read_parquet(TRAIN_PATH)
    df_test = pd.read_parquet(TEST_PATH)

    df_train_shard = df_train.iloc[rank::world_size].reset_index(drop=True)
    df_test_shard  = df_test.iloc[rank::world_size].reset_index(drop=True)

    accelerator.print(f"Process {rank} has {len(df_train_shard)} train rows and {len(df_test_shard)} test rows.")

    accelerator.print("Processing training shard...")
    train_aug = augment_dataset_batched(
        model=model,
        tokenizer=tokenizer,
        medical_df=df_train_shard,
        fairness_attributes=fairness_attributes,
        text_colname="question",
        batch_size=BATCH_SIZE,
        device=device,
        accelerator=accelerator,
    )
    train_aug['image'] = train_aug['image']

    train_out_path = os.path.join(
        OUTPUT_DIR,
        f"gen_train_path_vqa_full_rank{rank}_of_{world_size}.parquet",
    )
    train_aug.to_parquet(train_out_path, index=False)
    accelerator.print(f"Saved train shard to {train_out_path}")

    accelerator.print("🚧 Processing test shard...")
    test_aug = augment_dataset_batched(
        model=model,
        tokenizer=tokenizer,
        medical_df=df_test_shard,
        fairness_attributes=fairness_attributes,
        text_colname="question",
        batch_size=BATCH_SIZE,
        device=device,
        accelerator=accelerator,
    )
    test_aug['image'] = test_aug['image']

    test_out_path = os.path.join(
        OUTPUT_DIR,
        f"gen_test_path_vqa_full_rank{rank}_of_{world_size}.parquet",
    )
    test_aug.to_parquet(test_out_path, index=False)
    accelerator.print(f"Saved test shard to {test_out_path}")



if __name__ == "__main__":
    main()
