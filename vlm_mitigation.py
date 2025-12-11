import os

CACHE_DIR = os.path.abspath("./hf_cache")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

import torch
import pandas as pd
import gc
from collections import defaultdict
from datasets import load_dataset, concatenate_datasets
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from tqdm import tqdm
import random

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DATASET_ID = "chandsang/fair_path_vqa"
OUTPUT_DIR = "./qwen-fairness-finetune"

JUDGE_API_KEY = os.environ.get("OPENAI_API_KEY", None)
JUDGE_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o")
JUDGE_EVAL_STEPS = int(os.environ.get("JUDGE_EVAL_STEPS", "50"))
JUDGE_EVAL_SIZE = int(os.environ.get("JUDGE_EVAL_SIZE", "100"))
ENABLE_JUDGE = True

CATEGORY_MAPPING = {
    "Demographic": ["Age", "Gender", "Race"],
    "Geographic": ["Country", "Us Cities", "Zipcode", "Settlement Type"],
    "Socioeconomic": ["Education", "Income", "Occupation", "Insurance", "Healthcare"],
    "Cultural_Religious": ["Environmental"],
    "Linguistic": ["Native Language"],
}

def get_mapped_category(injected_category):
    if injected_category == "Vanilla":
        return "Vanilla"
    for target_cat, source_cats in CATEGORY_MAPPING.items():
        if injected_category in source_cats:
            return target_cat
    return "Unknown"


def format_for_chatml(example):
    """Format fair_path_vqa examples for Qwen2.5-VL training."""
    system_text = "You are an unbiased medical AI. Analyze the medical image strictly based on visual clinical evidence. Do not let patient demographics (race, income, location) influence your diagnosis."
    
    question = example['modified_text']
    user_prompt = f"{question}\nAnswer the question directly and concisely."
    response = str(example['answer'])
    
    image_placeholder = "<|vision_start|><|image_pad|><|vision_end|>"
    
    prompt = (
        f"<|im_start|>system\n{system_text}<|im_end|>\n"
        f"<|im_start|>user\n{image_placeholder}{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )
    
    return {"formatted_text": prompt}


class Qwen2VLDataCollator:
    """Data collator for Qwen2.5-VL that handles image + text processing."""
    
    def __init__(self, processor, sample_weights=None):
        self.processor = processor
        self.sample_weights = sample_weights
    
    def __call__(self, examples):
        texts = []
        images_list = []
        sample_indices = []
        
        for example in examples:
            texts.append(example["formatted_text"])
            image = example.get("image")
            images_list.append([image] if image is not None else None)
            
            if "idx" in example:
                sample_indices.append(example["idx"])
        
        batch = self.processor(
            text=texts,
            images=images_list,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        labels = batch["input_ids"].clone()
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        
        if self.sample_weights is not None and len(sample_indices) > 0:
            weights = self.sample_weights
            if callable(weights):
                weights = weights()
            
            if weights is not None:
                batch_weights = torch.tensor(
                    [weights[idx] for idx in sample_indices],
                    dtype=torch.float32
                )
                batch["sample_weights"] = batch_weights
        
        return batch


class WeightedLossTrainer(Trainer):
    """Trainer that supports sample-weighted loss calculation."""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        sample_weights = inputs.pop("sample_weights", None)
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        
        if sample_weights is not None:
            logits = outputs.logits
            labels = inputs.get("labels")
            
            if labels is not None:
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                
                per_token_loss = loss_fct(shift_logits, shift_labels)
                
                seq_len = labels.size(1) - 1
                per_token_loss = per_token_loss.view(-1, seq_len)
                
                sample_weights = sample_weights.to(per_token_loss.device)
                weights_expanded = sample_weights.view(-1, 1).expand_as(per_token_loss)
                
                weighted_loss = per_token_loss * weights_expanded
                
                shift_labels_reshaped = shift_labels.view(-1, seq_len)
                mask = (shift_labels_reshaped != -100).float()
                
                loss = (weighted_loss * mask).sum() / mask.sum()
        
        return (loss, outputs) if return_outputs else loss


EVAL_OUTPUT_DIR = "./output_data"
EVAL_SAMPLES_PER_CATEGORY = 1000


def generate_model_output(model, processor, example, max_new_tokens=64):
    """Generate model output for a given example."""
    try:
        system_text = "You are an unbiased medical AI. Analyze the medical image strictly based on visual clinical evidence. Do not let patient demographics (race, income, location) influence your diagnosis."
        
        question = example['modified_text']
        user_prompt = f"{question}\nAnswer the question directly and concisely."
        
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_prompt}
        ]
        
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        image = example.get('image')
        
        inputs = processor(
            text=[text],
            images=[[image]] if image is not None else None,
            return_tensors="pt",
            padding=True
        )
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        output_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return output_text.strip()
        
    except Exception as e:
        print(f"Error generating model output: {e}")
        return f"[Generation Error]"


def evaluate_and_save_csv(model, processor, dataset, output_dir=EVAL_OUTPUT_DIR, max_samples=EVAL_SAMPLES_PER_CATEGORY):
    """Evaluate model on dataset and save results as CSV files per category."""
    print(f"\n{'='*60}")
    print("POST-TRAINING EVALUATION")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    print("Converting dataset to pandas...")
    df_full = dataset.to_pandas()
    
    print(f"Dataset has {len(df_full)} total samples")
    print(f"Injected categories: {df_full['Injected_category'].unique().tolist()}")
    
    all_results = []
    
    # Vanilla samples
    print("\n[Evaluating Vanilla (baseline)]")
    vanilla_samples = df_full.drop_duplicates(subset=['original_text', 'answer']).head(max_samples)
    
    for idx, row in tqdm(vanilla_samples.iterrows(), total=len(vanilla_samples), desc="Vanilla"):
        example = {
            'modified_text': row['original_text'],
            'answer': row['answer'],
            'image': dataset[idx]['image'] if idx < len(dataset) else None
        }
        
        vlm_output = generate_model_output(model, processor, example)
        is_correct = vlm_output.strip().lower() == str(row['answer']).strip().lower()
        
        all_results.append({
            'question': row['original_text'],
            'answer': row['answer'],
            'vlm_output': vlm_output,
            'is_correct': is_correct,
            'category': 'Vanilla',
            'Injected_category': 'Vanilla',
            'attribute_injected': 'baseline'
        })
    
    vanilla_df = pd.DataFrame([r for r in all_results if r['category'] == 'Vanilla'])
    if not vanilla_df.empty:
        vanilla_path = os.path.join(output_dir, 'vanilla_lora_llm_judge.csv')
        vanilla_df.to_csv(vanilla_path, index=False)
        accuracy = vanilla_df['is_correct'].mean() * 100
        print(f"Saved {len(vanilla_df)} vanilla samples to {vanilla_path} (accuracy: {accuracy:.2f}%)")
    
    for target_category, source_categories in CATEGORY_MAPPING.items():
        print(f"\n[Evaluating {target_category}]")
        
        category_mask = df_full['Injected_category'].isin(source_categories)
        category_samples = df_full[category_mask].head(max_samples)
        
        if len(category_samples) == 0:
            print(f"  No samples found for {target_category}")
            continue
        
        category_results = []
        
        for _, row in tqdm(category_samples.iterrows(), total=len(category_samples), desc=target_category):
            try:
                orig_idx = df_full.index.get_loc(_) if _ in df_full.index else 0
                image = dataset[orig_idx]['image'] if orig_idx < len(dataset) else None
            except:
                image = None
            
            example = {
                'modified_text': row['modified_text'],
                'answer': row['answer'],
                'image': image
            }
            
            vlm_output = generate_model_output(model, processor, example)
            is_correct = vlm_output.strip().lower() == str(row['answer']).strip().lower()
            
            result = {
                'question': row['modified_text'],
                'answer': row['answer'],
                'vlm_output': vlm_output,
                'is_correct': is_correct,
                'category': target_category,
                'Injected_category': row['Injected_category'],
                'attribute_injected': row['attribute_injected']
            }
            category_results.append(result)
            all_results.append(result)
        
        if category_results:
            category_df = pd.DataFrame(category_results)
            filename = f"{target_category.lower().replace(' ', '_').replace('&', '_')}_lora_llm_judge.csv"
            category_path = os.path.join(output_dir, filename)
            category_df.to_csv(category_path, index=False)
            accuracy = category_df['is_correct'].mean() * 100
            print(f"Saved {len(category_df)} samples to {category_path} (accuracy: {accuracy:.2f}%)")
    
    print(f"\n[Saving Global Results]")
    global_df = pd.DataFrame(all_results)
    global_path = os.path.join(output_dir, 'global_lora_llm_judge.csv')
    global_df.to_csv(global_path, index=False)
    overall_accuracy = global_df['is_correct'].mean() * 100
    print(f"Saved {len(global_df)} total samples to {global_path} (overall accuracy: {overall_accuracy:.2f}%)")
    
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Category':<25} {'Samples':<10} {'Accuracy':<10}")
    print("-" * 45)
    
    for category in ['Vanilla'] + list(CATEGORY_MAPPING.keys()):
        cat_results = [r for r in all_results if r['category'] == category]
        if cat_results:
            n_samples = len(cat_results)
            accuracy = sum(1 for r in cat_results if r['is_correct']) / n_samples * 100
            print(f"{category:<25} {n_samples:<10} {accuracy:.2f}%")
    
    print("-" * 45)
    print(f"{'TOTAL':<25} {len(all_results):<10} {overall_accuracy:.2f}%")
    print(f"{'='*60}\n")
    
    return all_results


def main():
    print(f"Loading model: {MODEL_ID}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="flash_attention_2",
            cache_dir=CACHE_DIR
        )
        print("Using flash_attention_2")
    except (ImportError, ValueError) as e:
        print(f"flash_attention_2 not available ({e}), falling back to default attention")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=CACHE_DIR
        )
    
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    
    print(f"Loading dataset: {DATASET_ID}")
    ds = load_dataset(DATASET_ID, split="train", cache_dir=CACHE_DIR)
    print(f"Dataset size: {len(ds)}")
    
    formatted_ds = ds.map(format_for_chatml, num_proc=8)
    formatted_ds = formatted_ds.map(lambda example, idx: {**example, "idx": idx}, with_indices=True)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        max_steps=100,
        logging_steps=5,
        fp16=True,
        save_steps=50,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        save_total_limit=2,
    )
    
    judge_callback = None
    if ENABLE_JUDGE:
        if not JUDGE_API_KEY:
            print("WARNING: ENABLE_JUDGE is True but JUDGE_API_KEY is not set. Disabling judge.")
            print("Set OPENAI_API_KEY environment variable to enable LLM judge.")
        else:
            print(f"\nInitializing LLM Judge")
            print(f"  API Endpoint: {JUDGE_BASE_URL}")
            print(f"  Model: {JUDGE_MODEL}")
            print(f"  Eval Steps: {JUDGE_EVAL_STEPS}")
            print(f"  Eval Size: {JUDGE_EVAL_SIZE}\n")
            
            from llm_judge import LLMJudge
            from llm_judge_callback import JudgeCallback
            
            judge = LLMJudge(
                api_key=JUDGE_API_KEY,
                base_url=JUDGE_BASE_URL,
                model_name=JUDGE_MODEL
            )
            
            judge_callback = JudgeCallback(
                judge=judge,
                processor=processor,
                eval_steps=JUDGE_EVAL_STEPS,
                eval_size=JUDGE_EVAL_SIZE,
                weight_alpha=0.5,
                min_weight=0.3,
                max_weight=2.0
            )
    
    if judge_callback is not None:
        data_collator = Qwen2VLDataCollator(
            processor,
            sample_weights=lambda: judge_callback.get_sample_weights()
        )
    else:
        data_collator = Qwen2VLDataCollator(processor)
    
    callbacks = [judge_callback] if judge_callback is not None else []
    
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_ds,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    
    print("\nRunning post-training evaluation...")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    eval_ds = load_dataset(DATASET_ID, split="train", cache_dir=CACHE_DIR)
    
    evaluate_and_save_csv(
        model=model,
        processor=processor,
        dataset=eval_ds,
        output_dir=EVAL_OUTPUT_DIR,
        max_samples=EVAL_SAMPLES_PER_CATEGORY
    )
    
    print("Training and evaluation complete!")
    print(f"Model saved to: {OUTPUT_DIR}/final")
    print(f"Evaluation CSVs saved to: {EVAL_OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
