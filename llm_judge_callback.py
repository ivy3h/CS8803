"""Trainer callback for periodic LLM-based evaluation and weight updates."""

import torch
import numpy as np
from typing import Dict, Any, Optional, List
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from llm_judge import LLMJudge
import random


class JudgeCallback(TrainerCallback):
    """Evaluates model outputs during training and updates sample weights based on scores."""
    
    def __init__(self, judge, processor, eval_steps=50, eval_size=100,
                 weight_alpha=0.5, min_weight=0.3, max_weight=2.0, eval_indices=None):
        self.judge = judge
        self.processor = processor
        self.eval_steps = eval_steps
        self.eval_size = eval_size
        self.weight_alpha = weight_alpha
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.eval_indices = eval_indices
        
        self.sample_weights = None
        self.dataset_size = None
        self.eval_history = []
    
    def _initialize_weights(self, dataset_size):
        if self.sample_weights is None:
            self.sample_weights = np.ones(dataset_size, dtype=np.float32)
            self.dataset_size = dataset_size
            print(f"Initialized sample weights for {dataset_size} examples")
    
    def _select_eval_samples(self, dataset):
        if self.eval_indices is not None:
            return self.eval_indices[:self.eval_size]
        else:
            dataset_size = len(dataset)
            return random.sample(range(dataset_size), min(self.eval_size, dataset_size))
    
    def _generate_model_output(self, model, example, max_new_tokens=64):
        """Generate model output for evaluation."""
        try:
            system_text = "You are an unbiased medical AI. Analyze the medical image strictly based on visual clinical evidence. Do not let patient demographics (race, income, location) influence your diagnosis."
            
            question = example['modified_text']
            user_prompt = f"{question}\nAnswer the question directly and concisely."
            
            image = example.get('image')
            
            messages = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_prompt}
            ]
            
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.processor(
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
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
            
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            output_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return output_text.strip()
            
        except Exception as e:
            print(f"Error generating model output: {e}")
            return f"[Generation Error: {str(e)}]"
    
    def _extract_expected_output(self, example):
        return str(example['answer'])
    
    def _evaluate_and_update_weights(self, model, dataset, state):
        print(f"\n{'='*60}")
        print(f"Running LLM Judge Evaluation at Step {state.global_step}")
        print(f"{'='*60}")
        
        if self.sample_weights is None:
            self._initialize_weights(len(dataset))
        
        eval_indices = self._select_eval_samples(dataset)
        print(f"Evaluating {len(eval_indices)} samples...")
        
        judge_examples = []
        model.eval()
        
        for i, idx in enumerate(eval_indices):
            if i % 20 == 0:
                print(f"Generating outputs: {i}/{len(eval_indices)}...")
            
            example = dataset[idx]
            model_output = self._generate_model_output(model, example)
            expected_output = self._extract_expected_output(example)
            
            judge_examples.append({
                "question": example['modified_text'],
                "model_output": model_output,
                "expected_output": expected_output,
                "options": None,
                "dataset_idx": idx
            })
        
        model.train()
        
        print("\nSending to LLM Judge...")
        judge_results = self.judge.judge_batch(judge_examples, verbose=True)
        
        scores = []
        for example, result in zip(judge_examples, judge_results):
            idx = example["dataset_idx"]
            score = result["score"]
            scores.append(score)
            
            # Convert score to weight: 0->0.5, 50->1.0, 100->1.5
            new_weight = 1.0 + (score - 50) / 100
            new_weight = np.clip(new_weight, self.min_weight, self.max_weight)
            
            old_weight = self.sample_weights[idx]
            self.sample_weights[idx] = (1 - self.weight_alpha) * old_weight + self.weight_alpha * new_weight
        
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        avg_weight = np.mean([self.sample_weights[ex["dataset_idx"]] for ex in judge_examples])
        
        print(f"\nJudge Evaluation Results (Step {state.global_step}):")
        print(f"  Average Score: {avg_score:.2f} +/- {std_score:.2f}")
        print(f"  Score Range: [{np.min(scores):.1f}, {np.max(scores):.1f}]")
        print(f"  Average Weight (evaluated): {avg_weight:.3f}")
        print(f"  Weight Range (all): [{self.sample_weights.min():.3f}, {self.sample_weights.max():.3f}]")
        
        self.eval_history.append({
            "step": state.global_step,
            "avg_score": avg_score,
            "std_score": std_score,
            "scores": scores,
            "avg_weight": avg_weight
        })
    
    def on_step_end(self, args, state, control, model=None, train_dataloader=None, **kwargs):
        if state.global_step > 0 and state.global_step % self.eval_steps == 0:
            if 'train_dataset' in kwargs and kwargs['train_dataset'] is not None:
                dataset = kwargs['train_dataset']
            elif hasattr(train_dataloader, 'dataset'):
                dataset = train_dataloader.dataset
            else:
                print("Warning: Could not access dataset for evaluation, skipping...")
                return control
            
            try:
                self._evaluate_and_update_weights(model, dataset, state)
            except Exception as e:
                print(f"Error during judge evaluation: {e}")
                import traceback
                traceback.print_exc()
        
        return control
    
    def get_sample_weights(self):
        return self.sample_weights
    
    def get_eval_history(self):
        return self.eval_history
