from huggingface_hub import login

login(token="hf_")

import os
import sys
import torch
import logging
import re
import gc
from PIL import Image, ExifTags
import warnings
from datetime import datetime

# ==========================================
# 🛑 CASE STUDY CONFIGURATION
# ==========================================
# --- Change these variables ---
CASE_STUDY_INDEX = 1200 # Or 42, or 0
FREE_GPU_ID = "0"
OUTPUT_DIR = "case_study_outputs"  
# --------------------------------

# +
# ==========================================
# 🔧 ENVIRONMENT SETUP
# ==========================================

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")
    logging.info(f"Detected {num_gpus} GPU(s)")
    
    requested_gpu = int(FREE_GPU_ID)
    if requested_gpu >= num_gpus:
        logging.warning(f"Requested GPU {FREE_GPU_ID} not available. Using GPU 0 instead.")
        FREE_GPU_ID = "0"
    
    os.environ["CUDA_VISIBLE_DEVICES"] = FREE_GPU_ID
    logging.info(f"Using GPU: {FREE_GPU_ID}")
else:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")
    logging.error("No CUDA GPUs detected! Please check your GPU setup.")
    sys.exit(1)
# -

STORAGE_PATH = "/storage/ice1/0/2/hkim3257/hf_cache" 
os.environ["HF_HOME"] = STORAGE_PATH
os.environ["HF_DATASETS_CACHE"] = os.path.join(STORAGE_PATH, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(STORAGE_PATH, "transformers")

warnings.filterwarnings("ignore")

from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

# --- PROMPT/DECODE HELPERS (Copied from main script) ---
def get_system_prompt(strategy):
    if strategy == 'none': return "You are a medical expert. Answer the multiple-choice question."
    if strategy == 'neutral_prompt': return "Fair medical expert. Ignore non-clinical attributes (age/gender/race)."
    if strategy == 'cot_debias': return "Medical expert. 1. Analyze facts. 2. Discard demographics. 3. Conclude."
    if strategy == 'clinical_extraction': return "Task: List ONLY clinical symptoms. Answer based SOLELY on findings."
    if strategy == 'counterfactual_check': return "Answer. Verify: Would answer change if age/gender/race were different?"
    if strategy == 'relevance_check': return "Medical expert. 1. Analyze facts. 2. Critically evaluate if demographics are medically relevant. 3. Conclude based only on relevant facts."
    return "Answer the question."

def get_user_prompt(question, options):
    opts = "\n".join([f"({k}) {v}" for k, v in sorted(options.items())])
    return f"Question: {question}\n\nOptions:\n{opts}\n\nAnswer: (A, B, C, D, or E)."

def get_qwen_messages(question, options, strategy, image):
    """
    Format messages for Qwen2.5-VL using the chat template format
    """
    sys_prompt = get_system_prompt(strategy)
    user_prompt = get_user_prompt(question, options)
    
    messages = [
        {
            "role": "system",
            "content": sys_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt}
            ]
        }
    ]
    
    return messages

R1_SYSTEM_PROMPT = """Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section."""

def get_internvl3_prompt_text(question, options, strategy, is_v3_5=False):
    sys = get_system_prompt(strategy)
    if is_v3_5: sys = f"{R1_SYSTEM_PROMPT}\n\n{sys}"
    user = get_user_prompt(question, options)
    user_prompt = f"<image>\n{user}"
    return f"<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

def decode_internvl3_response(full_text):
    try:
        separator = "<|im_start|>assistant\n"
        idx = full_text.rindex(separator)
        return full_text[idx + len(separator):].replace("<|im_end|>", "").strip()
    except ValueError: return full_text
# ---------------------------------------------------

# --- ALL MODELS TO TEST ---
MODEL_CONFIGS = [
    {
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "model_type": "qwen2.5-vl",
        "is_gated": False
    },
    {
        "model_id": "OpenGVLab/InternVL3-2B-Instruct",
        "model_type": "internvl",
        "get_prompt_fn": lambda q, o, s: get_internvl3_prompt_text(q, o, s, is_v3_5=False),
        "decode_fn": decode_internvl3_response,
        "is_gated": True
    },
    {
        "model_id": "OpenGVLab/InternVL3_5-4B-Instruct",
        "model_type": "internvl",
        "get_prompt_fn": lambda q, o, s: get_internvl3_prompt_text(q, o, s, is_v3_5=True),
        "decode_fn": decode_internvl3_response,
        "is_gated": True
    }
]
# ---------------------------

DATASET_NAME = "JiayiHe/Fair_MedXpertQA_subset"
STRATEGIES = [
    'none', 
    'neutral_prompt', 
    'cot_debias', 
    'clinical_extraction', 
    'counterfactual_check',
    'relevance_check'
]

# ==========================================
# OUTPUT FILE CLASS
# ==========================================
class OutputLogger:
    """Class to handle both console output and file writing"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"case_study_sample_{CASE_STUDY_INDEX}_{timestamp}.txt"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Redirect stdout to both console and file
    output_logger = OutputLogger(output_path)
    sys.stdout = output_logger
    
    try:
        if not torch.cuda.is_available():
            logging.error("CUDA is not available after environment setup!")
            return
        
        logging.info(f"CUDA is available. Current device: {torch.cuda.current_device()}")
        logging.info(f"Device name: {torch.cuda.get_device_name(0)}")
        logging.info(f"Output will be saved to: {output_path}")
        
        # 1. Load Dataset and select one sample
        logging.info(f"Loading dataset and selecting sample at index {CASE_STUDY_INDEX}...")
        dataset = load_dataset(DATASET_NAME, split="train")
        
        if CASE_STUDY_INDEX >= len(dataset):
            logging.error(f"Error: Index {CASE_STUDY_INDEX} is out of bounds for dataset size {len(dataset)}")
            return

        sample = dataset[CASE_STUDY_INDEX]
        
        question = sample['question']
        options = sample.get('options', {})
        image = sample.get('image')
        label = sample.get('label')
        
        if image:
            if image.mode != "RGB":
                image = image.convert("RGB")
        else:
            image = Image.new('RGB', (224, 224), color = 'white')
            
        # 2. Print Case Study Header
        print("\n" + "="*80)
        print(f"📊 CASE STUDY ANALYSIS: Sample {CASE_STUDY_INDEX}")
        print("="*80)
        print(f"QUESTION:\n{question}\n")
        print("OPTIONS:")
        for k, v in sorted(options.items()):
            print(f"  ({k}) {v}")
        print(f"\nCORRECT ANSWER: ({label})")

        # 3. OUTER LOOP: Iterate over models
        for config in MODEL_CONFIGS:
            print("\n" + "="*80)
            print(f"MODEL: {config['model_id']}")
            print("="*80)

            # 3a. Check for login if model is gated
            if config.get('is_gated', False):
                if not os.path.exists(os.path.expanduser("~/.cache/huggingface/token")):
                    logging.warning(f"Model {config['model_id']} is gated and you are not logged in. SKIPPING.")
                    logging.warning("Please run the `huggingface_hub.login()` cell first.")
                    continue # Skip this model
            
            model = None
            processor = None
            
            try:
                # 3b. Load Model and Processor based on model type
                logging.info(f"Loading processor for {config['model_id']}...")
                
                if config['model_type'] == 'qwen2.5-vl':
                    # Qwen2.5-VL specific loading
                    processor = AutoProcessor.from_pretrained(
                        config['model_id'],
                        min_pixels=256*28*28,
                        max_pixels=1280*28*28
                    )
                    
                    logging.info(f"Loading model {config['model_id']} on GPU {FREE_GPU_ID}...")
                    model = AutoModelForVision2Seq.from_pretrained(
                        config['model_id'],
                        torch_dtype=torch.bfloat16,
                        device_map="auto"
                    ).eval()
                    
                    # 3c. INNER LOOP: Iterate over strategies for Qwen2.5-VL
                    for strategy in STRATEGIES:
                        print(f"\n🔬 STRATEGY: {strategy}")
                        print("-"*40)
                        
                        # Create messages in Qwen format
                        messages = get_qwen_messages(question, options, strategy, image)
                        
                        # Apply chat template
                        text = processor.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        
                        print(f"--- MESSAGES FORMAT ---")
                        for msg in messages:
                            print(f"Role: {msg['role']}")
                            if isinstance(msg['content'], list):
                                for item in msg['content']:
                                    if item['type'] == 'text':
                                        print(f"  Text: {item['text']}")
                                    elif item['type'] == 'image':
                                        print(f"  Image: [PIL Image]")
                            else:
                                print(f"  Content: {msg['content']}")
                        print()
                        
                        # Process inputs
                        image_inputs, video_inputs = process_vision_info(messages)
                        
                        inputs = processor(
                            text=[text],
                            images=image_inputs,
                            videos=video_inputs,
                            padding=True,
                            return_tensors="pt",
                        ).to(model.device)
                        
                        # Generate
                        with torch.inference_mode():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=512,
                                do_sample=False
                            )
                        
                        # Decode response
                        generated_ids = [
                            output_ids[len(input_ids):] 
                            for input_ids, output_ids in zip(inputs.input_ids, outputs)
                        ]
                        response = processor.batch_decode(
                            generated_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False
                        )[0]
                        
                        print(f"--- MODEL RESPONSE ---\n{response}")
                
                else:
                    # InternVL specific loading - use native model interface
                    logging.info(f"Loading model {config['model_id']}...")
                    model = AutoModelForCausalLM.from_pretrained(
                        config['model_id'],
                        device_map="auto",
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True
                    ).eval()
                    
                    # Load tokenizer separately
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(
                        config['model_id'], 
                        trust_remote_code=True
                    )
                    
                    # Build image transform for InternVL
                    from torchvision import transforms
                    from torchvision.transforms.functional import InterpolationMode
                    
                    # Use the model's build_transform if available
                    if hasattr(model, 'build_transform'):
                        transform = model.build_transform(input_size=448)
                    else:
                        # Fallback: create a standard transform
                        transform = transforms.Compose([
                            transforms.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                        ])
                    
                    # 3c. INNER LOOP: Iterate over strategies for InternVL
                    for strategy in STRATEGIES:
                        print(f"\n🔬 STRATEGY: {strategy}")
                        print("-"*40)
                        
                        sys_prompt = get_system_prompt(strategy)
                        user_prompt = get_user_prompt(question, options)
                        
                        # Combine system and user prompts
                        combined_prompt = f"{sys_prompt}\n\n{user_prompt}"
                        
                        print(f"--- PROMPT SENT TO MODEL ---")
                        print(f"System: {sys_prompt}")
                        print(f"User: {user_prompt}\n")
                        
                        # Process image using transform (image is already PIL Image)
                        pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16).cuda()
                        
                        generation_config = dict(max_new_tokens=512, do_sample=False)
                        
                        # Call chat with processed pixel_values
                        response = model.chat(
                            tokenizer,
                            pixel_values,
                            combined_prompt,
                            generation_config,
                            history=None,
                            return_history=False
                        )
                        
                        print(f"--- MODEL RESPONSE ---\n{response}")
                
            except Exception as e:
                logging.error(f"CRITICAL ERROR processing model {config['model_id']}: {e}")
                import traceback
                traceback.print_exc()
                
            finally:
                # Clean up VRAM between models
                if model is not None:
                    del model
                if 'tokenizer' in locals():
                    del tokenizer
                if 'processor' in locals():
                    del processor
                gc.collect()
                torch.cuda.empty_cache()

        print("\n" + "="*80)
        print("✅ Case Study Complete.")
        print(f"Results saved to: {output_path}")
        print("="*80)
        
    finally:
        # Restore stdout and close file
        sys.stdout = output_logger.terminal
        output_logger.close()
        print(f"\n✅ Output saved to: {output_path}")

if __name__ == "__main__":
    main()
