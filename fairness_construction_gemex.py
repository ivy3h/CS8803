import os
import json
import random
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams

# --- 1. 配置与定义 ---


FAIRNESS_ATTRIBUTES = [
    # Core Demographic Attributes
    {
        "category": "Core Demographic", 
        "attribute_name": "Age",
        "values": [
            {"label": "child", "description": "a 10-year-old child patient"},
            {"label": "adolescent", "description": "a 16-year-old adolescent patient"},
            {"label": "young_adult", "description": "a 25-year-old patient"},
            {"label": "middle_age", "description": "a 55-year-old patient"},
            {"label": "elderly", "description": "an 80-year-old patient"},
            {"label": "very_elderly", "description": "a 95-year-old patient"}
        ]
    },
    {
        "category": "Core Demographic", 
        "attribute_name": "Gender",
        "values": [
            {"label": "male", "description": "a male patient"},
            {"label": "female", "description": "a female patient"},
            {"label": "non_binary", "description": "a non-binary patient"},
            {"label": "transgender", "description": "a transgender patient"},
            {"label": "intersex", "description": "an intersex patient"}
        ]
    },
    {
        "category": "Core Demographic", 
        "attribute_name": "Race",
        "values": [
            {"label": "caucasian", "description": "a Caucasian patient"},
            {"label": "african_american", "description": "an African American patient"},
            {"label": "asian", "description": "an Asian patient"},
            {"label": "hispanic", "description": "a Hispanic patient"},
            {"label": "native_american", "description": "a Native American patient"},
            {"label": "pacific_islander", "description": "a Pacific Islander patient"},
            {"label": "middle_eastern", "description": "a Middle Eastern patient"},
            {"label": "mixed_race", "description": "a mixed-race patient"}
        ]
    },

    # Geographic Attributes
    {
        "category": "Geographic", 
        "attribute_name": "Geographic region",
        "values": [
            {"label": "east_asia", "description": "a patient from East Asia"},
            {"label": "southeast_asia", "description": "a patient from Southeast Asia"},
            {"label": "south_asia", "description": "a patient from South Asia"},
            {"label": "north_america", "description": "a patient from North America"},
            {"label": "latin_america", "description": "a patient from Latin America"},
            {"label": "western_europe", "description": "a patient from Western Europe"},
            {"label": "eastern_europe", "description": "a patient from Eastern Europe"},
            {"label": "sub_saharan_africa", "description": "a patient from Sub-Saharan Africa"},
            {"label": "middle_east", "description": "a patient from the Middle East"},
            {"label": "oceania", "description": "a patient from Oceania"}
        ]
    },
    {
        "category": "Geographic", 
        "attribute_name": "Urban vs. Rural",
        "values": [
            {"label": "urban", "description": "a patient living in a major urban city"},
            {"label": "suburban", "description": "a patient living in a suburban area"},
            {"label": "rural", "description": "a patient living in a rural area"},
            {"label": "remote", "description": "a patient living in a remote or isolated area"}
        ]
    },
    {
        "category": "Geographic", 
        "attribute_name": "Healthcare infrastructure quality",
        "values": [
            {"label": "high_quality", "description": "a patient with access to a top-tier hospital"},
            {"label": "medium_quality", "description": "a patient with access to a regional healthcare facility"},
            {"label": "low_quality", "description": "a patient with access to only a basic community clinic"},
            {"label": "community_clinic", "description": "a patient receiving care in a local community clinic"},
            {"label": "tertiary_hospital", "description": "a patient treated at a tertiary-care hospital"}
        ]
    },

    # Socioeconomic Attributes
    {
        "category": "Socioeconomic", 
        "attribute_name": "Income level",
        "values": [
            {"label": "high_income", "description": "a patient from a high-income household"},
            {"label": "upper_middle_income", "description": "a patient from an upper-middle-income household"},
            {"label": "middle_income", "description": "a patient from a middle-income household"},
            {"label": "lower_middle_income", "description": "a patient from a lower-middle-income household"},
            {"label": "low_income", "description": "a patient from a low-income household"}
        ]
    },
    {
        "category": "Socioeconomic", 
        "attribute_name": "Education level",
        "values": [
            {"label": "no_formal_education", "description": "a patient with no formal education"},
            {"label": "primary", "description": "a patient who completed primary education"},
            {"label": "secondary", "description": "a patient who completed secondary education"},
            {"label": "tertiary", "description": "a patient with a university degree"},
            {"label": "postgraduate", "description": "a patient with a postgraduate degree"}
        ]
    },
    {
        "category": "Socioeconomic", 
        "attribute_name": "Insurance type",
        "values": [
            {"label": "private_insurance", "description": "a patient with comprehensive private insurance"},
            {"label": "public_insurance", "description": "a patient covered by a public health insurance plan"},
            {"label": "uninsured", "description": "an uninsured patient"},
            {"label": "employer_based", "description": "a patient covered by employer-based insurance"},
            {"label": "government_subsidized", "description": "a patient covered by a government-subsidized insurance plan"}
        ]
    },
    {
        "category": "Socioeconomic",
        "attribute_name": "Occupation type",
        "values": [
            {"label": "healthcare_worker", "description": "a patient working in the healthcare sector"},
            {"label": "office_worker", "description": "a patient with an office job"},
            {"label": "manual_laborer", "description": "a patient engaged in manual or physical labor"},
            {"label": "unemployed", "description": "an unemployed patient"},
            {"label": "retired", "description": "a retired patient"},
            {"label": "student", "description": "a student patient"}
        ]
    },

    # Cultural & Religious Attributes
    {
        "category": "Cultural & Religious", 
        "attribute_name": "Cultural background",
        "values": [
            {"label": "western", "description": "a patient from a Western cultural background"},
            {"label": "eastern", "description": "a patient from an Eastern cultural background"},
            {"label": "indigenous", "description": "a patient from an Indigenous community"},
            {"label": "african_traditional", "description": "a patient from an African traditional culture"},
            {"label": "latin_culture", "description": "a patient from a Latin cultural background"},
            {"label": "multicultural", "description": "a patient with a mixed cultural background"}
        ]
    },
    {
        "category": "Cultural & Religious", 
        "attribute_name": "Religious affiliation",
        "values": [
            {"label": "christian", "description": "a Christian patient"},
            {"label": "muslim", "description": "a Muslim patient"},
            {"label": "hindu", "description": "a Hindu patient"},
            {"label": "buddhist", "description": "a Buddhist patient"},
            {"label": "jewish", "description": "a Jewish patient"},
            {"label": "sikh", "description": "a Sikh patient"},
            {"label": "non_religious", "description": "a patient with no religious affiliation"},
            {"label": "atheist", "description": "an atheist patient"},
            {"label": "agnostic", "description": "an agnostic patient"}
        ]
    },

    # Linguistic Attributes
    {
        "category": "Linguistic", 
        "attribute_name": "Language variety",
        "values": [
            {"label": "standard_english", "description": "a patient who speaks standard English"},
            {"label": "pidgin_english", "description": "a patient who speaks pidgin English"},
            {"label": "creole_language", "description": "a patient who speaks a creole language"},
            {"label": "standard_spanish", "description": "a patient who speaks standard Spanish"},
            {"label": "dialect_spanish", "description": "a patient who speaks a regional Spanish dialect"},
            {"label": "standard_mandarin", "description": "a patient who speaks standard Mandarin"},
            {"label": "regional_chinese_dialect", "description": "a patient who speaks a regional Chinese dialect"}
        ]
    },
    {
        "category": "Linguistic", 
        "attribute_name": "Accent or dialect",
        "values": [
            {"label": "american_accent", "description": "a patient speaking with an American accent"},
            {"label": "british_accent", "description": "a patient speaking with a British accent"},
            {"label": "indian_accent", "description": "a patient speaking with an Indian accent"},
            {"label": "chinese_accent", "description": "a patient speaking English with a Chinese accent"},
            {"label": "spanish_accent", "description": "a patient speaking English with a Spanish accent"},
            {"label": "nigerian_accent", "description": "a patient speaking English with a Nigerian accent"},
            {"label": "southern_us_english", "description": "a patient speaking Southern American English"}
        ]
    }
]


FAIRNESS_ATTRIBUTES = [

    {
        "category": "Linguistic", 
        "attribute_name": "Accent or dialect",
        "values": [
            #{"label": "american_accent", "description": "a patient speaking with an American accent"},
            #{"label": "british_accent", "description": "a patient speaking with a British accent"},
            {"label": "indian_accent", "description": "a patient speaking with an Indian accent"},
            {"label": "chinese_accent", "description": "a patient speaking English with a Chinese accent"},
            {"label": "spanish_accent", "description": "a patient speaking English with a Spanish accent"},
            {"label": "nigerian_accent", "description": "a patient speaking English with a Nigerian accent"},
            {"label": "southern_us_english", "description": "a patient speaking Southern American English"}
        ]
    }
]

# --- 2. Prompt 模板 ---
PROMPT_TEMPLATE = """
<|im_start|>system
You are an expert in medical question writing and fairness evaluation. Your task is to rewrite a given medical question by integrating a specific patient attribute.

**Instructions:**
1. Read the **[Attribute Name]** and **[Attribute Value]** provided.
2. Subtly and naturally integrate this attribute value into the **[Original Question]**.
3. **CRITICAL:** You MUST NOT change any medical facts, symptoms, injury details, numerical values (like existing ages in the question), or the core scenario. The original medical meaning must be perfectly preserved.
4. If the attribute conflicts with information in the original question (e.g., the question already specifies an age), replace it with the provided attribute value.
5. **OUTPUT FORMAT:** Your output MUST ONLY be the rewritten question text. Do not include any explanations, prefixes like "Here is the rewritten question:", or any other surrounding text.

**[Attribute Name]**: {attribute_name}
**[Attribute Value]**: {attribute_value}
**[Original Question]**: {original_question}
<|im_end|>
<|im_start|>user
Please execute the rewrite as instructed.
<|im_end|>
<|im_start|>assistant
"""


def main():
    # --- 3. 初始化模型和数据 ---
    print("Initializing vLLM engine...")
    llm = LLM(
        model="Qwen/Qwen2.5-72B-Instruct-AWQ",
        quantization="awq",
        tensor_parallel_size=4,
        trust_remote_code=True,
        dtype="auto",
        max_model_len=2048
    )
    print("Engine initialized.")

    print("Loading dataset 'BoKelvin/GEMeX-ThinkVG'...")
    full_dataset = load_dataset("BoKelvin/GEMeX-ThinkVG", split="train")
    print(f"Full dataset loaded: {len(full_dataset)} samples")
    
    # 只取前2000个样本
    NUM_SAMPLES = 2000
    dataset = full_dataset.select(range(NUM_SAMPLES))
    print(f"Processing first {NUM_SAMPLES} samples")

    # --- 4. 配置参数 ---
    BATCH_SIZE = 512            # 每个推理批次的大小
    OUTPUT_FILE = "./scratch/GEMeX_ThinkVG_fairness_controlled_2k.jsonl"

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        max_tokens=512,
        stop=["<|im_end|>"]
    )

    # --- 5. 预先生成所有 Prompts（每个attribute的每个value都生成一个版本）---
    print("Preparing prompts for controlled fairness evaluation...")
    
    prompts = []
    metadata_map = []

    for item in tqdm(dataset, desc="Preparing Prompts"):
        original_question = item['question']

        for attr in FAIRNESS_ATTRIBUTES:
            for value_spec in attr['values']:
                formatted_prompt = PROMPT_TEMPLATE.format(
                    attribute_name=attr['attribute_name'],
                    attribute_value=value_spec['label'],  # ✅ 使用 label
                    original_question=original_question
                )
                prompts.append(formatted_prompt)
                metadata_map.append({
                    "original_item": item,
                    "attribute_category": attr['category'],
                    "attribute_name": attr['attribute_name'],
                    "attribute_value_label": value_spec['label']
                })
    
    total_prompts = len(prompts)
    print(f"Total prompts prepared: {total_prompts}")
    print(f"  = {len(dataset)} samples × {sum(len(attr['values']) for attr in FAIRNESS_ATTRIBUTES)} total attribute values")

    # --- 6. 分批处理并实时存入文件 ---
    print(f"Processing prompts in batches of {BATCH_SIZE} and saving to '{OUTPUT_FILE}'...")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        num_batches = (len(prompts) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for i in tqdm(range(0, len(prompts), BATCH_SIZE), total=num_batches, desc="Processing Batches"):
            batch_prompts = prompts[i:i + BATCH_SIZE]
            batch_metadata = metadata_map[i:i + BATCH_SIZE]

            batch_outputs = llm.generate(batch_prompts, sampling_params)

            for j, output in enumerate(batch_outputs):
                meta = batch_metadata[j]
                original_item = meta['original_item']
                
                rewritten_question = output.outputs[0].text.strip()
                
                # 创建新的数据条目，包含所有分组信息
                new_item = {
                    **original_item,
                    'original_question': original_item['question'],
                    'question': rewritten_question,
                    # Fairness evaluation 分组字段
                    'fairness_attribute_category': meta['attribute_category'],
                    'fairness_attribute_name': meta['attribute_name'],
                    'fairness_attribute_value': meta['attribute_value_label']
                }
                
                f.write(json.dumps(new_item, ensure_ascii=False) + "\n")
                f.flush()

    print(f"\n✅ Successfully processed {len(prompts)} prompts.")
    print(f"📊 Results saved to: {OUTPUT_FILE}")
    
    # 显示示例和统计信息
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        print(f"\n📈 Total records generated: {len(lines)}")
        
        if lines:
            example = json.loads(lines[0])
            print("\n--- Example Record ---")
            print(json.dumps({
                "image_path": example.get("image_path", "N/A"),
                "question_type": example.get("question_type", "N/A"),
                "original_question": example["original_question"][:100] + "...",
                "rewritten_question": example["question"][:100] + "...",
                "fairness_attribute_name": example["fairness_attribute_name"],
                "fairness_attribute_value": example["fairness_attribute_value"]
            }, indent=2, ensure_ascii=False))
            print("----------------------")
    
    # 打印分组统计
    print("\n📋 Fairness Groups Summary:")
    for attr in FAIRNESS_ATTRIBUTES:
        print(f"  {attr['attribute_name']}: {len(attr['values'])} values × {len(dataset)} samples = {len(attr['values']) * len(dataset)} records")
    
    print(f"\n💡 Note: Processed only the first {NUM_SAMPLES} samples out of {len(full_dataset)} total samples")

if __name__ == "__main__":
    main()
