#!/bin/bash

# Batch fairness evaluation script for multiple vision-language models (V2)
# Evaluates model performance across different fairness subgroups

# Configuration
DATASET="JiayiHe/Fair_MedXpertQA_subset"
TASK="MM"
SPLIT="train"
TENSOR_PARALLEL_SIZE=1
BATCH_SIZE=512
TEMPERATURE=0
OUTPUT_DIR="outputs_fairness/InternVL3_new_2"
PYTHON_SCRIPT="fairness_eval.py"

# Define models to evaluate
MODELS=(
     #"Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
     #"OpenGVLab/InternVL3_5-14B-Instruct"
     #"Qwen/Qwen2.5-VL-3B-Instruct"
     "OpenGVLab/InternVL3-14B-Instruct"
     "OpenGVLab/InternVL3-38B"
     "OpenGVLab/InternVL3-38B-Instruct"
     #"OpenGVLab/InternVL3-38B"
     #"OpenGVLab/InternVL3-38B-Instruct"
     #"OpenGVLab/InternVL3_5-38B"
     #"OpenGVLab/InternVL3_5-8B"
)

# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Create main output directory
mkdir -p "${OUTPUT_DIR}"

# Summary files
SUMMARY_FILE="${OUTPUT_DIR}/fairness_evaluation_summary.txt"
OVERALL_CSV="${OUTPUT_DIR}/overall_fairness_results.csv"
DETAILED_CSV="${OUTPUT_DIR}/detailed_fairness_results.csv"

# Initialize summary file
echo "=================================================================" > "${SUMMARY_FILE}"
echo "Fairness Evaluation Summary - MedXpertQA (V2 Subgroup Analysis)" >> "${SUMMARY_FILE}"
echo "Dataset: ${DATASET}" >> "${SUMMARY_FILE}"
echo "Task: ${TASK}" >> "${SUMMARY_FILE}"
echo "Split: ${SPLIT}" >> "${SUMMARY_FILE}"
echo "Date: $(date)" >> "${SUMMARY_FILE}"
echo "=================================================================" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

# Initialize CSV files
echo "Model,Overall Accuracy,Total Samples,Correct,Status,Timestamp" > "${OVERALL_CSV}"
echo "Model,Subgroup,Accuracy,Correct,Total,Status" > "${DETAILED_CSV}" # CHANGED: Attribute -> Subgroup

# Function to extract model name
get_model_name() {
    echo "$1" | awk -F'/' '{print $NF}'
}

# Function to parse fairness summary JSON
parse_fairness_summary() {
    local summary_file=$1
    local model_name=$2
    
    if [ ! -f "${summary_file}" ]; then
        echo -e "${RED}Summary file not found: ${summary_file}${NC}"
        return 1
    fi
    
    # Extract overall stats
    local overall_acc=$(python3 -c "import json; data=json.load(open('${summary_file}')); print(f\"{data['overall_accuracy']:.4f}\")" 2>/dev/null)
    local total=$(python3 -c "import json; data=json.load(open('${summary_file}')); print(data['total_samples'])" 2>/dev/null)
    local correct=$(python3 -c "import json; data=json.load(open('${summary_file}')); print(data['correct'])" 2>/dev/null)
    
    if [ -z "${overall_acc}" ]; then
        echo -e "${RED}Failed to parse summary file${NC}"
        return 1
    fi
    
    # Display overall results
    echo -e "${CYAN}Overall Results:${NC}"
    echo -e "  Overall Accuracy: ${GREEN}${overall_acc}${NC} (${correct}/${total})"
    
    # Append to overall CSV
    echo "${model_name},${overall_acc},${total},${correct},SUCCESS,$(date -Iseconds)" >> "${OVERALL_CSV}"
    
    # Display per-subgroup results
    echo -e "${CYAN}Per-Subgroup Results:${NC}" # CHANGED: Attribute -> Subgroup
    
    # Extract subgroup stats using Python
    python3 << EOF
import json
import sys

try:
    with open('${summary_file}', 'r') as f:
        data = json.load(f)
    
    # CHANGED: Use 'per_subgroup_stats' key from the v2 script
    per_subgroup = data.get('per_subgroup_stats', {})
    
    # Print formatted results
    for group in sorted(per_subgroup.keys()):
        stats = per_subgroup[group]
        acc = stats['accuracy']
        correct = stats['correct']
        total = stats['total']
        print(f"  {group:30s} {acc:.4f} ({correct}/{total})")
        
        # Append to detailed CSV
        with open('${DETAILED_CSV}', 'a') as csv_f:
            csv_f.write(f"${model_name},{group},{acc:.4f},{correct},{total},SUCCESS\n")
except Exception as e:
    print(f"Error parsing summary: {e}", file=sys.stderr)
    sys.exit(1)
EOF
    
    return 0
}

# Function to evaluate a single model
evaluate_model() {
    local model=$1
    local model_name=$(get_model_name "${model}")
    local dataset_name=$(echo "${DATASET}" | awk -F'/' '{print $NF}') # CHANGED: Dynamically get dataset name
    
    echo -e "${BLUE}=================================================================${NC}"
    echo -e "${BLUE}Starting fairness evaluation for: ${model}${NC}"
    echo -e "${BLUE}=================================================================${NC}"
    
    # CHANGED: Use dynamic dataset_name for the path to match python script's output
    local model_output_dir="${OUTPUT_DIR}/${model_name}/${dataset_name}/${TASK}"
    mkdir -p "${model_output_dir}"
    
    # Log file
    local log_file="${model_output_dir}/evaluation.log"
    
    # Run evaluation
    echo -e "${YELLOW}Running fairness evaluation...${NC}"
    python "${PYTHON_SCRIPT}" \
        --model "${model}" \
        --dataset "${DATASET}" \
        --task "${TASK}" \
        --split "${SPLIT}" \
        --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
        --batch-size ${BATCH_SIZE} \
        --temperature ${TEMPERATURE} \
        --output-dir "${OUTPUT_DIR}" \
        2>&1 | tee "${log_file}"
    
    # Check if evaluation succeeded
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo -e "${GREEN}✓ Fairness evaluation completed for ${model_name}${NC}"
        echo ""
        
        # Parse and display results
        local summary_json="${model_output_dir}/${SPLIT}_fairness_summary.json"
        
        if parse_fairness_summary "${summary_json}" "${model_name}"; then
            # Append to summary file
            echo "=================================================================" >> "${SUMMARY_FILE}"
            echo "Model: ${model}" >> "${SUMMARY_FILE}"
            echo "=================================================================" >> "${SUMMARY_FILE}"
            cat "${model_output_dir}/${SPLIT}_fairness_report.txt" >> "${SUMMARY_FILE}"
            echo "" >> "${SUMMARY_FILE}"
        else
            echo -e "${RED}Failed to parse results for ${model_name}${NC}"
            echo "Model: ${model}" >> "${SUMMARY_FILE}"
            echo "Status: COMPLETED (parsing failed)" >> "${SUMMARY_FILE}"
            echo "" >> "${SUMMARY_FILE}"
        fi
    else
        echo -e "${RED}✗ Fairness evaluation failed for ${model_name}${NC}"
        echo "Model: ${model}" >> "${SUMMARY_FILE}"
        echo "Status: FAILED" >> "${SUMMARY_FILE}"
        echo "Timestamp: $(date)" >> "${SUMMARY_FILE}"
        echo "" >> "${SUMMARY_FILE}"
        
        echo "${model_name},N/A,N/A,N/A,FAILED,$(date -Iseconds)" >> "${OVERALL_CSV}"
    fi
    
    echo ""
}

# Main execution
echo -e "${GREEN}Starting fairness evaluation of ${#MODELS[@]} models${NC}"
echo -e "${GREEN}Results will be saved to: ${OUTPUT_DIR}${NC}"
echo ""

# Record start time
START_TIME=$(date +%s)

# Evaluate each model
for model in "${MODELS[@]}"; do
    evaluate_model "${model}"
    echo -e "${YELLOW}Waiting 10 seconds before next model...${NC}"
    sleep 10
done

# Record end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

# Create comparison table
echo -e "${BLUE}=================================================================${NC}"
echo -e "${BLUE}Creating comparison tables...${NC}"
echo -e "${BLUE}=================================================================${NC}"

# Generate comparison CSV with all subgroups
COMPARISON_FILE="${OUTPUT_DIR}/fairness_comparison_table.csv"

# --- FIX: Export DATASET variable ---
export OUTPUT_DIR
export TASK
export SPLIT
export DATASET # <--- 确保 DATASET 被导出
# --- END FIX ---

python3 << 'EOFPYTHON'
import json
import csv
import os
import sys
from pathlib import Path

# Use os.environ.get and handle None explicitly
output_dir = os.environ.get('OUTPUT_DIR')
task = os.environ.get('TASK')
split = os.environ.get('SPLIT')
dataset_path = os.environ.get('DATASET')

if not dataset_path:
    print("Error: DATASET environment variable is not set.", file=sys.stderr)
    sys.exit(1)

dataset_name = dataset_path.split('/')[-1]

# Collect all models and their results
models_data = []

for model_dir in Path(output_dir).iterdir():
    if not model_dir.is_dir():
        continue
    
    # Construct the expected path based on the Python script's save logic
    summary_path = model_dir / dataset_name / task / f'{split}_fairness_summary.json'
    
    if summary_path.exists():
        try:
            with open(summary_path, 'r') as f:
                data = json.load(f)
            
            model_name = model_dir.name
            overall_acc = data['overall_accuracy']
            # CHANGED: Use 'per_subgroup_stats' key
            per_subgroup = data.get('per_subgroup_stats', {})
            
            model_row = {
                'Model': model_name,
                'Overall': f"{overall_acc:.4f}"
            }
            
            for group, stats in per_subgroup.items():
                model_row[group] = f"{stats['accuracy']:.4f}"
            
            models_data.append(model_row)
        except Exception as e:
            print(f"Error processing {model_name}: {e}", file=sys.stderr)

if not models_data:
    print("No model results found!", file=sys.stderr)
    sys.exit(1)

# Get all subgroup names
all_subgroups = set()
for model_row in models_data:
    all_subgroups.update(k for k in model_row.keys() if k not in ['Model', 'Overall'])

# Sort subgroups
sorted_subgroups = sorted(list(all_subgroups))

# Write comparison CSV
comparison_file = Path(output_dir) / 'fairness_comparison_table.csv'
with open(comparison_file, 'w', newline='') as f:
    fieldnames = ['Model', 'Overall'] + sorted_subgroups
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    
    writer.writeheader()
    for model_row in sorted(models_data, key=lambda x: x['Model']):
        # Fill missing subgroups with 'N/A'
        for group in sorted_subgroups:
            if group not in model_row:
                model_row[group] = 'N/A'
        writer.writerow(model_row)

print(f"Comparison table saved to: {comparison_file}")
EOFPYTHON

# Final summary
echo -e "${GREEN}=================================================================${NC}"
echo -e "${GREEN}All fairness evaluations completed!${NC}"
echo -e "${GREEN}=================================================================${NC}"
echo -e "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo -e "Results saved to:"
echo -e "  Main summary: ${BLUE}${SUMMARY_FILE}${NC}"
echo -e "  Overall CSV: ${BLUE}${OVERALL_CSV}${NC}"
echo -e "  Detailed CSV: ${BLUE}${DETAILED_CSV}${NC}"
echo -e "  Comparison table: ${BLUE}${OUTPUT_DIR}/fairness_comparison_table.csv${NC}"
echo ""

# Add timing to summary
echo "=================================================================" >> "${SUMMARY_FILE}"
echo "Total execution time: ${HOURS}h ${MINUTES}m ${SECONDS}s" >> "${SUMMARY_FILE}"
echo "Completed: $(date)" >> "${SUMMARY_FILE}"
echo "=================================================================" >> "${SUMMARY_FILE}"

# Display final comparison table
echo -e "${CYAN}Final Fairness Comparison:${NC}"
echo ""
if [ -f "${OUTPUT_DIR}/fairness_comparison_table.csv" ]; then
    # Use column for better formatting
    column -t -s ',' "${OUTPUT_DIR}/fairness_comparison_table.csv"
else
    echo -e "${RED}Comparison table not generated${NC}"
fi

echo ""
echo -e "${GREEN}Done!${NC}"