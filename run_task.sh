#!/bin/bash

tasks=("iberlef_iberautextification_2024")

for task in "${tasks[@]}"; do
    start_time=$(date +%s)  # Record the start time
    echo "running task $task"    
    CUDA_VISIBLE_DEVICES="0,2" lm_eval --model hf \
        --model_args pretrained="EleutherAI/pythia-2.8b",trust_remote_code=True,parallelize=True,max_length=2048 \
        --tasks $task \
        --batch_size auto:16 \
        --limit 1000 \
        --trust_remote_code \
        --device cuda > "shell_scripts/test_${task}_log.txt" 2>&1
    end_time=$(date +%s)  # Record the end time
    elapsed_time=$((end_time - start_time))  # Calculate the elapsed time
    echo "$task task completed in $elapsed_time seconds."
done