#!/bin/bash

# Check if model name argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <model_name>"
    exit 1
fi

model_name=$1

run_commands() {
    echo "Starting metrics calculation at $(date)"
    command1="python3.10 -m scripts.download_model $model_name"
    command2="python3.10 -m gpt2.high_level_piano_eval logging.wandb_log=true stage=piano_task init_from='tmp/checkpoints/$model_name'"

    # Run the commands sequentially
    eval "$command1"
    eval "$command2"
    echo "Finished commands at $(date)"
}

# Main loop
while true; do
    # Record start time
    start_time=$(date +%s)

    # Run the commands
    run_commands

    # Calculate how long the commands took
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    # If duration was less than 30 minutes, sleep for the remaining time
    if [ $duration -lt 1800 ]; then
        sleep_time=$((1800 - duration))
        echo "Sleeping for $sleep_time seconds..."
        sleep $sleep_time
    else
        echo "Execution took longer than 30 minutes, running again immediately"
    fi
done
