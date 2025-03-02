#!/bin/bash

run_commands() {
    echo "Starting model upload at $(date)"

    command1="python3.10 -m scripts.upload_models"

    eval "$command1"

    echo "Finished command at $(date)"
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
