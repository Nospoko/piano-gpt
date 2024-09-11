import os
import re
import argparse
import subprocess
from typing import Dict, List

import pandas as pd

from data.tasks import task_map


def run_validation_for_task(model_path: str, device: str, task: str) -> Dict[str, float]:
    cmd = [
        "python",
        "eval.py",
        f"init_from={os.path.basename(model_path)}",
        f"system.device={device}",
        "task=multi",
        f"tasks.list=[{task}]",
    ]

    print(f"Running validation for model: {model_path}, task: {task}")
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    # Extract the validation loss from the output
    output = result.stdout
    match = re.search(r"Validation loss for.*?is (\d+\.\d+)", output, re.DOTALL)
    if match:
        loss = float(match.group(1))
        return {"model": os.path.basename(model_path), "task": task, "loss": loss}
    else:
        print(f"Warning: Couldn't extract loss for {model_path} on task {task}")
        return {"model": os.path.basename(model_path), "task": task, "loss": None}


def main(model_paths: List[str], device: str, tasks: List[str]):
    results = []
    for model_path in model_paths:
        for task in tasks:
            result = run_validation_for_task(model_path, device, task)
            results.append(result)

    # DataFrame from the results
    df = pd.DataFrame(results)

    # Pivot the DataFrame to have tasks as columns
    df_pivot = df.pivot(index="model", columns="task", values="loss")
    df_pivot.columns.name = None
    df_pivot = df_pivot.reset_index()

    output_file = "validation_results.csv"
    df_pivot.to_csv(output_file, index=False)
    print(f"Validation results saved to {output_file}")

    # Display the results
    print("\nValidation Results:")
    print(df_pivot.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run validations for each task separately.")
    parser.add_argument("model_paths", nargs="+", type=str, help="Paths to the model checkpoints.")
    parser.add_argument("device", type=str, help="Device to perform calculations on (e.g., 'cuda' or 'cpu').")
    parser.add_argument(
        "--tasks", nargs="+", default=task_map.keys(), help="List of tasks to run validation on. Default is all tasks."
    )
    args = parser.parse_args()

    main(model_paths=args.model_paths, device=args.device, tasks=args.tasks)
