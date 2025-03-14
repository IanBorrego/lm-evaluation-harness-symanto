"""
Usage example:
python -m scripts.eval_iberbench --id 8e5ff0da-8d26-4f87-8ebf-dc7ed7b3cb3c --batch_size 4
"""
import os
import json
import subprocess
import argparse
import sys
import yaml
from pathlib import Path
import pandas as pd
from typing import Optional
from datasets import load_dataset, Dataset, disable_caching
from huggingface_hub import create_repo, HfApi
from huggingface_hub.utils import HfHubHTTPError
from datasets.data_files import EmptyDatasetError

disable_caching()

USER_REQUESTS_DATASET = "iberbench/user-requests"
RESULTS_DATASET = "iberbench/lm-eval-results"
HF_TOKEN = os.environ["HF_API_KEY"]
IBERBENCH_YAML_PATH = "./lm_eval/tasks/iberbench/iberbench.yaml"
RESULTS_PATH = "./iberbench_results"


def load_model_request_from_hub(request_id: str):
    client = HfApi()
    file_name = client.hf_hub_download(
        repo_id=USER_REQUESTS_DATASET,
        filename=f"data/data_{request_id}.json",
        token=HF_TOKEN,
        repo_type="dataset",
    )
    with open(file_name, "r") as fr:
        return json.load(fr)

def create_model_args(model_request: dict) -> str:
    model_args = f"pretrained={model_request['model_name']}"
    if model_request["precision_option"] == "GPTQ":
        model_args = f"{model_args},autogptq=True"
    else:
        model_args = f"{model_args},dtype={model_request['precision_option']}"

    if model_request["weight_type_option"] != "Original":
        model_args = f"{model_args},peft={model_request['base_model_name']}"
    return model_args


def create_hf_results_repo():
    try:
        create_repo(RESULTS_DATASET, repo_type="dataset", private=True)
        print(f"Created {RESULTS_DATASET} in the hub.")
    except HfHubHTTPError:
        print(f"{RESULTS_DATASET} already exist in the hub.")


def get_model_results(model_name: str) -> dict:
    """
    Retrieve the results of the model from the local path after evaluation.
    """
    model_name_dir = model_name.replace("/", "__")
    model_folder = Path(f"{RESULTS_PATH}/{model_name_dir}")
    results = {}
    for task_json in model_folder.glob("*.json"):
        with open(task_json, "r") as fr:
            content = json.load(fr)
        task_name = list(content["results"])[0]
        task_results = list(content["results"].values())[0]
        results[task_name] = task_results["f1,none"]
    return results


def load_results_dataset() -> Optional[Dataset]:
    client = HfApi()
    exist_dataset = (
        len(
            list(
                client.list_datasets(
                    author="iberbench",
                    dataset_name=RESULTS_DATASET.split("/")[1],
                    token=HF_TOKEN,
                )
            )
        )
        > 0
    )
    if not exist_dataset:
        return None
    try:
        return load_dataset(
            RESULTS_DATASET, split="train", download_mode="force_redownload"
        )
    except EmptyDatasetError:
        return None


def update_hub_results(model_request: dict) -> None:
    model_name = model_request["model_name"]
    model_results = get_model_results(model_name)

    # Get current results in the hub for this model
    results_dataset = load_results_dataset()

    # If no results dataset, we have to create it with the results of this model
    if results_dataset is None:
        model_results = {**{"model_name": model_name}, **model_results}
        results_dataset = Dataset.from_pandas(pd.DataFrame([model_results]))

    # Otherwise, check if the model already exists
    else:
        model_prev_results = results_dataset.filter(
            lambda x: x == model_name, input_columns=["model_name"]
        )
        # When the model exists:
        if len(model_prev_results) > 0:
            # If we are adding new columns that are not in the hub dataset,
            # we have to extend the hub dataset with None values.
            # Take care later when averaging in the interface.
            results_tasks = set(model_results.keys())
            prev_results_tasks = set(model_prev_results.features.keys())
            new_tasks = results_tasks.difference(prev_results_tasks)
            if len(new_tasks) > 0:
                print("Adding new tasks to the hub dataset:", new_tasks)
                results_dataset = results_dataset.map(
                    lambda example: {
                        **example,
                        **{feature: None for feature in new_tasks},
                    }
                )
            # And then assign the values of this model
            # which is including values for new columns
            model_results = {**model_prev_results[0], **model_results}
            results_dataset = results_dataset.to_pandas()
            results_dataset.loc[results_dataset["model_name"] == model_name] = (
                pd.DataFrame([model_results])
            )
            results_dataset = Dataset.from_pandas(results_dataset)
        # If the model doesn't exist, just add the row
        else:
            results_dataset = results_dataset.add_item(model_results)

    # Push the dataset to the hub
    create_hf_results_repo()
    results_dataset.push_to_hub(RESULTS_DATASET, private=True)
    print(
        f"Successfully updated the dataset on the Hugging Face Hub: {RESULTS_DATASET}"
    )


def get_pending_tasks(model_name: str, task_list: list[str]) -> list[str]:
    """
    Get the tasks that have been not already computed by the model.
    """
    results_dataset = load_results_dataset()
    if results_dataset is None:
        return task_list

    model_prev_results = results_dataset.filter(
        lambda x: x == model_name, input_columns=["model_name"]
    )
    if len(model_prev_results) == 0:
        return task_list

    row = model_prev_results[0]
    columns = list(row.keys())
    completed_tasks = [
        column
        for column in columns
        if column != "model_name" and row[column] is not None
    ]

    return set(task_list).difference(set(completed_tasks))


def main(request_id: str, batch_size: int) -> None:
    # Create results path
    Path(RESULTS_PATH).mkdir(parents=True, exist_ok=True)

    # Get the model request
    model_request = load_model_request_from_hub(request_id)

    # Get model args to call lm eval
    model_args = create_model_args(model_request)

    # Get the tasks to run
    with open(IBERBENCH_YAML_PATH, "r") as file:
        data = yaml.safe_load(file)

    all_tasks = data.get("tasks", [])

    # Filter the tasks for which this model has been already evaluated
    pending_tasks = get_pending_tasks(model_request["model_name"], all_tasks)
    print("Evaluating on pending tasks:", pending_tasks)

    # Run lm eval
    for task in pending_tasks:
        print(f"Running {task}")
        command = [
            "lm_eval",
            "--model",
            "hf",
            "--model_args",
            model_args,
            "--tasks",
            task,
            "--batch_size",
            f"{batch_size}",
            "--output_path",
            RESULTS_PATH,
        ]
        subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr, text=True)

    # Update the hub with the new results
    update_hub_results(model_request)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to eval a user requested model in iberbench"
    )

    # adding arguments
    parser.add_argument(
        "--id",
        type=str,
        required=True,
        help="Request id from iberbench/user-requests (id in the json file)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size used in lm_eval"
    )

    # run process
    args = parser.parse_args()
    main(args.id, args.batch_size)
