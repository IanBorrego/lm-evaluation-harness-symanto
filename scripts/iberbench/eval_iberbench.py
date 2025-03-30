"""
Script to evaluate models requested in user-requests.

Usage example to evaluate a single model:
python -m scripts.iberbench.eval_iberbench single --id 8e5ff0da-8d26-4f87-8ebf-dc7ed7b3cb3c

Usage example to evaluate all the models:
python -m scripts.iberbench.eval_iberbench all
"""

import os
import json
import subprocess
import argparse
import sys
import yaml
import gc
from pathlib import Path
import pandas as pd
from typing import Optional
from transformers import AutoModelForCausalLM
from datasets import load_dataset, Dataset, disable_caching
from huggingface_hub import create_repo, HfApi
from huggingface_hub.utils import HfHubHTTPError
from datasets.data_files import EmptyDatasetError

disable_caching()

USER_REQUESTS_DATASET = "iberbench/user-requests"
RESULTS_DATASET = "iberbench/lm-eval-results"
HF_TOKEN = os.environ["HF_API_KEY"]
IBERBENCH_YAML_PATH = "./lm_eval/tasks/iberbench/iberbench.yaml"
# Do not modify RESULTS_PATH!!
RESULTS_PATH = "./iberbench_results"


def load_model_request_from_hub(request_id: str) -> dict:
    """
    Loads a model request from the hub given `request_id`.

    Args:
        request_id (str): the id of the request (id a json file within
                          https://huggingface.co/datasets/iberbench/user-requests/tree/main/data)

    Returns:
        dict: with the model information
    """
    client = HfApi()
    file_name = client.hf_hub_download(
        repo_id=USER_REQUESTS_DATASET,
        filename=f"data/data_{request_id}.json",
        token=HF_TOKEN,
        repo_type="dataset",
    )
    with open(file_name, "r") as fr:
        return json.load(fr)


def get_model_params(model_name: str) -> int:
    """
    Get the number of parameters of a given model.

    Args:
        model_name (str): name of the model

    Returns:
        int: number of model parameters
    """
    model = AutoModelForCausalLM.from_pretrained(model_name)
    num_parameters = model.num_parameters()
    del model
    gc.collect()
    return num_parameters


def create_model_args(model_request: dict) -> str:
    """
    Get the model args as string, from the model request,
    to be use as argument in lm eval harness.

    Args:
        model_request (dict): request dict of the model

    Returns:
        str: model args as string.
    """
    model_args_parts = []
    if model_request["weight_type_option"] != "Original":
        model_args_parts.append(f"pretrained={model_request['base_model_name']}")
        model_args_parts.append(f"peft={model_request['model_name']}")
    else:
        model_args_parts.append(f"pretrained={model_request['model_name']}")

    if model_request["precision_option"] == "GPTQ":
        model_args_parts.append("autogptq=True")
    else:
        model_args_parts.append(f"dtype={model_request['precision_option']}")

    return ",".join(model_args_parts)


def create_hf_results_repo() -> None:
    """
    Creates the results repository if it does not exist in the hub
    """
    try:
        create_repo(RESULTS_DATASET, repo_type="dataset", private=True)
        print(f"Created {RESULTS_DATASET} in the hub.")
    except HfHubHTTPError:
        print(f"{RESULTS_DATASET} already exist in the hub.")


def get_model_results(model_name: str) -> dict:
    """
    Retrieve the results of the model from the local path after evaluation.

    Args:
        model_name (str): name of the model

    Returns:
        dict: map with tasks as keys and macro-f1 scores as values.

    Raises:
        StopIteration: if the json file has not been saved due to
                       an error during evaluation with lm eval
    """
    model_name_dir = model_name.replace("/", "__")
    model_folder = Path(f"{RESULTS_PATH}/{model_name_dir}")
    model_results = {}

    json_file = next(model_folder.glob("*.json"))
    with open(json_file, "r") as fr:
        content = json.load(fr)

    results = content["results"]
    for task in results:
        if "f1,none" in results[task]:
            model_results[task] = results[task]["f1,none"]
        elif "iberbench_seqeval,none" in results[task]:
            model_results[task] = results[task]["iberbench_seqeval,none"]
        elif "rouge1,none" in results[task]:
            model_results[task] = results[task]["rouge1,none"]
        else:
            model_results[task] = results[task]["acc,none"]
    # Remove the json file to avoid inconsistencies
    # We have to be careful with this, so let's add
    # redundant checkers to ensure it is a real file
    # and not a symbolic link
    if json_file.is_file() and not json_file.is_symlink():
        json_file.unlink()

    return model_results


def load_results_dataset() -> Optional[Dataset]:
    """
    Load the results dataset from the `lm-eval-results`
    repository, avoiding to load it from the cache by
    forcing always a fresh download from the hub.

    Returns:
        Optional[Dataset]: None if the hub repository does not exist or
                           if it is empty. The results dataset otherwise.
    """
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
    """
    Updates the results of the model. Loads the results
    from the local `RESULTS_PATH` of the model and update
    the hub with them.

    1) If the results repository do not exist in the hub, this
       function creates it and upload the results row
    2) Otherwise (the results repository exists):
    2.1) If the model exists, the we check if there are new tasks in local
         that are not included in the results repository. If so, we add
         the new columns with None values for all the other models and
         update the row with the new results for this model.
    2.2) If the model do not exist, then it is added a new row with its
         results into the results repository.

    Args:
        model_request (dict): model info in the request
    """
    model_name = model_request["model_name"]
    model_type = model_request["model_type"]
    num_parameters = model_request["num_parameters"]

    # Get results of this model run in local
    model_results = get_model_results(model_name)

    # Get current results in the hub for this model
    results_dataset = load_results_dataset()

    # If no results dataset, we have to create it with the results of this model
    if results_dataset is None:
        model_results = {
            "model_name": model_name,
            "model_type": model_type,
            "num_parameters": num_parameters,
        } | model_results
        results_dataset = Dataset.from_pandas(pd.DataFrame([model_results]))

    # If the results dataset exists and it is not empty
    else:
        # First check if the model already exists
        model_prev_results = results_dataset.filter(
            lambda x: x == model_name, input_columns=["model_name"]
        )
        # If the model exists:
        if len(model_prev_results) > 0:
            # If we are adding new tasks that are not in the hub dataset,
            # we have to extend the hub dataset with None values.
            # Take care later when averaging in the interface.
            results_tasks = set(model_results.keys())
            prev_results_tasks = set(model_prev_results.features.keys())
            new_tasks = results_tasks.difference(prev_results_tasks)
            if len(new_tasks) > 0:
                print("Adding new tasks to the hub dataset:", new_tasks)
                results_dataset = results_dataset.map(
                    lambda example: {feature: None for feature in new_tasks} | example
                )
            # And then assign the scores of this model
            # which includes scores for new tasks
            model_results = model_prev_results[0] | model_results
            results_dataset = results_dataset.to_pandas()
            model_index = results_dataset[
                results_dataset["model_name"] == model_name
            ].index
            results_dataset.iloc[model_index, :] = pd.DataFrame([model_results])
            results_dataset = Dataset.from_pandas(results_dataset)
        # If the model doesn't exist, just add the row
        else:
            results_dataset = results_dataset.add_item(
                {
                    "model_name": model_name,
                    "model_type": model_type,
                    "num_parameters": num_parameters,
                }
                | model_results
            )

    # Push the dataset to the hub
    create_hf_results_repo()
    results_dataset.push_to_hub(RESULTS_DATASET, private=True)
    print(f"Successfully updated the dataset on the hub: {RESULTS_DATASET}")


def get_pending_tasks(model_name: str, task_list: list[str]) -> list[str]:
    """
    Get the tasks that have been not already computed by the model.

    Args:
        model_name (str): name of the model
        task_list (list[str]): list of tasks within the IberBench YAML file

    Returns:
        list[str]: tasks in the YAML file that have not been already evaluated
                   (not in the `lm-eval-results` repository for this model)
    """
    results_dataset = load_results_dataset()

    # If the `lm-eval-results` do not exists or it is empty
    # we have to do all the tasks in `task_list`
    if results_dataset is None:
        return task_list

    # If `lm-eval-results` exists but the model is not in
    # the repo, we have to do all the tasks in `task_list`
    model_prev_results = results_dataset.filter(
        lambda x: x == model_name, input_columns=["model_name"]
    )
    if len(model_prev_results) == 0:
        return task_list

    # Otherwise, the model is in the repo, we have to do
    # all the tasks in which the model doesn't have results
    row = model_prev_results[0]
    columns = list(row.keys())
    completed_tasks = [
        column
        for column in columns
        if column != "model_name" and row[column] is not None
    ]

    return set(task_list).difference(set(completed_tasks))


def eval_all_models() -> None:
    """
    Run the evaluation of all the models requested in the user-requests
    repository, by calling `eval_single_model` for each request id.
    """
    client = HfApi()
    repo_files = client.list_repo_files(
        repo_id=USER_REQUESTS_DATASET, token=HF_TOKEN, repo_type="dataset"
    )
    request_ids = [
        file_name.split("/")[1].removeprefix("data_").removesuffix(".json")
        for file_name in repo_files
        if file_name.startswith("data/data_")
    ]

    for request_id in request_ids:
        try:
            eval_single_model(request_id)
        except Exception as e:
            print(f"There was an error evaluating the model with id={request_id}: {e}")


def eval_single_model(request_id: str) -> None:
    """
    Run the evaluation of a requested model using lm eval harness.

    This function checks what tasks has been already run by a model,
    and avoids recomputing them. Only computes new tasks that are
    in the YAML config of IberBench, but do not appear in the row
    of the model results in the `lm-eval-results` repository.

    The model request is not removed from the huggingface hub, so
    it could be evaluated in the future if new tasks are added.

    Args:
        request_id (str): the id of the request (id a json file within
                          https://huggingface.co/datasets/iberbench/user-requests/tree/main/data)
    """
    # Create results path
    Path(RESULTS_PATH).mkdir(parents=True, exist_ok=True)

    # Get the model request
    model_request = load_model_request_from_hub(request_id)
    model_name = model_request["model_name"]

    # Get model args to call lm eval
    model_args = create_model_args(model_request)

    # Get all the IberBench tasks
    with open(IBERBENCH_YAML_PATH, "r") as file:
        data = yaml.safe_load(file)

    all_tasks = data.get("tasks", [])

    # Filter the tasks for which this model has been already evaluated
    pending_tasks = get_pending_tasks(model_name, all_tasks)

    if not pending_tasks:
        print(f"No pending tasks for model {model_name} with id {request_id}.")

    else:
        print(f"Evaluating {model_name} on pending tasks: {pending_tasks}")

        # Add model num params to the model request
        model_request["num_parameters"] = get_model_params(model_name)

        # Run lm eval
        command = [
            "accelerate",
            "launch",
            "-m",
            "lm_eval",
            "--model",
            "hf",
            "--model_args",
            model_args,
            "--tasks",
            ",".join(pending_tasks),
            "--batch_size",
            "auto:4",
            "--output_path",
            RESULTS_PATH,
            "--seed",
            "13",
        ]
        subprocess.run(command, stdout=sys.stdout, stderr=sys.stderr, text=True)

        # Update the hub with the new results
        update_hub_results(model_request)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to evaluate user-requested models in IberBench"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for evaluating all models
    parser_all = subparsers.add_parser(
        "all",
        help="Evaluate all models requested in the user-requests repository",
    )

    # Subparser for evaluating a single model
    parser_single = subparsers.add_parser(
        "single", help="Evaluate a single model based on request ID"
    )
    parser_single.add_argument(
        "--id",
        type=str,
        required=True,
        help="Request ID from iberbench/user-requests (id in the json file)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Execute the chosen command
    if args.command == "all":
        eval_all_models()
    elif args.command == "single":
        eval_single_model(args.id)
