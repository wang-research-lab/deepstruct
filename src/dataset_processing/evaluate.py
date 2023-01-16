# Adapted from https://github.com/amazon-science/tanl
from typing import List, Dict
import torch
import logging
import numpy as np
from transformers import PreTrainedTokenizer

from arguments import DataTrainingArguments
from datasets import load_dataset


def get_avg_results(results: List[dict]) -> dict:
    """
    Compute average results and standard deviation from many episodes.
    """
    aggregate_results = {'num_episodes': len(results)}

    for key in results[0]:
        try:
            numbers = np.array([res[key] for res in results])
            aggregate_results[key] = (numbers.mean(), numbers.std())

        except:
            pass

    return aggregate_results


TASK_METRIC_MAPPING = {
    "ace2005event_argument": [{'key': 'relation_f1_no_type', 'tkey': 'Argument Id F1'},
                              {'key': 'relation_f1', 'tkey': 'Argument Cl F1'}]
}


def print_results(results: dict):

    header = f"########## ace2005event_argument Evaluation ##########"
    print(header)
    for metric in TASK_METRIC_MAPPING['ace2005event_argument']:
        print(f"{metric['tkey']:20}: {results[metric['key']][0]}")
    print("#" * len(header))














def evaluate(model, dataset_name: str, data_args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, split: str,
             seed: int, gpu: int, batch_size: int, mode: str = 'default') -> Dict[str, float]:
    """
    Evaluate a model on some dataset.
    """
    device = torch.device("cuda", gpu)
    test_dataset = load_dataset(
        dataset_name, data_args,
        max_input_length=data_args.max_seq_length_eval,
        max_output_length=data_args.max_output_seq_length_eval,
        tokenizer=tokenizer, split=split, seed=seed, shuffle=False, is_eval=True,
    )

    return test_dataset.evaluate_dataset(
        data_args=data_args, model=model, device=device, batch_size=batch_size, mode=mode,
        external=data_args.data_dir + dataset_name + "/" + "test.jsonl.hyps"
    )
