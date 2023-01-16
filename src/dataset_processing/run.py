# Adapted from https://github.com/amazon-science/tanl
import argparse
import configparser
import itertools
import json
import logging
import os
from collections import defaultdict
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, AutoModelForSeq2SeqLM, Trainer

from arguments import ModelArguments, DataTrainingArguments, TrainingArguments
from datasets import load_dataset
from evaluate import evaluate, get_avg_results, print_results
from utils import get_episode_indices


def main():
    assert torch.cuda.is_available(), 'CUDA not available'


    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    parser.add_argument('-c', '--config_file', type=str, default='config.ini', help='configuration file')
    parser.add_argument('-e', '--eval', action='store_true', default=False, help='run evaluation only')
    parser.add_argument('--evaluate_checkpoints', action='store_true', default=False,
                        help='evaluate intermediate checkpoints instead of the final model')
    parser.add_argument('--evaluate_last_checkpoint', action='store_true', default=False,
                        help='evaluate the last intermediate checkpoint instead of the final model')
    parser.add_argument('--evaluate_checkpoint_in_dir', type=str, default=None,
                        help='evaluate the checkpoint in the given directory')
    parser.add_argument('-a', '--evaluate_all', action='store_true', default=False,
                        help='evaluate intermediate checkpoints together with the final model')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='which GPU to use for evaluation')
    parser.add_argument('-v', '--verbose_results', action='store_true', default=False,
                        help='print results for each evaluation run')
    parser.add_argument('-mode', dest='mode', type=str, default='default', help='mode')
    parser.add_argument('--data_only', dest='data_only', action='store_true', default=False,
                        help='Data generation only')
    parser.add_argument('--evaluate_only', dest='evaluate_only', action='store_true', default=False,
                        help='Evaluation only')
    parser.add_argument('--debug', dest='debug', action='store_true', default=False,
                        help='Debug mode (small dev, test set)')
    args, remaining_args = parser.parse_known_args()


    config = configparser.ConfigParser(allow_no_value=False)
    config.read(args.config_file)
    job = args.job
    assert job in config


    defaults = {
        'overwrite_output_dir': True,
        'overwrite_cache': True,
        'per_device_eval_batch_size': 4,
        'learning_rate': 5e-4,
        'logging_steps': 5000,
        'save_steps': 0,
    }


    defaults.update(dict(config.items(job)))
    for key in defaults:
        if defaults[key] in ['True', 'False']:

            defaults[key] = config.getboolean(job, key)
        if defaults[key] == 'None':

            defaults[key] = None

    if args.eval:

        defaults['do_train'] = False


    second_parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    second_parser.set_defaults(**defaults)

    model_args, data_args, training_args = second_parser.parse_args_into_dataclasses(remaining_args)
    training_args.do_train = training_args.do_train and not args.evaluate_only
    training_args.do_eval = training_args.do_eval and not args.evaluate_only

    try:
        os.mkdir(training_args.output_dir)
    except FileExistsError:
        pass


    if data_args.max_output_seq_length_eval is None:

        data_args.max_output_seq_length_eval = data_args.max_output_seq_length \
                                               or data_args.max_seq_length_eval \
                                               or data_args.max_seq_length

    if data_args.max_output_seq_length is None:

        data_args.max_output_seq_length = data_args.max_seq_length

    if data_args.max_seq_length_eval is None:

        data_args.max_seq_length_eval = data_args.max_seq_length

    if data_args.chunk_size_eval is None:

        data_args.chunk_size_eval = data_args.chunk_size

    if data_args.chunk_overlap_eval is None:

        data_args.chunk_overlap_eval = data_args.chunk_overlap



    output_dir = os.path.join(
        training_args.output_dir,
        f'{args.job}'
        f'-{model_args.model_name_or_path.split("/")[-1]}'
        f'-ep{round(training_args.num_train_epochs)}'
        f'-len{data_args.max_seq_length}'
    )

    if data_args.max_output_seq_length != data_args.max_seq_length:
        output_dir += f'-{data_args.max_output_seq_length}'

    if training_args.learning_rate != 5e-4:
        output_dir += f'-lr{training_args.learning_rate}'

    output_dir += f'-b{training_args.per_device_train_batch_size}' \
                  f'-{data_args.train_split}'

    if data_args.chunk_size != 128:
        output_dir += f'-chunk{data_args.chunk_size}'
    if data_args.chunk_overlap != 64:
        output_dir += f'-overlap{data_args.chunk_overlap}'

    if data_args.output_format is not None:
        output_dir += f'-{data_args.output_format}'
    if data_args.input_format is not None:
        output_dir += f'-{data_args.input_format}'
    if data_args.train_subset < 1:
        output_dir += f'-size{data_args.train_subset:.2f}'

    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass


    logging.basicConfig(
      filename=os.path.join(output_dir, 'logs.log'),
      format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
      datefmt='%Y-%m-%d %H:%M:%S',
      level=logging.INFO,
    )
    logging.getLogger().addHandler(logging.StreamHandler())


    evaluation_output_filename = f'results'
    if data_args.num_beams is not None:
        evaluation_output_filename += f'-{data_args.num_beams}beams'
    if data_args.max_seq_length_eval is not None:
        evaluation_output_filename += f'-len{data_args.max_seq_length_eval}'


    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )


    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    )


    dataset_names = data_args.datasets.split(',')


    if args.job.startswith('ade'):
        episode_indices = [int(args.job[-1])]
    else:
        episode_indices = [-1]



    evaluation_results = defaultdict(list)
    for ep_idx in episode_indices:
        print()
        logging.info(f'Episode {ep_idx} ({len(episode_indices)} episodes total)')
        episode_output_dir = os.path.join(output_dir, f'episode{ep_idx}')

        try:
            os.mkdir(episode_output_dir)
        except FileExistsError:
            pass

        logging.info(f'Output directory: {episode_output_dir}')

        training_args.output_dir = episode_output_dir


        model = None


        if training_args.do_train:

            datasets = []
            print(dataset_names)
            for dataset_name in dataset_names:
                logging.info(f'Process dataset {dataset_name} (train)')
                dataset = load_dataset(
                    dataset_name, data_args, split=data_args.train_split,
                    max_input_length=data_args.max_seq_length, max_output_length=data_args.max_output_seq_length,
                    tokenizer=tokenizer, seed=ep_idx, train_subset=data_args.train_subset,
                )


                dataset.preprocess_for_glm(mode=args.mode,dataset=dataset_name,debug=args.debug)

                datasets.append(dataset)


            if args.data_only:
                exit(0)


        if training_args.local_rank in [-1, 0] and (training_args.do_eval or training_args.do_predict):

            evaluation_splits = []
            if training_args.do_eval:
                evaluation_splits.append('dev')
            if training_args.do_predict:
                evaluation_splits.append('test')


            evaluation_dirs = []

            if args.evaluate_checkpoints or args.evaluate_last_checkpoint or \
                    args.evaluate_checkpoint_in_dir or args.evaluate_all:

                evaluation_dirs = list(sorted([
                    checkpoint_dir
                    for checkpoint_dir in os.listdir(episode_output_dir)
                    if checkpoint_dir.startswith('checkpoint-')
                ], key=lambda x: int(x[len('checkpoint-'):])))
                if args.evaluate_last_checkpoint:

                    evaluation_dirs = [evaluation_dirs[-1]]
                elif args.evaluate_checkpoint_in_dir:
                    assert args.evaluate_checkpoint_in_dir in evaluation_dirs, \
                        "checkpoint {} does not exist".format(args.evaluate_checkpoint_in_dir)
                    evaluation_dirs = [args.evaluate_checkpoint_in_dir]

            if args.evaluate_all or (not args.evaluate_checkpoints and not args.evaluate_last_checkpoint):

                evaluation_dirs += ['']


            if data_args.eval_datasets is None:
                eval_dataset_names = dataset_names
            else:
                eval_dataset_names = data_args.eval_datasets.split(',')


            for comb in itertools.product(evaluation_splits, evaluation_dirs, eval_dataset_names):
                split, evaluation_dir, dataset_name = comb
                model_dir = os.path.join(episode_output_dir, evaluation_dir)

                if len(evaluation_dir) > 0:
                    logging.info(f'Evaluate {evaluation_dir} on {dataset_name} {split}')
                else:
                    logging.info(f'Evaluate on {dataset_name} {split}')

                res = evaluate(
                    model=model, dataset_name=dataset_name, data_args=data_args, tokenizer=tokenizer, split=split,
                    seed=ep_idx, batch_size=training_args.per_device_eval_batch_size, gpu=args.gpu, mode=args.mode
                )

                evaluation_results[comb].append(res)


                if args.verbose_results:
                    print_results(res)


                with open(
                        os.path.join(model_dir, evaluation_output_filename + f'-{dataset_name}-{split}.json'), 'w'
                ) as f:
                    json.dump(res, f, indent=0)


    for comb, results in evaluation_results.items():
        split, evaluation_dir, dataset_name = comb

        print()
        logging.info(
            f'Average of {split} results over {len(results)} episodes ({dataset_name} {evaluation_dir}):'
        )
        res = get_avg_results(results)


        print_results(res)


        filename = evaluation_output_filename + f'-{dataset_name}-{split}'
        if len(evaluation_dir) > 0:
            filename += '-'
        filename += f'{evaluation_dir}.json'

        with open(os.path.join(output_dir, filename), 'w') as f:
            json.dump(res, f, indent=0)

    print()
    logging.info(f'Model weights and intermediate checkpoints saved in {output_dir}')


if __name__ == "__main__":
    main()
