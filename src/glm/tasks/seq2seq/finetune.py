
"""Race."""
import torch
import mpu
import json
import functools
from tasks.eval_utils import accuracy_func_provider, f1_metric, f1_macro_metric
from finetune_glm import finetune
from pretrain_glm import get_batch
from collections import OrderedDict
from tasks.seq2seq.dataset import Seq2SeqDataset, BlankLMDataset, ExtractionDataset
from tasks.seq2seq.evaluate import rouge_metric, DecoderEvaluater, BlankLMEvaluater
from tasks.superglue.evaluate import squad_exact_match, squad_f1

global_tokenizer = None


def seq2seq_forward_step(data, model, args, timers, mems):
    """Forward step."""


    if timers is not None:
        timers('batch generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data, args)
    if timers is not None:
        timers('batch generator').stop()

    logits, *mems = model(tokens, position_ids, attention_mask, *mems)


    losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(), labels)
    if args.label_smoothing > 0.0:
        epsilon = args.label_smoothing
        smooth_loss = -torch.nn.functional.log_softmax(logits, dim=-1).mean(dim=-1)
        losses = (1 - epsilon) * losses + epsilon * smooth_loss
    loss_mask = loss_mask.reshape(-1)

    loss = torch.sum(losses.reshape(-1) * loss_mask) / loss_mask.sum()
    return loss, mems, 'bert'


def train_valid_datasets_provider(args, tokenizer):
    """Provide train and validation datasets."""
    if args.task.lower() == 'blank':
        train_dataset = BlankLMDataset(args, split='train', tokenizer=tokenizer)
        valid_dataset = None
    elif args.task.lower() == 'extraction':
        train_dataset = ExtractionDataset(args, split='train', tokenizer=tokenizer)
        valid_dataset = None
    else:
        train_dataset = Seq2SeqDataset(args, split='train', tokenizer=tokenizer)
        valid_dataset = None
    global global_tokenizer
    global_tokenizer = tokenizer
    return train_dataset, valid_dataset


def metrics_func_provider(args, tokenizer, is_test):
    """Provide metrics callback function."""

    def single_dataset_provider(split):
        if args.task.lower() == 'blank':
            return BlankLMDataset(args, split=split, tokenizer=tokenizer)
        elif args.task.lower() == 'extraction':
            return ExtractionDataset(args, split=split, tokenizer=tokenizer)
        else:
            return Seq2SeqDataset(args, split=split, tokenizer=tokenizer)

    if args.task.lower() in ['blank', 'extraction']:
        evaluater = BlankLMEvaluater(args, tokenizer)
        eval_func = evaluater.evaluate
        metric_dict = {}
    else:
        evaluater = DecoderEvaluater(args, tokenizer)
        eval_func = evaluater.evaluate
        if args.tokenizer_type == "BertWordPieceTokenizer":
            dataset = 'cnn_dm'
        elif args.task.lower() == 'gigaword':
            dataset = 'gigaword'
        else:
            dataset = 'cnn_dm_org'
        if args.task.lower() in ['squad', 'squad_v1']:
            metric_dict = {"EM": squad_exact_match, "F1": squad_f1}
        else:



            metric_dict = {"F1": f1_metric}


    def output_func(predictions, examples, output_file):
        print("[Output]")
        if args.task.lower() in ['squad', 'squad_v1']:
            with open(output_file, "w", encoding='utf-8') as output:
                res = {}
                for prediction, example in zip(predictions, examples):
                    idx = example.idx
                    if prediction.lower().replace(' ', '') == 'n/a':
                        prediction = ''
                    if idx not in res or res[idx] == '':
                        res[idx] = prediction
                json.dump(res, output)
            with open(output_file + ".refs", "w", encoding='utf-8') as output:
                for prediction, example in zip(predictions, examples):
                    res = {'id': example.idx, 'pred': prediction, 'gold': example.meta['answers']}
                    output.write(json.dumps(res) + '\n')
            return
        with open(output_file + ".hyps", "w", encoding='utf-8') as output:
            for prediction in predictions:
                output.write(prediction)
                output.write("\n")
        with open(output_file + ".refs", "w", encoding='utf-8') as output:
            for example in examples:
                output.write(example.meta["ref"])
                output.write("\n")
        if args.task.lower() == 'squad_generation':
            with open(output_file + ".source", "w", encoding='utf-8') as output:
                for example in examples:
                    output.write(example.text_a.replace("\n", " ") + " Answer: " + example.meta["answer"])
                    output.write("\n")

    return accuracy_func_provider(single_dataset_provider, metric_dict, args, is_test=is_test, eval_func=eval_func,
                                  output_func=output_func, only_rank0=False)


def main(args):
    if args.src_seq_length > args.max_position_embeddings:
        args.max_position_embeddings = args.src_seq_length
    if args.task.lower() in ['cnn_dm', 'cnn_dm_original', 'gigaword', 'blank', 'squad_generation', 'xsum',
                             'squad', 'squad_v1', 'extraction', 'PRETRAIN']:
        finetune(args, train_valid_datasets_provider, {}, end_of_epoch_callback_provider=metrics_func_provider,
                 forward_step=seq2seq_forward_step)
    else:
        raise NotImplementedError(args.task)
