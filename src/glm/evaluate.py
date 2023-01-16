from fuzzywuzzy import fuzz
from tqdm import tqdm
import argparse

import os
import re
import json

from pyheaven import *
from collections import defaultdict, Counter
import numpy as np

USE_TOKEN_RECOVER = False
TOKEN_RECOVER_STRATEGY = 2
THRESHOLD = 60
NOT_GIVEN = "not given"
INSTANCE_OF = "instance of"
REFER_TO = "refer to"
INTENT = "intent"
IS = "is"


def fix(string):
    return string.strip('[').strip(']') \
        .replace(' ', '') \
        .replace('.', '') \
        .replace(',', '') \
        .replace('(', '') \
        .replace(')', '').lower()


def find_best_sim(query, values):
    result = []
    for v in values:
        if v is None:
            v = ""
        result.append(fuzz.token_sort_ratio(query, v))
    most_similar = max(result) if len(result) > 0 else 0
    return most_similar, result.index(most_similar) if most_similar > 0 else -1


def generate_ngram(n, text_tokens):
    if n == 1:
        return text_tokens
    return [text_tokens[i: i + n] for i in range(len(text_tokens) - n)]


def find_best_ngram(query_tokens, text_tokens, threshold=60):
    ngrams = generate_ngram(len(query_tokens), text_tokens)
    ngram_strs = [' '.join(ngram) for ngram in ngrams]
    most_similar, index = find_best_sim(' '.join(query_tokens), ngram_strs)
    if most_similar >= threshold:
        return ngrams[index]
    return query_tokens


def token_recovery(token_source, text_split):
    if USE_TOKEN_RECOVER:
        tokens = token_source.split(' ')
        if TOKEN_RECOVER_STRATEGY <= 1:

            for i, token in enumerate(tokens.copy()):
                if token in text_split:
                    continue
                similarity, index = find_best_sim(token, text_split)
                candidate = text_split[index]
                if similarity >= THRESHOLD:
                    tokens[i] = candidate
            return ' '.join(tokens)
        elif TOKEN_RECOVER_STRATEGY <= 2:

            i, token = 0, tokens[0]
            if token in text_split and len(re.findall(re.escape(token), text.lower())) == 1:
                offset = text_split.index(token)
                return ' '.join(text_split[offset: offset + len(tokens)])
            else:
                for i, token in enumerate(tokens.copy()):
                    similarity, index = find_best_sim(token, text_split[:-(len(tokens) - i)])
                    candidate = text_split[index]
                    if similarity >= THRESHOLD:
                        for j in range(len(tokens) - i):
                            tokens[i + j] = text_split[index + j]
                        return ' '.join(tokens)
        elif TOKEN_RECOVER_STRATEGY <= 3:

            tokens = find_best_ngram(tokens, text_split, THRESHOLD)
            return ' '.join(tokens)
    else:
        return token_source


def _f1_metric(hyps, refs, raw, type_file=None, ner_mapping=None, rel_mapping=None, return_all=False):
    if len(refs) == 0:
        return 1

    assert (len(hyps) == len(refs)) or (len(hyps) == 0)

    if len(raw) != len(refs):
        raw = [None for _ in refs]
    scores = []
    allowed_types = set(list(ner_mapping.values())) if ner_mapping else set()
    allowed_relations = set(list(rel_mapping.values())) if rel_mapping else set()
    for (hyp, ref, text) in (zip(hyps, refs, raw) if len(hyps) > 0 else zip(refs, refs, raw)):
        result = defaultdict(set)
        score = defaultdict(float)
        if text is not None:
            if 'Sentence : ' in text:
                text = text.split('Sentence : ')[-1].lower()
            elif 'sentence : ' in text:
                text = text.split('sentence : ')[-1].lower()
            text_split = text.split(' ')
        else:
            text_split = None
        hyp = hyp.strip()[2:-2]
        ref = ref.strip()[2:-2]

        hyp_ents, ref_ents, hyp_rels, ref_rels = dict(), dict(), set(), set()
        for triple in hyp.split(' ) ( '):
            triple = [
                s.lower().strip('[ ').strip(' ]').strip()
                for s in triple.lower().strip().split(' ; ')
            ]
            if len(triple) != 3:
                continue
            if text_split is not None:
                triple[0] = token_recovery(triple[0], text_split)
                if triple[1] != INSTANCE_OF:
                    triple[2] = token_recovery(triple[2], text_split)
                if (triple[0] != INTENT) and (
                        (triple[0] not in text) or ((triple[2] not in text) and (triple[1] != INSTANCE_OF))):
                    continue
            triple = (fix(triple[0]), triple[1], fix(triple[2]))
            if triple[1] != INSTANCE_OF and triple[1] != REFER_TO:
                if rel_mapping and triple[1] not in rel_mapping and triple[1] not in allowed_relations:
                    continue
                hyp_rels.add(tuple(triple))
            else:
                if ner_mapping and triple[2] not in ner_mapping and triple[2] not in allowed_types:
                    continue
                hyp_ents[triple[0]] = triple[2]
        for triple in ref.split(' ) ( '):
            triple = [
                s.lower().strip('[ ').strip(' ]').strip()
                for s in triple.lower().strip().split(' ; ')
            ]
            if len(triple) != 3:
                continue
            triple = (fix(triple[0]), triple[1], fix(triple[2]))
            if triple[1] != INSTANCE_OF and triple[1] != REFER_TO:
                ref_rels.add(tuple(triple))
            else:
                ref_ents[triple[0]] = triple[2]

        ent_metrics = {
            'correct': lambda x: x[0] == x[1],
            'ntype': lambda x: x[0][0] == x[1][0],
            'close': lambda x: (x[0][1] == x[1][1]) and ((x[0][0] in x[1][0]) or (x[1][0] in x[0][0])),
            'ntype_close': lambda x: (x[0][0] in x[1][0]) or (x[1][0] in x[0][0]),
        }


        result['ent_gt'] = set(ref_ents)
        result['ent_pd'] = set(hyp_ents)


        for key, metric in ent_metrics.items():
            score['ent_' + key + '_rec'], score['ent_' + key + '_pre'], score['ent_' + key + '_f1'] = 0, 0, 0
        for ent, ent_t in hyp_ents.items():
            for key, metric in ent_metrics.items():
                for ref, ref_t in ref_ents.items():

                    matched = metric(((ent, ent_t), (ref, ref_t))) or \
                              (ner_mapping and metric(((ent, ner_mapping.get(ent_t)), (ref, ref_t))))
                    if matched:
                        result['ent_' + key].add(ent);
                        break
                rec = len(result['ent_' + key]) / len(result['ent_gt']) if len(result['ent_gt']) > 0 else 0
                pre = len(result['ent_' + key]) / len(result['ent_pd']) if len(result['ent_pd']) > 0 else 0
                f1 = 2 * rec * pre / (rec + pre) if (rec + pre) > 0 else 0
                score['ent_' + key + '_rec'], score['ent_' + key + '_pre'], score['ent_' + key + '_f1'] = rec, pre, f1
        score['ent_gt'] = len(result['ent_gt'])
        score['ent_pd'] = len(result['ent_pd'])

        zero_shot = lambda x: rel_mapping and (rel_mapping.get(x[0][1]) == x[1][1])

        rel_metrics = {
            'correct': lambda x: (x[0][1] == x[1][1] or zero_shot(x)) and (x[0][0] in result['ent_correct']) and (
                    x[0][2] in result['ent_correct']) and (x[0][0] == x[1][0]) and (x[0][2] == x[1][2]),
            'ntype': lambda x: (x[0][1] == x[1][1] or zero_shot(x)) and (x[0][0] in result['ent_ntype']) and (
                    x[0][2] in result['ent_ntype']) and (x[0][0] == x[1][0]) and (x[0][2] == x[1][2]),
            'close': lambda x: (x[0][1] == x[1][1] or zero_shot(x)) and (x[0][0] in result['ent_close']) and (
                    x[0][2] in result['ent_close']),
            'ntype_close': lambda x: (x[0][1] == x[1][1] or zero_shot(x)) and (x[0][0] in result['ntype_close']) and (
                    x[0][2] in result['ntype_close']),
            'ent_correct': lambda x: (x[0][0] in result['ent_correct']) and (x[0][2] in result['ent_correct']) and (
                    x[0][0] == x[1][0]) and (x[0][2] == x[1][2]),
            'ent_ntype': lambda x: (x[0][0] in result['ent_ntype']) and (x[0][2] in result['ent_ntype']) and (
                    x[0][0] == x[1][0]) and (x[0][2] == x[1][2]),
            'ent_close': lambda x: (x[0][0] in result['ent_close']) and (x[0][2] in result['ent_close']),
            'ent_ntype_close': lambda x: (x[0][0] in result['ntype_close']) and (x[0][2] in result['ntype_close']),
            'arg_correct': lambda x: (x[0] == x[1]) and (x[0][0] in result['ent_correct']) and (
                    x[0][0] == x[1][0]) and (x[0][2] == x[1][2]),
            'arg_ntype': lambda x: (x[0] == x[1]) and (x[0][0] == x[1][0]) and (x[0][2] == x[1][2]),
            'zero_shot': lambda x: (x[0][1] == x[1][1] or zero_shot(x)) and (x[0][0] == x[1][0]) and (
                    x[0][2] == x[1][2]),
        }








        result['rel_gt'] = set(ref_rels)
        result['rel_pd'] = set(hyp_rels)

        for key, metric in rel_metrics.items():
            score['rel_' + key + '_rec'], score['rel_' + key + '_pre'], score['rel_' + key + '_f1'] = 0, 0, 0
        for rel in hyp_rels:
            t = 1
            for key, metric in rel_metrics.items():
                for ref in ref_rels:
                    matched = metric((rel, ref))
                    if matched:
                        result['rel_' + key].add(rel);
                        break
                rec = len(result['rel_' + key]) / len(result['rel_gt']) if len(result['rel_gt']) > 0 else 0
                pre = len(result['rel_' + key]) / len(result['rel_pd']) if len(result['rel_pd']) > 0 else 0
                f1 = 2 * rec * pre / (rec + pre) if (rec + pre) > 0 else 0
                score['rel_' + key + '_rec'], score['rel_' + key + '_pre'], score['rel_' + key + '_f1'] = rec, pre, f1
        score['rel_gt'] = len(result['rel_gt']);
        score['rel_pd'] = len(result['rel_pd'])

        scores.append(score)

    if len(scores) > 0:
        averaged_scores = {
            key: (
                (np.sum([score[key] * score[key[:4] + 'pd'] for score in scores]) / np.sum(
                    [score[key[:4] + 'pd'] for score in scores])
                 if np.sum([score[key[:4] + 'pd'] for score in scores]) > 0 else 0)
                if key.endswith('_pre') else
                (np.sum([score[key] * score[key[:4] + 'gt'] for score in scores]) / np.sum(
                    [score[key[:4] + 'gt'] for score in scores])
                 if np.sum([score[key[:4] + 'gt'] for score in scores]) > 0 else 0)
            ) for key in scores[0] if (key.endswith('_pre') or key.endswith('_rec'))
        }
        averaged_scores.update({
            key: (
                2 * averaged_scores[key[:-3] + "_pre"] * averaged_scores[key[:-3] + "_rec"] /
                (averaged_scores[key[:-3] + "_pre"] + averaged_scores[key[:-3] + "_rec"])
                if (averaged_scores[key[:-3] + "_pre"] + averaged_scores[key[:-3] + "_rec"]) > 0 else 0
            ) for key in scores[0] if key.endswith('_f1')
        })
    else:
        averaged_scores = {};
        return 0




    return averaged_scores['ent_correct_f1'] if not return_all else averaged_scores


def f1_metric(predictions, labels, examples, type_file=None):
    hyps = predictions
    refs = [example.meta["ref"] for example in examples]
    raw = [example.text_a for example in examples]
    return _f1_metric(hyps, refs, raw, type_file=None)


def read_dst_triples(hyps, refs, raw, type_file=None, ner_mapping=None, rel_mapping=None, return_all=False):
    if len(refs) == 0:
        return 1

    assert (len(hyps) == len(refs)) or (len(hyps) == 0)

    if len(raw) != len(refs):
        raw = [None for _ in refs]
    scores = []
    allowed_types = set(list(ner_mapping.values())) if ner_mapping else set()
    allowed_relations = set(list(rel_mapping.values())) if rel_mapping else set()
    for (hyp, ref, text) in (zip(hyps, refs, raw) if len(hyps) > 0 else zip(refs, refs, raw)):
        result = defaultdict(set)
        score = defaultdict(float)
        if text is not None:
            if 'sentence:' in text:
                text = text.split('sentence: ')[-1].lower()
            text_split = text.split(' ')
        else:
            text_split = None
        hyp = hyp.strip()[2:-2]
        ref = ref.strip()[2:-2]

        hyp_ents, ref_ents, hyp_rels, ref_rels = dict(), dict(), set(), set()
        for triple in hyp.split(' ) ( '):
            triple = [
                s.lower().strip('[ ').strip(' ]').strip()
                for s in triple.lower().strip().split(' ; ')
            ]
            if len(triple) != 3:
                continue
            if text_split is not None:
                triple[0] = token_recovery(triple[0], text_split)
                triple[2] = token_recovery(triple[2], text_split)
            triple = (fix(triple[0]), triple[1], fix(triple[2]))
            if rel_mapping and triple[1] not in rel_mapping and triple[1] not in allowed_relations:
                continue
            hyp_rels.add(tuple(triple))
        for triple in ref.split(' ) ( '):
            triple = [
                s.lower().strip('[ ').strip(' ]').strip()
                for s in triple.lower().strip().split(' ; ')
            ]
            if len(triple) != 3:
                continue
            triple = (fix(triple[0]), triple[1], fix(triple[2]))
            ref_rels.add(tuple(triple))

        ent_metrics = {
            'correct': lambda x: x[0] == x[1],
            'ntype': lambda x: x[0][0] == x[1][0],
            'close': lambda x: (x[0][1] == x[1][1]) and ((x[0][0] in x[1][0]) or (x[1][0] in x[0][0])),
            'ntype_close': lambda x: (x[0][0] in x[1][0]) or (x[1][0] in x[0][0]),
        }


        result['ent_gt'] = set(ref_ents)
        result['ent_pd'] = set(hyp_ents)


        for key, metric in ent_metrics.items():
            score['ent_' + key + '_rec'], score['ent_' + key + '_pre'], score['ent_' + key + '_f1'] = 0, 0, 0
        for ent, ent_t in hyp_ents.items():
            for key, metric in ent_metrics.items():
                for ref, ref_t in ref_ents.items():

                    matched = metric(((ent, ent_t), (ref, ref_t))) or \
                              (ner_mapping and metric(((ent, ner_mapping.get(ent_t)), (ref, ref_t))))
                    if matched:
                        result['ent_' + key].add(ent);
                        break
                rec = len(result['ent_' + key]) / len(result['ent_gt']) if len(result['ent_gt']) > 0 else 0
                pre = len(result['ent_' + key]) / len(result['ent_pd']) if len(result['ent_pd']) > 0 else 0
                f1 = 2 * rec * pre / (rec + pre) if (rec + pre) > 0 else 0
                score['ent_' + key + '_rec'], score['ent_' + key + '_pre'], score['ent_' + key + '_f1'] = rec, pre, f1
        score['ent_gt'] = len(result['ent_gt'])
        score['ent_pd'] = len(result['ent_pd'])

        zero_shot = lambda x: rel_mapping and (rel_mapping.get(x[0][1]) == x[1][1])

        rel_metrics = {
            'correct': lambda x: (x[0][1] == x[1][1] or zero_shot(x)) and (x[0][0] in result['ent_correct']) and (
                    x[0][2] in result['ent_correct']) and (x[0][0] == x[1][0]) and (x[0][2] == x[1][2]),
            'ntype': lambda x: (x[0][1] == x[1][1] or zero_shot(x)) and (x[0][0] in result['ent_ntype']) and (
                    x[0][2] in result['ent_ntype']) and (x[0][0] == x[1][0]) and (x[0][2] == x[1][2]),
            'close': lambda x: (x[0][1] == x[1][1] or zero_shot(x)) and (x[0][0] in result['ent_close']) and (
                    x[0][2] in result['ent_close']),
            'ntype_close': lambda x: (x[0][1] == x[1][1] or zero_shot(x)) and (x[0][0] in result['ntype_close']) and (
                    x[0][2] in result['ntype_close']),
            'ent_correct': lambda x: (x[0][0] in result['ent_correct']) and (x[0][2] in result['ent_correct']) and (
                    x[0][0] == x[1][0]) and (x[0][2] == x[1][2]),
            'ent_ntype': lambda x: (x[0][0] in result['ent_ntype']) and (x[0][2] in result['ent_ntype']) and (
                    x[0][0] == x[1][0]) and (x[0][2] == x[1][2]),
            'ent_close': lambda x: (x[0][0] in result['ent_close']) and (x[0][2] in result['ent_close']),
            'ent_ntype_close': lambda x: (x[0][0] in result['ntype_close']) and (x[0][2] in result['ntype_close']),
            'arg_correct': lambda x: (x[0] == x[1]) and (x[0][0] in result['ent_correct']) and (
                    x[0][0] == x[1][0]) and (x[0][2] == x[1][2]),
            'arg_ntype': lambda x: (x[0] == x[1]) and (x[0][0] == x[1][0]) and (x[0][2] == x[1][2]),
            'zero_shot': lambda x: (x[0][1] == x[1][1] or zero_shot(x)) and (x[0][0] == x[1][0]) and (
                    x[0][2] == x[1][2]),
        }








        result['rel_gt'] = set(ref_rels)
        result['rel_pd'] = set(hyp_rels)

        for key, metric in rel_metrics.items():
            score['rel_' + key + '_rec'], score['rel_' + key + '_pre'], score['rel_' + key + '_f1'] = 0, 0, 0
        for rel in hyp_rels:
            t = 1
            for key, metric in rel_metrics.items():
                for ref in ref_rels:
                    matched = metric((rel, ref))
                    if matched:
                        result['rel_' + key].add(rel);
                        break
                rec = len(result['rel_' + key]) / len(result['rel_gt']) if len(result['rel_gt']) > 0 else 0
                pre = len(result['rel_' + key]) / len(result['rel_pd']) if len(result['rel_pd']) > 0 else 0
                f1 = 2 * rec * pre / (rec + pre) if (rec + pre) > 0 else 0
                score['rel_' + key + '_rec'], score['rel_' + key + '_pre'], score['rel_' + key + '_f1'] = rec, pre, f1
        score['rel_gt'] = len(result['rel_gt']);
        score['rel_pd'] = len(result['rel_pd'])

        score['jointrel_gt'] = len(result['rel_gt']) > 0
        score['jointrel_pd'] = len(result['rel_pd']) > 0
        score['jointrel_correct'] = len(result['rel_arg_ntype']) == len(result['rel_gt'])
        scores.append(score)

    if len(scores) > 0:
        averaged_scores = {
            key: (
                (np.sum([score[key] * score[key[:4] + 'pd'] for score in scores]) / np.sum(
                    [score[key[:4] + 'pd'] for score in scores])
                 if np.sum([score[key[:4] + 'pd'] for score in scores]) > 0 else 0)
                if key.endswith('_pre') else
                (np.sum([score[key] * score[key[:4] + 'gt'] for score in scores]) / np.sum(
                    [score[key[:4] + 'gt'] for score in scores])
                 if np.sum([score[key[:4] + 'gt'] for score in scores]) > 0 else 0)
            ) for key in scores[0] if (key.endswith('_pre') or key.endswith('_rec'))
        }
        averaged_scores.update({
            key: (
                2 * averaged_scores[key[:-3] + "_pre"] * averaged_scores[key[:-3] + "_rec"] /
                (averaged_scores[key[:-3] + "_pre"] + averaged_scores[key[:-3] + "_rec"])
                if (averaged_scores[key[:-3] + "_pre"] + averaged_scores[key[:-3] + "_rec"]) > 0 else 0
            ) for key in scores[0] if key.endswith('_f1')
        })
    else:
        averaged_scores = {};
        return 0

    pre = sum([score['jointrel_correct'] for score in scores]) / sum([score['jointrel_pd'] for score in scores])
    rec = sum([score['jointrel_correct'] for score in scores]) / sum([score['jointrel_gt'] for score in scores])
    f1 = 2 * rec * pre / (rec + pre) if (rec + pre) > 0 else 0
    averaged_scores['jointrel_pre'], averaged_scores['jointrel_rec'], averaged_scores['jointrel_f1'] = pre, rec, f1
    for key, value in sorted(averaged_scores.items()):
        print(key, ":", value)

    return averaged_scores['ent_correct_f1'] if not return_all else averaged_scores


def read_oie_triples(hyps, refs, raw, type_file=None, ner_mapping=None, rel_mapping=None):
    if len(refs) == 0:
        return 1

    assert (len(hyps) == len(refs)) or (len(hyps) == 0)

    if len(raw) != len(refs):
        raw = [None for _ in refs]
    results = []
    allowed_types = set(list(ner_mapping.values())) if ner_mapping else set()
    allowed_relations = set(list(rel_mapping.values())) if rel_mapping else set()
    for (hyp, ref, text) in (zip(hyps, refs, raw) if len(hyps) > 0 else zip(refs, refs, raw)):
        result = defaultdict(set)
        score = defaultdict(float)
        if text is not None:
            if 'Sentence : ' in text:
                text = text.split('Sentence : ')[-1].lower()
            elif 'sentence : ' in text:
                text = text.split('sentence : ')[-1].lower()
            text_split = text.split(' ')
        else:
            text_split = None
        hyp = hyp[2:-2]
        ref = ref[2:-2]

        triples = set()
        for triple in hyp.split(' ) ( '):
            triple = [
                s.lower().strip('[ ').strip(' ]').strip()
                for s in triple.lower().strip().split(' ; ')
            ]
            if len(triple) != 3:
                continue
            if triple[1] == INSTANCE_OF:
                continue
            if text_split is not None:
                triple[0] = token_recovery(triple[0], text_split)
                triple[2] = token_recovery(triple[2], text_split)
                if (triple[0] not in text) or (triple[2] not in text):
                    continue
            triple = (triple[0], triple[1], triple[2])
            triples.add(triple)

        results.append(list(triples))

    return results


def read_zero_shot_triples(hyps, refs, type_file=None, dataset_path=None):
    if len(refs) == 0:
        return 1

    assert (len(hyps) == len(refs)) or (len(hyps) == 0)

    if len(raw) != len(refs):
        raw = [None for _ in refs]
    ner_mapping = defaultdict(list)
    rel_mapping = defaultdict(list)
    for (hyp, ref, text) in (zip(hyps, refs, raw) if len(hyps) > 0 else zip(refs, refs, raw)):
        result = defaultdict(set)
        score = defaultdict(float)
        if text is not None:
            if 'Sentence : ' in text:
                text = text.split('Sentence : ')[-1].lower()
            elif 'sentence : ' in text:
                text = text.split('sentence : ')[-1].lower()
            text_split = text.split(' ')
        else:
            text_split = None
        hyp = hyp[2:-2]
        ref = ref[2:-2]

        hyp_ents, ref_ents, hyp_rels, ref_rels = dict(), dict(), set(), set()
        for triple in hyp.split(' ) ( '):
            triple = [
                s.lower().strip('[ ').strip(' ]').strip()
                for s in triple.lower().strip().split(' ; ')
            ]
            if len(triple) != 3:
                continue
            if text_split is not None:
                triple[0] = token_recovery(triple[0], text_split)
                if triple[1] != INSTANCE_OF:
                    triple[2] = token_recovery(triple[2], text_split)
                if (triple[0] not in text) or (
                        (triple[2] not in text) and (triple[1] != INSTANCE_OF)
                ):
                    continue
            triple = (fix(triple[0]), triple[1], fix(triple[2]))
            if triple[1] != INSTANCE_OF:
                hyp_rels.add(tuple(triple))
            else:
                hyp_ents[triple[0]] = triple[2]
        for triple in ref.split(' ) ( '):
            triple = [
                s.lower().strip('[ ').strip(' ]').strip()
                for s in triple.lower().strip().split(' ; ')
            ]
            if len(triple) != 3:
                continue
            triple = (fix(triple[0]), triple[1], fix(triple[2]))
            if triple[1] != INSTANCE_OF:
                ref_rels.add(tuple(triple))
            else:
                ref_ents[triple[0]] = triple[2]

        for ent_surface, ner_type in ref_ents.items():
            if ent_surface in hyp_ents:
                ner_mapping[hyp_ents[ent_surface]].append(ner_type)
        for hyp_rel in hyp_rels:
            for ref_rel in ref_rels:
                if (hyp_rel[0] == ref_rel[0] or ref_rel[0] in hyp_rel[0]) and \
                        (hyp_rel[2] == ref_rel[2] or ref_rel[2] in hyp_rel[2]):
                    rel_mapping[hyp_rel[1]].append(ref_rel[1])

    _ner_mapping, _rel_mapping = ner_mapping.copy(), rel_mapping.copy()
    for zs_type in list(ner_mapping.keys()):
        ner_mapping[zs_type] = list(Counter(ner_mapping[zs_type]).items())
        ner_mapping[zs_type] = ner_mapping[zs_type][0][0]
    for zs_type in list(rel_mapping.keys()):
        rel_mapping[zs_type] = list(Counter(rel_mapping[zs_type]).items())
        rel_mapping[zs_type] = rel_mapping[zs_type][0][0]

    if len(ner_mapping) > 0:
        print("Save to:", pjoin(dataset_path, 'zero_shot_ner_mapping.json'))
        json.dump(ner_mapping, open(pjoin(dataset_path, 'zero_shot_ner_mapping.json'), 'w'), indent=4,
                  ensure_ascii=False)
        json.dump(_ner_mapping, open(pjoin(dataset_path, 'zero_shot_ner_mapping_stats.json'), 'w'), indent=4,
                  ensure_ascii=False)
    if len(rel_mapping) > 0:
        print("Save to:", pjoin(dataset_path, 'zero_shot_rel_mapping.json'))
        json.dump(rel_mapping, open(pjoin(dataset_path, 'zero_shot_rel_mapping.json'), 'w'), indent=4,
                  ensure_ascii=False)
        json.dump(_rel_mapping, open(pjoin(dataset_path, 'zero_shot_rel_mapping_stats.json'), 'w'), indent=4,
                  ensure_ascii=False)


def read_rc_triples(hyps, refs, raw, type_file=None, ner_mapping=None, rel_mapping=None):
    if len(refs) == 0:
        return 1

    assert (len(hyps) == len(refs)) or (len(hyps) == 0)

    if len(raw) != len(refs):
        raw = [None for _ in refs]
    results = []
    allowed_types = set(list(ner_mapping.values())) if ner_mapping else set()
    allowed_relations = set(list(rel_mapping.values())) if rel_mapping else set()
    for (hyp, ref, text) in (zip(hyps, refs, raw) if len(hyps) > 0 else zip(refs, refs, raw)):
        result = defaultdict(set)
        score = defaultdict(float)
        if text is not None:
            if 'Sentence : ' in text:
                text = text.split('Sentence : ')[-1].lower()
            elif 'sentence : ' in text:
                text = text.split('sentence : ')[-1].lower()
            text_split = text.split(' ')
        else:
            text_split = None
        hyp = hyp[2:-2]
        ref = ref[2:-2]

        top1_hyp_triple, top1_ref_triple = None, None
        for triple in hyp.split(' ) ( '):
            triple = [
                s.lower().strip('[ ').strip(' ]').strip()
                for s in triple.lower().strip().split(' ; ')
            ]
            if len(triple) != 3:
                continue
            if triple[1] == INSTANCE_OF:
                continue
            if text_split is not None:
                triple[0] = token_recovery(triple[0], text_split)
                triple[2] = token_recovery(triple[2], text_split)
                if (triple[0] not in text) or (triple[2] not in text):
                    continue
            top1_hyp_triple = (fix(triple[0]), triple[1], fix(triple[2]))
            break

        for triple in ref.split(' ) ( '):
            triple = [
                s.lower().strip('[ ').strip(' ]').strip()
                for s in triple.lower().strip().split(' ; ')
            ]
            if len(triple) != 3:
                continue
            if triple[1] == INSTANCE_OF:
                continue
            if text_split is not None:
                triple[0] = token_recovery(triple[0], text_split)
                triple[2] = token_recovery(triple[2], text_split)
                if (triple[0] not in text) or (triple[2] not in text):
                    continue
            top1_ref_triple = (triple[0], triple[1], triple[2])
            break
        NO_RELATION = (None, "no relation", None)
        results.append((NO_RELATION if top1_hyp_triple is None else top1_hyp_triple,
                        NO_RELATION if top1_ref_triple is None else top1_ref_triple))

    r_gt, r_pd, r_correct, nor_gt, nor_pd, nor_correct = 0, 0, 0, 0, 0, 0
    for hyp, ref in results:
        nor_gt += 1;
        nor_pd += 1
        if ref[1] == hyp[1]:
            nor_correct += 1
        if ref[1] == "no relation" and hyp[1] == "no relation":
            pass
        elif ref[1] == hyp[1]:
            r_gt += 1
            r_pd += 1
            r_correct += 1
        elif hyp[1] == "no relation":
            r_gt += 1
        elif ref[1] == "no relation":
            r_pd += 1
    r_pre = r_correct / r_pd if r_pd > 0 else 0
    r_rec = r_correct / r_gt if r_gt > 0 else 0
    r_f1 = 2 * r_pre * r_rec / (r_pre + r_rec) if (r_pre + r_rec) > 0 else 0
    nor_pre = nor_correct / nor_pd if nor_pd > 0 else 0
    nor_rec = nor_correct / nor_gt if nor_gt > 0 else 0
    nor_f1 = 2 * nor_pre * nor_rec / (nor_pre + nor_rec) if (nor_pre + nor_rec) > 0 else 0
    print(f"""
r_pre:{r_pre},
r_rec:{r_rec},
r_f1:{r_f1},
nor_pre:{nor_pre},
nor_rec:{nor_rec},
nor_f1:{nor_f1},
""")
    return results


def read_fp_triples(hyps, refs, raw, type_file=None, ner_mapping=None, rel_mapping=None):
    if len(refs) == 0:
        return 1

    assert (len(hyps) == len(refs)) or (len(hyps) == 0)

    if len(raw) != len(refs):
        raw = [None for _ in refs]
    results = []
    allowed_types = set(list(ner_mapping.values())) if ner_mapping else set()
    allowed_relations = set(list(rel_mapping.values())) if rel_mapping else set()
    for (hyp, ref, text) in (zip(hyps, refs, raw) if len(hyps) > 0 else zip(refs, refs, raw)):
        result = defaultdict(set)
        score = defaultdict(float)
        if text is not None:
            if 'Sentence : ' in text:
                text = text.split('Sentence : ')[-1].lower()
            elif 'sentence : ' in text:
                text = text.split('sentence : ')[-1].lower()
            text_split = text.split(' ')
        else:
            text_split = None
        hyp = hyp[2:-2]
        ref = ref[2:-2]

        top1_hyp_triple, top1_ref_triple = None, None
        for triple in ref.split(' ) ( '):
            triple = [
                s.lower().strip('[ ').strip(' ]').strip()
                for s in triple.lower().strip().split(' ; ')
            ]
            if len(triple) != 3:
                continue
            if triple[1] == INSTANCE_OF:
                continue
            if text_split is not None:
                triple[0] = token_recovery(triple[0], text_split)
                triple[2] = token_recovery(triple[2], text_split)
                if (triple[0] not in text) or (triple[2] not in text):
                    continue
            top1_ref_triple = (fix(triple[0]), triple[1], fix(triple[2]))
            break

        if top1_ref_triple is None:
            continue

        for triple in hyp.split(' ) ( '):
            triple = [
                s.lower().strip('[ ').strip(' ]').strip()
                for s in triple.lower().strip().split(' ; ')
            ]
            if len(triple) != 3:
                continue
            if triple[1] == INSTANCE_OF:
                continue
            if text_split is not None:
                triple[0] = token_recovery(triple[0], text_split)
                triple[2] = token_recovery(triple[2], text_split)
                if (triple[0] not in text) or (triple[2] not in text):
                    continue
            if fix(triple[0]) == top1_ref_triple[0] and triple[1] == top1_ref_triple[1]:
                top1_hyp_triple = (fix(triple[0]), triple[1], fix(triple[2]))
            break
        if top1_hyp_triple is not None:
            results.append((top1_hyp_triple, top1_ref_triple))
        else:
            results.append((None, top1_ref_triple))

    gt, pd, correct = 0, 0, 0
    for hyp, ref in results:
        gt += 1;
        pd += hyp is not None;
        correct += (hyp is not None) and (ref[2] == hyp[2])
    pre = correct / pd if pd > 0 else 0
    rec = correct / gt if gt > 0 else 0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0
    print(f"""
pre:{pre},
rec:{rec},
f1:{f1},
""")
    return results


def main(directory):
    hyps, refs, raw = open(directory + 'test.jsonl.hyps').readlines() if os.path.exists(
        directory + 'test.jsonl.hyps') else [], \
                      open(directory + 'test.jsonl.refs').readlines(), open(directory + 'test.source').readlines()
    return _f1_metric(hyps, refs, raw, type_file=None)


def grid_search():
    global THRESHOLD
    top_f1, top_threshold = 0, 0
    for THRESHOLD in tqdm(range(50, 80)):
        f1 = main()
        if f1 > top_f1:
            top_f1 = f1
            top_threshold = THRESHOLD
    print("Top f1:", top_f1, "Top threshold:", top_threshold)


TASK_METRIC_MAPPING = {

    'conll04': [{'key': 'ent_correct_pre', 'tkey': 'Entity Precision'},
                {'key': 'ent_correct_rec', 'tkey': 'Entity Recall'},
                {'key': 'ent_correct_f1', 'tkey': 'Entity F1'}],
    'conll04_re': [{'key': 'rel_arg_ntype_pre', 'tkey': 'Relation Precision'},
                   {'key': 'rel_arg_ntype_rec', 'tkey': 'Relation Recall'},
                   {'key': 'rel_arg_ntype_f1', 'tkey': 'Relation F1'}],
    'ade0': [{'key': 'ent_correct_pre', 'tkey': 'Entity Precision'},
             {'key': 'ent_correct_rec', 'tkey': 'Entity Recall'},
             {'key': 'ent_correct_f1', 'tkey': 'Entity F1'}],
    'ade_re0': [{'key': 'rel_arg_ntype_pre', 'tkey': 'Relation Precision'},
                {'key': 'rel_arg_ntype_rec', 'tkey': 'Relation Recall'},
                {'key': 'rel_arg_ntype_f1', 'tkey': 'Relation F1'}],
    'nyt': [{'key': 'ent_correct_pre', 'tkey': 'Entity Precision'},
            {'key': 'ent_correct_rec', 'tkey': 'Entity Recall'},
            {'key': 'ent_correct_f1', 'tkey': 'Entity F1'}],
    'nyt_re': [{'key': 'rel_arg_ntype_pre', 'tkey': 'Relation Precision'},
               {'key': 'rel_arg_ntype_rec', 'tkey': 'Relation Recall'},
               {'key': 'rel_arg_ntype_f1', 'tkey': 'Relation F1'}],
    'ace2005_joint_er': [{'key': 'ent_correct_pre', 'tkey': 'Entity Precision'},
                         {'key': 'ent_correct_rec', 'tkey': 'Entity Recall'},
                         {'key': 'ent_correct_f1', 'tkey': 'Entity F1'}],
    'ace2005_joint_er_re': [{'key': 'rel_arg_ntype_pre', 'tkey': 'Relation Precision'},
                            {'key': 'rel_arg_ntype_rec', 'tkey': 'Relation Recall'},
                            {'key': 'rel_arg_ntype_f1', 'tkey': 'Relation F1'}],

    'conll05_srl_wsj': [{'key': 'ent_correct_pre', 'tkey': 'Precision'},
                        {'key': 'ent_correct_rec', 'tkey': 'Recall'},
                        {'key': 'ent_correct_f1', 'tkey': 'F1'}],
    'conll05_srl_brown': [{'key': 'ent_correct_pre', 'tkey': 'Precision'},
                          {'key': 'ent_correct_rec', 'tkey': 'Recall'},
                          {'key': 'ent_correct_f1', 'tkey': 'F1'}],
    'conll12_srl': [{'key': 'ent_correct_pre', 'tkey': 'Precision'},
                    {'key': 'ent_correct_rec', 'tkey': 'Recall'},
                    {'key': 'ent_correct_f1', 'tkey': 'F1'}],

    'multi_woz': [{'key': 'jointrel_pre', 'tkey': 'Precision'},
                  {'key': 'jointrel_rec', 'tkey': 'Recall'},
                  {'key': 'jointrel_f1', 'tkey': 'F1'}],

    'atis': [{'key': 'rel_arg_ntype_rec', 'tkey': 'Precision'},
             {'key': 'rel_arg_ntype_rec', 'tkey': 'Recall'},
             {'key': 'rel_arg_ntype_rec', 'tkey': 'F1'}],
    'snips': [{'key': 'rel_arg_ntype_rec', 'tkey': 'Precision'},
              {'key': 'rel_arg_ntype_rec', 'tkey': 'Recall'},
              {'key': 'rel_arg_ntype_rec', 'tkey': 'F1'}],

    'ace2005event_trigger': [{'key': 'ent_ntype_f1', 'tkey': 'Trigger Id F1'},
                             {'key': 'ent_correct_f1', 'tkey': 'Trigger Cl F1'}],
    'ace2005event_argument': [{'key': 'ent_ntype_f1', 'tkey': 'Argument Id F1'},
                              {'key': 'ent_correct_f1', 'tkey': 'Argument Cl F1'}]
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', dest='task', type=str)
    parser.add_argument('--zero-shot', action='store_true')
    args = parser.parse_args()

    dataset_path = f"../data/{args.task}/"
    with open(dataset_path + "test.jsonl.hyps", "r") as f:
        hyps = [s.strip() for s in f.readlines()]
    with open(dataset_path + "test.target", "r") as f:
        refs = [s.strip() for s in f.readlines()]
    with open(dataset_path + "test.source", "r") as f:
        raw = [s.strip() for s in f.readlines()]
    if args.zero_shot:
        read_zero_shot_triples(hyps, refs, raw, dataset_path=dataset_path)
        ner_mapping_file = pjoin(dataset_path, 'zero_shot_ner_mapping.json')
        ner_mapping = LoadJson(ner_mapping_file) if ExistFile(ner_mapping_file) else None
        rel_mapping_file = pjoin(dataset_path, 'zero_shot_rel_mapping.json')
        rel_mapping = LoadJson(rel_mapping_file) if ExistFile(rel_mapping_file) else None
        print(_f1_metric(hyps, refs, raw, ner_mapping=ner_mapping, rel_mapping=rel_mapping))
    else:
        metrics = None
        if args.task in ['oie_oie2016', 'oie_nyt', 'oie_web', 'oie_penn']:
            metrics = read_oie_triples(hyps, refs, raw)
        elif args.task in ['tacred']:
            metrics = read_rc_triples(hyps, refs, raw)
        elif args.task in ['multi_woz']:
            metrics = read_dst_triples(hyps, refs, raw, return_all=True)
        elif args.task in ['trex', 'googlere']:
            metrics = read_fp_triples(hyps, refs, raw)
        else:
            metrics = _f1_metric(hyps, refs, raw, return_all=True)


        if args.task in ['ace2005event_argument']:
            exit(0)
        header = f"########## {args.task} Evaluation ##########"
        print(header)
        for metric in TASK_METRIC_MAPPING[args.task]:
            print(f"{metric['tkey']:20}: {metrics[metric['key']]}")
        print("#" * len(header))
