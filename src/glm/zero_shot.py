from os.path import join
from new_eval_updated import defaultdict, token_recovery, INSTANCE_OF, fix
from collections import Counter, OrderedDict
import argparse

import json


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--path", type=str)

    return parser.parse_args()


def generate_mapping(dataset, runs):
    dataset_path = "/dataset/fd5061f6/liuxiao/data"
    dataset_path = join(dataset_path, dataset)
    path = join("/dataset/fd5061f6/liuxiao/deepstruct/glm/runs", runs)
    hyps, refs, raw = open(join(path, 'test.jsonl.hyps')).readlines(), \
                      open(join(path, 'test.jsonl.refs')).readlines(), \
                      open(join(dataset_path,'test.source')).readlines()

    ner_mapping = defaultdict(list)
    rel_mapping = defaultdict(list)

    if len(refs)==0:
        return 1

    assert (len(hyps) == len(refs)) or (len(hyps) == 0)
    assert (len(hyps) == len(raw)) or (len(hyps) == 0)

    scores = []; gt_tails = set()
    for hyp, ref, text in (zip(hyps, refs, raw) if len(hyps)>0 else zip(refs, refs, raw)):
        result = defaultdict(set); score = defaultdict(float)
        text = text.split('Sentence : ')[-1].lower()
        text_split = text.split(' ')
        hyp = hyp[2:-2]; ref = ref[2:-2]

        hyp_ents, ref_ents, hyp_rels, ref_rels = dict(), dict(), set(), set()
        for triple in hyp.split(' ) ( '):
            triple = [
                s.lower().strip('[ ').strip(' ]').strip()
                for s in triple.lower().strip().split(' ; ')
            ]
            if len(triple) != 3:
                continue
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
                gt_tails.add(triple[1])
            else:
                ref_ents[triple[0]] = triple[2]
                gt_tails.add(triple[2])


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




        rel_mapping[zs_type] = rel_mapping[zs_type][0][0]

    if len(ner_mapping) > 0:
        print(json.dumps(ner_mapping, indent=4))
        print("Save to:", join(dataset_path, 'zero_shot_ner_mapping.json'))
        json.dump(ner_mapping, open(join(dataset_path, 'zero_shot_ner_mapping.json'), 'w'), indent=4, ensure_ascii=False)
        json.dump(_ner_mapping, open(join(dataset_path, 'zero_shot_ner_mapping_stats.json'), 'w'), indent=4, ensure_ascii=False)
    if len(rel_mapping) > 0:
        print(json.dumps(_rel_mapping, indent=4))
        print("Save to:", join(dataset_path, 'zero_shot_rel_mapping.json'))
        json.dump(rel_mapping, open(join(dataset_path, 'zero_shot_rel_mapping.json'), 'w'), indent=4, ensure_ascii=False)
        json.dump(_rel_mapping, open(join(dataset_path, 'zero_shot_rel_mapping_stats.json'), 'w'), indent=4, ensure_ascii=False)


if __name__ == '__main__':
    args = get_args()
    generate_mapping(args.dataset, args.path)
