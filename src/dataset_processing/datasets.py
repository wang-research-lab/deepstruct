# Adapted from https://github.com/amazon-science/tanl
import bisect
import copy
import os
import logging
import json
from itertools import islice
from collections import Counter, defaultdict
import numpy as np
import random
import networkx as nx
from typing import Dict, List, Tuple, Set
import torch
from transformers import PreTrainedTokenizer

from arguments import DataTrainingArguments
from input_example import InputFeatures, EntityType, RelationType, Entity, Relation, Intent, InputExample, CorefDocument
from base_dataset import BaseDataset
from utils import get_precision_recall_f1
from coreference_metrics import CorefAllMetrics
from input_formats import INPUT_FORMATS
from output_formats import OUTPUT_FORMATS

def FlattenList(L:List):
    F = lambda x:[e for i in x for e in F(i)] if isinstance(x,list) else [x]; return F(L)

TRIGGER_HYPS_FILE = "../../data/ace2005event_trigger/test.jsonl.hyps"
TRIGGER_REFS_FILE = "../../data/ace2005event_trigger/test.jsonl.refs"

DATASETS = {}

TASK_MAPPING = {
    "conll04":"Named Entity Recognition",
    "conll04_re":"Relation Extraction",
    "ade":"Named Entity Recognition",
    "ade0":"Named Entity Recognition",
    "ade1":"Named Entity Recognition",
    "ade2":"Named Entity Recognition",
    "ade3":"Named Entity Recognition",
    "ade4":"Named Entity Recognition",
    "ade5":"Named Entity Recognition",
    "ade6":"Named Entity Recognition",
    "ade7":"Named Entity Recognition",
    "ade8":"Named Entity Recognition",
    "ade9":"Named Entity Recognition",
    "ade_re":"Relation Extraction",
    "ade_re0":"Relation Extraction",
    "ade_re1":"Relation Extraction",
    "ade_re2":"Relation Extraction",
    "ade_re3":"Relation Extraction",
    "ade_re4":"Relation Extraction",
    "ade_re5":"Relation Extraction",
    "ade_re6":"Relation Extraction",
    "ade_re7":"Relation Extraction",
    "ade_re8":"Relation Extraction",
    "ade_re9":"Relation Extraction",
    "nyt":"Named Entity Recognition",
    "nyt_re":"Relation Extraction",
    "ace2005_joint_er":"Named Entity Recognition",
    "ace2005_joint_er_re":"Relation Extraction",
    "conll03":"Named Entity Recognition",
    "oie_nyt":"Open Information Extraction",
    "oie_oie2016":"Open Information Extraction",
    "oie_penn":"Open Information Extraction",
    "oie_web":"Open Information Extraction",
    "ontonotes":"Named Entity Recognition",
    "genia":"Named Entity Recognition",
    "googlere":"Factual Probe",
    "ace2005_ner":"Named Entity Recognition",
    "ace2005event_trigger":"Event Extraction",
    "ace2005event_argument":"Event Extraction",
    "conll05_srl_wsj":"Semantic Role Labeling",
    "conll05_srl_brown":"Semantic Role Labeling",
    "conll12_srl":"Semantic Role Labeling",
    "conll12_coref":"Coreference Resolution",
    "tacred":"Relation Classification",
    "trex":"Factual Probe",
    "FewRel":"Relation Classification",
    "FewRelEpisodic":"Relation Classification",
    "atis":"Intent Detection",
    "snips":"Intent Detection",
}

def fix(string):
    return string.strip('[').strip(']') \
                 .replace(' ','') \
                 .replace('.','') \
                 .replace(',','') \
                 .replace('(','') \
                 .replace(')','').lower()

def register_dataset(dataset_class):
    DATASETS[dataset_class.name] = dataset_class
    return dataset_class


def load_dataset(
        dataset_name: str,
        data_args: DataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        split: str,
        max_input_length: int,
        max_output_length: int,
        train_subset: float = 1,
        seed: int = None,
        shuffle: bool = True,
        is_eval: bool = False
):
    """
    Load a registered dataset.
    """
    return DATASETS[dataset_name](
        tokenizer=tokenizer,
        max_input_length=max_input_length,
        max_output_length=max_output_length,
        mode=split,
        overwrite_cache=data_args.overwrite_cache,
        train_subset=train_subset,
        seed=seed,
        shuffle=shuffle,
        data_args=data_args,
        is_eval=is_eval,
    )


class OIEDataset(BaseDataset):
    """
    Base class for datasets of open information extraction.
    """
    entity_types = None
    relation_types = None

    natural_entity_types = None
    natural_relation_types = None

    default_output_format = 'oie'

    LEXICAL_THRESHOLD = 0.5

    def load_cached_data(self, cached_features_file):
        d = torch.load(cached_features_file)
        self.entity_types, self.relation_types, self.examples, self.features = \
            d['entity_types'], d['relation_types'], d['examples'], d['features']

    def save_data(self, cached_features_file):
        torch.save({
            'entity_types': self.entity_types,
            'relation_types': self.relation_types,
            'examples': self.examples,
            'features': self.features,
        }, cached_features_file)

    def load_schema(self):
        """
        Load entity and relation types.

        This is the default implementation which uses the dictionaries natural_entity_types and natural_relation_types.
        """
        if self.natural_entity_types is not None:
            self.entity_types = {short: EntityType(
                short=short,
                natural=natural,
            ) for short, natural in self.natural_entity_types.items()}

        if self.natural_relation_types is not None:
            self.relation_types = {short: RelationType(
                short=short,
                natural=natural,
            ) for short, natural in self.natural_relation_types.items()}

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).

        This is the default implementation for datasets in the SpERT format
        (see https://github.com/markus-eberts/spert).
        """
        examples = []
        name = self.name if self.data_name is None else self.data_name
        file_path = os.path.join(self.data_dir(), f'{name}_{split}.json')

        with open(file_path, 'r') as f:
            data = json.load(f)
            logging.info(f"Loaded {len(data)} sentences for split {split} of {self.name}")

            for i, x in enumerate(data):
                entities = [
                    Entity(id=j, type=y['type'], start=y['start'], end=y['end'])
                    for j, y in enumerate(x['entities'])
                ]

                relations = [
                    Relation(
                        type=y['type'], head=entities[y['head']], tail=entities[y['tail']]
                    )
                    for y in x['relations']
                ]

                tokens = x['tokens']

                example = InputExample(
                    id=f'{split}-{i}',
                    tokens=tokens,
                    entities=entities,
                    relations=relations,
                )
                examples.append(example)

        return examples

    def _lexical_match(self, extraction, reference):
        sRef = reference.split(' ')
        sEx = extraction.split(' ')
        count = 0

        for w1 in sRef:
            for w2 in sEx:
                if w1 == w2:
                    count += 1




        coverage = float(count) / len(sRef)

        return coverage > self.LEXICAL_THRESHOLD

    def lexical_match(self, predicted_triples, true_triples):
        correct_count = 0
        for extraction in predicted_triples:
            for reference in true_triples:
                correct_count += self._lexical_match(extraction.replace('(', '').replace(')', '').strip(), reference.replace('(', '').replace(')', '').strip())
        return correct_count



    def evaluate_example(self, example: InputExample, output_sentence: str, model=None, tokenizer=None, mode='default') -> Counter:
        """
        Evaluate an output sentence on a single example of this dataset.
        """

        predicted_relations, true_relations, wrong_reconstruction, format_error = self.output_format.run_inference(
            example,
            output_sentence,
            entity_types=self.entity_types,
            relation_types=self.relation_types,
            mode = mode,
        )


        correct_relations = self.lexical_match(predicted_relations, true_relations)

        res = Counter({
            'num_sentences': 1,
            'wrong_reconstructions': 1 if wrong_reconstruction else 0,
            'format_error': 1 if format_error else 0,
            'true_relations': len(true_relations),
            'predicted_relations': len(predicted_relations),
            'correct_relations': correct_relations,
        })

        return res


    def _evaluate_dataset_calculate_results(self, results):

        relation_precision, relation_recall, relation_f1 = get_precision_recall_f1(
            num_correct=results['correct_relations'],
            num_predicted=results['predicted_relations'],
            num_gt=results['true_relations'],
        )

        res = {
            'wrong_reconstruction': results['wrong_reconstructions'] / results['num_sentences'],
            'format_error': results['format_error'] / results['num_sentences'],
            'relation_precision': relation_precision,
            'relation_recall': relation_recall,
            'relation_f1': relation_f1,
        }

        return res

    def evaluate_dataset(self, data_args: DataTrainingArguments, model, device, batch_size: int, macro: bool = False, mode: str = 'default', external: str = None) \
            -> Dict[str, float]:
        results = Counter()

        for example, output_sentence in (
            self.generate_output_sentences(data_args, model, device, batch_size)
            if external is None else self.generate_external_pairs(external)
        ):
            new_result = self.evaluate_example(
                    example=example,
                    output_sentence=output_sentence,
                    model=model,
                    tokenizer=self.tokenizer,
                    mode=mode
            )
            results += new_result

        return self._evaluate_dataset_calculate_results(results)

    def load_data(self, mode: str, seed: int = None, glm: bool = False) -> List[InputExample]:
        """
        Load all data, where 'mode' is a list of comma-separated splits to use.
        """
        examples = []

        if isinstance(mode, str):
            splits = mode.split(',')
        else:
            assert isinstance(mode, (list, tuple))
            splits = mode


        if glm:
            for split in splits:
                for i in range(self.num_episodes):
                    examples += self.load_data_single_split(split, seed=i)
        else:
            for split in splits:
                examples += self.load_data_single_split(split, seed=seed)

        return examples



    def generate_external_pairs(self, external):
        predictions = []

        if os.path.exists(external):
            with open(external, "r") as f:
                predictions = [line for line in f]
        else:
            assert (0), "External file '{0}' not found!".format(external)

        for i, pred_text in enumerate(predictions):
            example = self.get_example(i)
            yield example, pred_text

    def preprocess_for_glm_single(self, examples, mode, dataset):
        source = []; target = []
        for example in examples:
            source.append(self.output_format.SOURCE_FORMATS['oie'].format(dataset.split('_')[-1],' '.join(example.tokens),TASK_MAPPING[dataset],TASK_MAPPING[dataset]))
            target.append(self.output_format.format_output(example,mode))
        return source, target

    def preprocess_for_glm(self, mode, dataset, fewshot=-1, debug=False):
        dataset_name = self.name; DATA_DIR = self.data_dir()+"/"
        source,target = self.preprocess_for_glm_single(self.load_data_single_split('train'), mode, dataset)
        ind = [i for i in range(len(source))]
        if fewshot>0:
            np.random.seed(0)
            np.random.shuffle(ind)
            ind = ind[:fewshot]
        source, target = [source[i] for i in ind], [target[i] for i in ind]
        with open(DATA_DIR+"train.source","w") as f:
            for line in source:
                print((self.prompt if mode=="fewshot" else "")+line, file=f)
        with open(DATA_DIR+"train.target","w") as f:
            for line in target:
                print(line, file=f)

        source,target = self.preprocess_for_glm_single(self.load_data_single_split('dev'), mode, dataset)
        ind = [i for i in range(len(source))]
        if debug:
            np.random.seed(0)
            np.random.shuffle(ind)
            ind = ind[:8]
        source, target = [source[i] for i in ind], [target[i] for i in ind]
        with open(DATA_DIR+"val.source","w") as f:
            for line in source:
                print((self.prompt if mode=="fewshot" else "")+line, file=f)
        with open(DATA_DIR+"val.target","w") as f:
            for line in target:
                print(line, file=f)

        source,target = self.preprocess_for_glm_single(self.load_data_single_split('test'), mode, dataset)
        ind = [i for i in range(len(source))]
        if debug:
            np.random.seed(0)
            np.random.shuffle(ind)
            ind = ind[:8]
        source, target = [source[i] for i in ind], [target[i] for i in ind]
        with open(DATA_DIR+"test.source","w") as f:
            for line in source:
                print((self.prompt if mode=="fewshot" else "")+line, file=f)
        with open(DATA_DIR+"test.target","w") as f:
            for line in target:
                print(line, file=f)

@register_dataset
class OIE2016Dataset(OIEDataset):
    """
    OIE2016 dataset (open information extraction).

    The original data is in https://github.com/gabrielStanovsky/supervised-oie/tree/0e9beb33c472c3540b42547cc772f000fac3ae91/supervised-oie-benchmark/oie_corpus

    Download our reprocessed formatted data here https://cloud.tsinghua.edu.cn/d/fd63cca83fa442a29d57/
    """
    name = 'oie_oie2016'


@register_dataset
class OIE_NYTDataset(OIEDataset):
    """
    NYT dataset (open information extraction).

    The original data is in https://github.com/gabrielStanovsky/supervised-oie/tree/0e9beb33c472c3540b42547cc772f000fac3ae91/supervised-oie-benchmark/oie_corpus

    Download our reprocessed formatted data here https://cloud.tsinghua.edu.cn/d/fd63cca83fa442a29d57/
    """
    name = 'oie_nyt'


@register_dataset
class WEBDataset(OIEDataset):
    """
    WEB dataset (open information extraction).

    The original data is in https://github.com/gabrielStanovsky/supervised-oie/tree/0e9beb33c472c3540b42547cc772f000fac3ae91/supervised-oie-benchmark/oie_corpus

    Download our reprocessed formatted data here https://cloud.tsinghua.edu.cn/d/fd63cca83fa442a29d57/
    """
    name = 'oie_web'


@register_dataset
class PENNDataset(OIEDataset):
    """
    PENN dataset (open information extraction).

    The original data is in https://github.com/gabrielStanovsky/supervised-oie/tree/0e9beb33c472c3540b42547cc772f000fac3ae91/supervised-oie-benchmark/oie_corpus

    Download our reprocessed formatted data here https://cloud.tsinghua.edu.cn/d/fd63cca83fa442a29d57/
    """
    name = 'oie_penn'


class JointERDataset(BaseDataset):
    """
    Base class for datasets of joint entity and relation extraction.
    """
    entity_types = None
    relation_types = None

    natural_entity_types = None
    natural_relation_types = None

    default_output_format = 'joint_er'

    def load_cached_data(self, cached_features_file):
        d = torch.load(cached_features_file)
        self.entity_types, self.relation_types, self.examples, self.features = \
            d['entity_types'], d['relation_types'], d['examples'], d['features']

    def save_data(self, cached_features_file):
        torch.save({
            'entity_types': self.entity_types,
            'relation_types': self.relation_types,
            'examples': self.examples,
            'features': self.features,
        }, cached_features_file)

    def load_schema(self):
        """
        Load entity and relation types.

        This is the default implementation which uses the dictionaries natural_entity_types and natural_relation_types.
        """
        if self.natural_entity_types is not None:
            self.entity_types = {short: EntityType(
                short=short,
                natural=natural,
            ) for short, natural in self.natural_entity_types.items()}

        if self.natural_relation_types is not None:
            self.relation_types = {short: RelationType(
                short=short,
                natural=natural,
            ) for short, natural in self.natural_relation_types.items()}

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).

        This is the default implementation for datasets in the SpERT format
        (see https://github.com/markus-eberts/spert).
        """
        examples = []
        name = self.name if self.data_name is None else self.data_name
        file_path = os.path.join(self.data_dir(), f'{name}_{split}.json')

        with open(file_path, 'r') as f:
            data = json.load(f)
            logging.info(f"Loaded {len(data)} sentences for split {split} of {self.name}")

            for i, x in enumerate(data):
                entities = [
                    Entity(id=j, type=self.entity_types[y['type']], start=y['start'], end=y['end'])
                    for j, y in enumerate(x['entities'])
                ]

                relations = [
                    Relation(
                        type=self.relation_types[y['type']], head=entities[y['head']], tail=entities[y['tail']]
                    )
                    for y in x['relations']
                ]

                tokens = x['tokens']

                example = InputExample(
                    id=f'{split}-{i}',
                    tokens=tokens,
                    entities=entities,
                    relations=relations,
                )

                examples.append(example)

        return examples

    def evaluate_example(self, example: InputExample, output_sentence: str, model=None, tokenizer=None, mode='default') -> Counter:
        """
        Evaluate an output sentence on a single example of this dataset.
        """

        predicted_relations, predicted_entities, predicted_entities_no_type, true_relations, true_entities, true_entities_no_type, wrong_reconstruction, label_error, entity_error, format_error = self.output_format.run_inference(
            example,
            output_sentence,
            entity_types=self.entity_types,
            relation_types=self.relation_types,
            mode = mode,
        )


        correct_entities = predicted_entities & true_entities
        correct_entities_no_type = predicted_entities_no_type & true_entities_no_type


        correct_relations = predicted_relations & true_relations


        close_relations = set(correct_relations)
        for relation in true_relations:
            if relation not in correct_relations:
                for prediction in predicted_relations:
                    if( (relation[1] == prediction[1]) and (prediction not in correct_relations)
                    and((relation[0] in prediction[0]) or  (prediction[0] in relation[0]))
                    and((relation[2] in prediction[2]) or  (prediction[2] in relation[2]))):
                        close_relations.add(relation)

        res = Counter({
            'num_sentences': 1,
            'wrong_reconstructions': 1 if wrong_reconstruction else 0,
            'label_error': 1 if label_error else 0,
            'entity_error': 1 if entity_error else 0,
            'format_error': 1 if format_error else 0,
            'true_entities': len(true_entities),
            'predicted_entities': len(predicted_entities),
            'correct_entities': len(correct_entities),
            'true_entities_no_type': len(true_entities_no_type),
            'predicted_entities_no_type': len(predicted_entities_no_type),
            'correct_entities_no_type': len(correct_entities_no_type),
            'true_relations': len(true_relations),
            'predicted_relations': len(predicted_relations),
            'correct_relations': len(correct_relations),
            'close_relations': len(close_relations),
        })

        return res


    def _evaluate_dataset_calculate_results(self, results):
        entity_precision, entity_recall, entity_f1 = get_precision_recall_f1(
            num_correct=results['correct_entities'],
            num_predicted=results['predicted_entities'],
            num_gt=results['true_entities'],
        )

        entity_precision_no_type, entity_recall_no_type, entity_f1_no_type = get_precision_recall_f1(
            num_correct=results['correct_entities_no_type'],
            num_predicted=results['predicted_entities_no_type'],
            num_gt=results['true_entities_no_type'],
        )

        entity_precision_by_type = []
        entity_recall_by_type = []
        entity_f1_by_type = []

        relation_precision, relation_recall, relation_f1 = get_precision_recall_f1(
            num_correct=results['correct_relations'],
            num_predicted=results['predicted_relations'],
            num_gt=results['true_relations'],
        )

        close_relation_precision, close_relation_recall, close_relation_f1 = get_precision_recall_f1(
            num_correct=results['close_relations'],
            num_predicted=results['predicted_relations'],
            num_gt=results['true_relations'],
        )

        res = {
            'wrong_reconstruction': results['wrong_reconstructions'] / results['num_sentences'] if results['num_sentences']>0 else 0,
            'label_error': results['label_error'] / results['num_sentences'] if results['num_sentences']>0 else 0,
            'entity_error': results['entity_error'] / results['num_sentences'] if results['num_sentences']>0 else 0,
            'format_error': results['format_error'] / results['num_sentences'] if results['num_sentences']>0 else 0,
            'entity_precision': entity_precision,
            'entity_recall': entity_recall,
            'entity_f1': entity_f1,
            'relation_precision': relation_precision,
            'relation_recall': relation_recall,
            'relation_f1': relation_f1,
            'entity_precision_no_type': entity_precision_no_type,
            'entity_recall_no_type': entity_recall_no_type,
            'entity_f1_no_type': entity_f1_no_type,
            'close_relation_precision': close_relation_precision,
            'close_relation_recall': close_relation_recall,
            'close_relation_f1': close_relation_f1,
        }

        return res

    def evaluate_dataset(self, data_args: DataTrainingArguments, model, device, batch_size: int, macro: bool = False, mode: str = 'default', external: str = None) \
            -> Dict[str, float]:
        results = Counter()

        for example, output_sentence in (
            self.generate_output_sentences(data_args, model, device, batch_size)
            if external is None else self.generate_external_pairs(external)
        ):
            new_result = self.evaluate_example(
                    example=example,
                    output_sentence=output_sentence,
                    model=model,
                    tokenizer=self.tokenizer,
                    mode=mode
            )
            results += new_result

        return self._evaluate_dataset_calculate_results(results)

    def load_data(self, mode: str, seed: int = None, glm: bool = False) -> List[InputExample]:
        """
        Load all data, where 'mode' is a list of comma-separated splits to use.
        """
        examples = []

        if isinstance(mode, str):
            splits = mode.split(',')
        else:
            assert isinstance(mode, (list, tuple))
            splits = mode


        if glm:
            for split in splits:
                for i in range(self.num_episodes):
                    examples += self.load_data_single_split(split, seed=i)
        else:
            for split in splits:
                examples += self.load_data_single_split(split, seed=seed)

        return examples


    def generate_external_pairs(self, external):
        predictions = []

        if os.path.exists(external):
            with open(external, "r") as f:
                predictions = [line for line in f]
        else:
            assert (0), "External file '{0}' not found!".format(external)

        for i, pred_text in enumerate(predictions):
            example = self.get_example(i)
            yield example, pred_text


    def preprocess_for_glm_single(self, examples, mode, dataset):
        source = []; target = []
        for example in examples:
            source.append(self.output_format.SOURCE_FORMATS[mode].format(
            dataset.replace('_re','').strip('0').strip('1').strip('2').strip('3').strip('4').strip('5').strip('6').strip('7').strip('8').strip('9'),
            ' '.join(example.tokens),TASK_MAPPING[dataset]))
            target.append(self.output_format.format_output(example,mode))
        return source, target

    def preprocess_for_glm(self, mode, dataset, fewshot=-1, debug=False):
        dataset_name = self.name; DATA_DIR = self.data_dir()+"/"
        source,target = self.preprocess_for_glm_single(self.load_data_single_split('train'), mode, dataset)


        ind = [i for i in range(len(source))]
        if fewshot>0:
            np.random.seed(0)
            np.random.shuffle(ind)
            ind = ind[:fewshot]
        source, target = [source[i] for i in ind], [target[i] for i in ind]
        with open(DATA_DIR+"train.source","w") as f:
            for line in source:
                print((self.prompt if mode=="fewshot" else "")+line, file=f)
        with open(DATA_DIR+"train.target","w") as f:
            for line in target:
                print(line, file=f)

        source,target = self.preprocess_for_glm_single(self.load_data_single_split('dev'), mode, dataset)
        ind = [i for i in range(len(source))]
        if debug:
            np.random.seed(0)
            np.random.shuffle(ind)
            ind = ind[:8]
        source, target = [source[i] for i in ind], [target[i] for i in ind]
        with open(DATA_DIR+"val.source","w") as f:
            for line in source:
                print((self.prompt if mode=="fewshot" else "")+line, file=f)
        with open(DATA_DIR+"val.target","w") as f:
            for line in target:
                print(line, file=f)

        source,target = self.preprocess_for_glm_single(self.load_data_single_split('test'), mode, dataset)
        ind = [i for i in range(len(source))]
        if debug:
            np.random.seed(0)
            np.random.shuffle(ind)
            ind = ind[:8]
        source, target = [source[i] for i in ind], [target[i] for i in ind]
        with open(DATA_DIR+"test.source","w") as f:
            for line in source:
                print((self.prompt if mode=="fewshot" else "")+line, file=f)
        with open(DATA_DIR+"test.target","w") as f:
            for line in target:
                print(line, file=f)

class RelationExtractionDataset(JointERDataset):
    default_output_format = 're'

@register_dataset
class Conll04Dataset(JointERDataset):
    """
    CoNLL04 dataset (joint entity and relation extraction).

    Downloaded using https://github.com/markus-eberts/spert/blob/master/scripts/fetch_datasets.sh
    """
    name = 'conll04'
    num_episodes = 1

    prompt = "conll04 sentence : Newspaper ` Explains ' U.S. Interests Section Events FL1402001894 Havana Radio Reloj Network in Spanish 2100 GMT 13 Feb 94 [SEP] tuples : ( radio reloj network ; organization based in ; havana ) ( radio reloj network ; instance of ; organization ) ( havana ; instance of ; location ) ( u.s. ; instance of ; location ) ( 2100 gmt ; instance of ; other ) ( 13 feb 94 ; instance of ; other ) [SEP] conll04 sentence : ` ` If it does not snow , and a lot , within this month we will have no water to submerge 150 , 000 hectares ( 370 , 500 acres ) of rice , ' ' said Bruno Pusterla , a top official of the Italian Agricultural Confederation . [SEP] tuples : ( bruno pusterla ; works for ; italian agricultural confederation ) ( bruno pusterla ; instance of ; person ) ( italian agricultural confederation ; instance of ; organization ) ( 150 , 000 hectares ; instance of ; other ) ( 370 , 500 acres ; instance of ; other ) ( rice ; instance of ; other ) [SEP] conll04 sentence : The self-propelled rig Avco 5 was headed to shore with 14 people aboard early Monday when it capsized about 20 miles off the Louisiana coast , near Morgan City , Lifa said. [SEP] tuples : ( morgan city ; located in ; louisiana ) ( morgan city ; instance of ; location ) ( louisiana ; instance of ; location ) ( 20 miles ; instance of ; other ) ( lifa ; instance of ; person ) [SEP] conll04 sentence : Annie Oakley , also known as Little Miss Sure Shot , was born Phoebe Ann Moses in Willowdell , Darke County , in 1860 . [SEP] tuples : ( annie oakley ; lives in ; willowdell , darke county ) ( annie oakley ; instance of ; person ) ( willowdell , darke county ; instance of ; location ) ( little miss sure shot ; lives in ; willowdell , darke county ) ( little miss sure shot ; instance of ; person ) ( willowdell , darke county ; instance of ; location ) ( phoebe ann moses ; lives in ; willowdell , darke county ) ( phoebe ann moses ; instance of ; person ) ( willowdell , darke county ; instance of ; location ) [SEP] conll04 sentence : Penry raped Pamela Moseley Carpenter on Oct. 15 , 1979 , in Livingston , Texas , then stabbed her to death . [SEP] tuples : ( livingston ; located in ; texas ) ( livingston ; instance of ; location ) ( texas ; instance of ; location ) ( penry ; instance of ; person ) ( pamela moseley carpenter ; instance of ; person ) ( oct. 15 , 1979 ; instance of ; other ) [SEP] "

    natural_entity_types = {
        'Loc': 'location',
        'Org': 'organization',
        'Peop': 'person',
        'Other': 'other',
    }

    natural_relation_types = {
        'Work_For': 'works for',
        'Kill': 'kills',
        'OrgBased_In': 'organization based in',
        'Live_In': 'lives in',
        'Located_In': 'located in'
    }

@register_dataset
class Conll04REDataset(Conll04Dataset):
    name = 'conll04_re'
    default_output_format = 're'

@register_dataset
class ADEDataset(JointERDataset):
    """
    ADE dataset (joint entity and relation extraction).

    Downloaded using https://github.com/markus-eberts/spert/blob/master/scripts/fetch_datasets.sh
    """
    name = 'ade'

    prompt = "ade sentence : We report a case of fulminant hepatic failure associated with didanosine and masquerading as a surgical abdomen and compare the clinical , biologic , histologic , and ultrastructural findings with r [SEP] tuples: ( fulminant hepatic failure ; effect ; didanosine ) ( fulminant hepatic failure ; instance of ; disease ) ( didanosine ; instance of ; drug ) [SEP] ade sentence : Depressive symptoms disappeared after interferon therapy was stopped . [SEP] tuples: ( depressive symptoms ; effect ; interferon ) ( depressive symptoms ; instance of ; disease ) ( interferon ; instance of ; drug ) [SEP] ade sentence : A case of a 53-year - old man who developed acute pneumonitis after bleomycin and moderate oxygen administration is presented . [SEP] tuples: ( acute pneumonitis ; effect ; bleomycin ) ( acute pneumonitis ; instance of ; disease ) ( bleomycin ; instance of ; drug ) ( acute pneumonitis ; effect ; oxygen ) ( acute pneumonitis ; instance of ; disease ) ( oxygen ; instance of ; drug ) [SEP] ade sentence : Systemic corticosteroids in the phenytoin hypersensitivity syndrome . [SEP] tuples: ( phenytoin hypersensitivity syndrome ; effect ; phenytoin ) ( phenytoin hypersensitivity syndrome ; instance of ; disease ) ( phenytoin ; instance of ; drug ) [SEP] ade sentence : Pathogenesis of methotrexate - induced papular eruption in psoriasis may involve immune mechanisms other than those of methotrexate - induced cutaneous vasculitis in collagen vascular disease . [SEP] tuples: ( cutaneous vasculitis ; effect ; methotrexate ) ( cutaneous vasculitis ; instance of ; disease ) ( methotrexate ; instance of ; drug ) ( papular eruption ; effect ; methotrexate ) ( papular eruption ; instance of ; disease ) ( methotrexate ; instance of ; drug ) [SEP] "

    natural_entity_types = {
        'Adverse-Effect': 'disease',
        'Drug': 'drug',
    }

    natural_relation_types = {
        'Adverse-Effect': 'effect',
    }

    num_episodes = 10

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).

        We decide which split to use based on the seed.
        In this way, running episodes 1-10 has the effect of running on all 10 different splits once.
        """
        if seed is None:
            i = 0
        else:
            i = seed % self.num_episodes

        if split == 'train':
            return super().load_data_single_split(f'split_{i}_train', seed)

        elif split == 'dev':
            return []

        elif split == 'test':
            return super().load_data_single_split(f'split_{i}_test', seed)

    def evaluate_dataset(self, *args, **kwargs):
        """
        Evaluate model on this dataset.

        We include the macro entity scores, since it is standard to report them.
        """
        return super().evaluate_dataset(*args, **kwargs, macro=True)


    def preprocess_for_glm(self, mode, dataset, fewshot=-1, debug=False):
        dataset_name = self.name; DATA_DIR = self.data_dir()
        all_train_source, all_train_target = [], []
        all_val_source, all_val_target = [], []
        all_test_source, all_test_target = [], []
        for seed in range(self.num_episodes):
            source, target = [], []
            S,T = self.preprocess_for_glm_single(self.load_data_single_split('train', seed=seed), mode, dataset)
            source.extend(S); target.extend(T)
            ind = [i for i in range(len(source))]
            if fewshot>0:
                np.random.seed(0)
                np.random.shuffle(ind)
                ind = ind[:fewshot]
            source, target = [source[i] for i in ind], [target[i] for i in ind]
            if not os.path.exists(DATA_DIR+f"{seed}"):
                os.mkdir(DATA_DIR+f"{seed}")
            with open(DATA_DIR+f"{seed}/train.source","w") as f:
                for line in source:
                    print((self.prompt if mode=="fewshot" else "")+line, file=f)
            with open(DATA_DIR+f"{seed}/train.target","w") as f:
                for line in target:
                    print(line, file=f)
            all_train_source.extend(source)
            all_train_target.extend(target)


            source, target = [], []
            S,T = self.preprocess_for_glm_single(self.load_data_single_split('dev', seed=seed), mode, dataset)
            source.extend(S); target.extend(T)
            ind = [i for i in range(len(source))]
            if debug:
                np.random.seed(0)
                np.random.shuffle(ind)
                ind = ind[:8]
            source, target = [source[i] for i in ind], [target[i] for i in ind]
            with open(DATA_DIR+f"{seed}/val.source","w") as f:
                for line in source:
                    print((self.prompt if mode=="fewshot" else "")+line, file=f)
            with open(DATA_DIR+f"{seed}/val.target","w") as f:
                for line in target:
                    print(line, file=f)
            all_val_source.extend(source)
            all_val_target.extend(target)

            source, target = [], []
            S,T = self.preprocess_for_glm_single(self.load_data_single_split('test', seed=seed), mode, dataset)
            source.extend(S); target.extend(T)
            ind = [i for i in range(len(source))]
            if debug:
                np.random.seed(0)
                np.random.shuffle(ind)
                ind = ind[:8]
            source, target = [source[i] for i in ind], [target[i] for i in ind]
            with open(DATA_DIR+f"{seed}/test.source","w") as f:
                for line in source:
                    print((self.prompt if mode=="fewshot" else "")+line, file=f)
            with open(DATA_DIR+f"{seed}/test.target","w") as f:
                for line in target:
                    print(line, file=f)
            all_test_source.extend(source)
            all_test_target.extend(target)

        with open(DATA_DIR+"/train.source","w") as f:
            for line in all_train_source:
                print((self.prompt if mode=="fewshot" else "")+line, file=f)
        with open(DATA_DIR+"/train.target","w") as f:
            for line in all_train_target:
                print(line, file=f)
        with open(DATA_DIR+"/val.source","w") as f:
            for line in all_val_source:
                print((self.prompt if mode=="fewshot" else "")+line, file=f)
        with open(DATA_DIR+"/val.target","w") as f:
            for line in all_val_target:
                print(line, file=f)
        with open(DATA_DIR+"/test.source","w") as f:
            for line in all_test_source:
                print((self.prompt if mode=="fewshot" else "")+line, file=f)
        with open(DATA_DIR+"/test.target","w") as f:
            for line in all_test_target:
                print(line, file=f)

@register_dataset
class ADEREDataset(ADEDataset):
    name = 'ade_re'
    default_output_format = 're'

DATASETS['ade0'] = ADEDataset; DATASETS['ade_re0'] = ADEREDataset
DATASETS['ade1'] = ADEDataset; DATASETS['ade_re1'] = ADEREDataset
DATASETS['ade2'] = ADEDataset; DATASETS['ade_re2'] = ADEREDataset
DATASETS['ade3'] = ADEDataset; DATASETS['ade_re3'] = ADEREDataset
DATASETS['ade4'] = ADEDataset; DATASETS['ade_re4'] = ADEREDataset
DATASETS['ade5'] = ADEDataset; DATASETS['ade_re5'] = ADEREDataset
DATASETS['ade6'] = ADEDataset; DATASETS['ade_re6'] = ADEREDataset
DATASETS['ade7'] = ADEDataset; DATASETS['ade_re7'] = ADEREDataset
DATASETS['ade8'] = ADEDataset; DATASETS['ade_re8'] = ADEREDataset
DATASETS['ade9'] = ADEDataset; DATASETS['ade_re9'] = ADEREDataset


@register_dataset
class NYTDataset(JointERDataset):
    """
    NYT dataset (joint entity and relation extraction).

    Downloaded from https://github.com/yubowen-ph/JointER/tree/master/dataset/NYT-multi/data
    """
    name = 'nyt'
    default_input_format = name

    natural_entity_types = {
        'PERSON': 'person',
        'LOCATION': 'location',
        'ORGANIZATION': 'organization',
    }

    num_episodes = 1

    @staticmethod
    def to_natural_relation_type(relation_type: str) -> str:

        return relation_type.split('/')[-1].replace('_', ' ')

    def load_schema(self):
        """
        Load entity and relation types.
        """

        super().load_schema()


        with open(os.path.join(self.data_dir(), f'schemas.json'), 'r') as f:
            types = json.load(f)





            self.relation_types = {name: RelationType(
                natural=self.to_natural_relation_type(name)
            ) for name in types[0].values()}

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        examples = []
        file_path = os.path.join(self.data_dir(), f'{split}.json')

        with open(file_path, 'r') as f:
            data = json.load(f)
            logging.info(f"Loaded {len(data)} sentences for split {split} of {self.name}")

            for i, x in enumerate(data):
                entities = []
                relations = []

                for y in x['spo_details']:
                    entity1_start, entity1_end, entity1_type, relation_type, \
                        entity2_start, entity2_end, entity2_type = y

                    entity1 = Entity(type=self.entity_types[entity1_type], start=entity1_start, end=entity1_end)
                    entity2 = Entity(type=self.entity_types[entity2_type], start=entity2_start, end=entity2_end)

                    try:
                        i1 = entities.index(entity1)
                    except ValueError:

                        i1 = len(entities)
                        entities.append(entity1)

                    try:
                        i2 = entities.index(entity2)
                    except ValueError:

                        i2 = len(entities)
                        entities.append(entity2)

                    relation = Relation(
                        type=self.relation_types[relation_type], head=entities[i1], tail=entities[i2],
                    )

                    relations.append(relation)

                tokens = x['tokens']

                example = InputExample(
                    id=f'{split}-{i}',
                    tokens=tokens,
                    entities=entities,
                    relations=relations,
                )
                examples.append(example)

        return examples

@register_dataset
class NYTREDataset(NYTDataset):
    name = 'nyt_re'
    default_output_format = 're'

@register_dataset
class ACE2005Dataset(JointERDataset):
    """
    ACE2005 dataset (joint entity and relation extraction.

    Processed using https://github.com/luanyi/DyGIE/tree/master/preprocessing
    """
    name = 'ace2005_joint_er'
    default_input_format = name

    natural_entity_types = {
        'PER': 'person',
        'LOC': 'location',
        'ORG': 'organization',
        'VEH': 'vehicle',
        'GPE': 'geographical entity',
        'WEA': 'weapon',
        'FAC': 'facility',
    }

    natural_relation_types = {
        'PHYS': 'located in',
        'ART': 'artifact',
        'ORG-AFF': 'employer',
        'GEN-AFF': 'affiliation',
        'PER-SOC': 'social',
        'PART-WHOLE': 'part of',
    }

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        file_path = os.path.join(self.data_dir(), f'{split}.json')

        examples = []
        num_documents = 0
        num_entities = 0
        num_relations = 0

        with open(file_path, 'r') as f:
            for j, l in enumerate(f):
                document = json.loads(l)
                num_documents += 1
                offset = 0

                for i, tokens in enumerate(document['sentences']):
                    num_entities += len(document['ner'][i])
                    num_relations += len(document['relations'][i])

                    # if len(document['ner'][i]) > 0:
                    entities = [
                        Entity(type=self.entity_types[entity_type], start=start-offset, end=end-offset+1)
                        for start, end, entity_type in document['ner'][i]
                    ]

                    relations = []

                    skip = False
                    for start1, end1, start2, end2, relation_type in document['relations'][i]:

                        if len([e for e in entities if e.start == start1-offset and e.end == end1-offset+1]) > 1 \
                                or \
                                len([e for e in entities if e.start == start2-offset and e.end == end2-offset+1]) \
                                > 1:
                            skip = True
                            break

                        [head] = [e for e in entities if e.start == start1-offset and e.end == end1-offset+1]
                        [tail] = [e for e in entities if e.start == start2-offset and e.end == end2-offset+1]

                        relations.append(
                            Relation(type=self.relation_types[relation_type], head=head, tail=tail)
                        )

                    if not skip:
                        example = InputExample(
                            id=f'{split}-{j}-{i}',
                            tokens=tokens,
                            entities=entities,
                            relations=relations,
                        )
                        examples.append(example)

                    offset += len(tokens)

        logging.info(f'Constructed {len(examples)} examples (from {num_documents} documents) for {self.name} ({split})')
        return examples

@register_dataset
class ACE2005REDataset(ACE2005Dataset):
    name = 'ace2005_joint_er_re'
    default_output_format = 're'

class NERDataset(JointERDataset):
    """
    Base class for NER datasets.
    """

    default_output_format = "ner"

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        file_path = os.path.join(self.data_dir(), f'{split}.txt')

        raw_examples = []
        tokens = []
        labels = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        raw_examples.append((tokens, labels))
                        tokens = []
                        labels = []
                else:
                    splits = line.split()
                    tokens.append(splits[0])
                    if len(splits) > 1:
                        label = splits[-1].strip()
                        if label == 'O':
                            label = None
                        labels.append(label)
                    else:
                        labels.append(None)

            if tokens:
                raw_examples.append((tokens, labels))

        logging.info(f"Loaded {len(raw_examples)} sentences for split {split} of {self.name}")

        examples = []
        for i, (tokens, labels) in enumerate(raw_examples):
            assert len(tokens) == len(labels)


            entities = []

            current_entity_start = None
            current_entity_type = None

            for j, label in enumerate(labels + [None]):
                previous_label = labels[j-1] if j > 0 else None
                if (label is None and previous_label is not None) \
                        or (label is not None and previous_label is None) \
                        or (label is not None and previous_label is not None and (
                            label[2:] != previous_label[2:] or label.startswith('B-') or label.startswith('S-')
                        )):
                    if current_entity_start is not None:

                        entities.append(Entity(
                            id=len(entities),
                            type=self.entity_types[current_entity_type],
                            start=current_entity_start,
                            end=j,
                        ))

                        current_entity_start = None
                        current_entity_type = None

                    if label is not None:

                        current_entity_start = j
                        assert any(label.startswith(f'{prefix}-') for prefix in 'BIS')
                        current_entity_type = label[2:]
                        assert current_entity_type in self.entity_types

            example = InputExample(
                id=f'{split}-{i}',
                tokens=tokens,
                entities=entities,
                relations=[],
            )
            examples.append(example)

        return examples

    def evaluate_dataset(self, data_args: DataTrainingArguments, model, device, batch_size: int, macro: bool = False, mode: str = 'default', external: str = None) \
            -> Dict[str, float]:
        """
        Evaluate model on this dataset, and return entity metrics only.
        """
        results = super().evaluate_dataset(data_args, model, device, batch_size, macro=macro, mode=mode, external=external)
        return {k: v for k, v in results.items() if k.startswith('entity') and k != 'entity_error'}


@register_dataset
class CoNLL03Dataset(NERDataset):
    """
    CoNLL03 dataset (NER).
    """
    name = 'conll03'

    natural_entity_types = {
        'LOC': 'location',
        'MISC': 'miscellaneous',
        'ORG': 'organization',
        'PER': 'person',
    }

    default_output_format = "ner"

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        file_path = os.path.join(self.data_dir(), f'mrc-ner.{split}')
        examples = []

        with open(file_path, 'r') as f:
            data = json.load(f)

            for i, x in enumerate(data):
                tokens = x['context'].split()
                entities = []

                if 'label' not in x:
                    x['label'] = {
                        x['entity_label']:x['span_position'],
                    }

                for entity_type, l in x['label'].items():
                    for start_end in l:
                        start, end = map(int, start_end.split(';'))
                        end += 1

                        entities.append(Entity(
                            id=len(entities),
                            type=self.entity_types[entity_type],
                            start=start,
                            end=end,
                        ))

                example = InputExample(
                    id=f'{split}-{i}',
                    tokens=tokens,
                    entities=entities,
                    relations=[],
                )
                examples.append(example)

        merged_examples = {' '.join(e.tokens):[] for e in examples}
        merged_tokens = {' '.join(e.tokens):e.tokens for e in examples}
        for e in examples:
            merged_examples[' '.join(e.tokens)].extend(e.entities)
        final_examples = []
        for i, t in enumerate(merged_examples):
            final_examples.append(
                InputExample(
                    id=f'{split}-{i}',
                    tokens=merged_tokens[t],
                    entities=merged_examples[t],
                    relations=[],
                )
            )

        logging.info(f"Loaded {len(final_examples)} sentences for split {split} of {self.name}")
        return final_examples


@register_dataset
class OntonotesDataset(NERDataset):
    """
    Ontonotes dataset (NER).
    """
    name = 'ontonotes'

    natural_entity_types = {
        'CARDINAL': 'cardinal',
        'DATE': 'date',
        'EVENT': 'event',
        'FAC': 'facility',
        'GPE': 'country city state',
        'LANGUAGE': 'language',
        'LAW': 'law',
        'LOC': 'location',
        'MONEY': 'monetary',
        'NORP': 'nationality religious political group',
        'ORDINAL': 'ordinal',
        'ORG': 'organization',
        'PERCENT': 'percent',
        'PERSON': 'person',
        'PRODUCT': 'product',
        'QUANTITY': 'quantity',
        'TIME': 'time',
        'WORK_OF_ART': 'work_of_art',
    }

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        file_path = os.path.join(self.data_dir(), f'{split}.ner')

        raw_examples = []
        tokens = []
        labels = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        raw_examples.append((tokens, labels))
                        tokens = []
                        labels = []
                else:
                    splits = line.split()
                    tokens.append(splits[0])
                    if len(splits) > 1:
                        label = splits[-1].strip()
                        if label == 'O':
                            label = None
                        labels.append(label)
                    else:
                        labels.append(None)

            if tokens:
                raw_examples.append((tokens, labels))

        logging.info(f"Loaded {len(raw_examples)} sentences for split {split} of {self.name}")

        examples = []
        for i, (tokens, labels) in enumerate(raw_examples):
            assert len(tokens) == len(labels)


            entities = []

            current_entity_start = None
            current_entity_type = None

            for j, label in enumerate(labels + [None]):
                previous_label = labels[j-1] if j > 0 else None
                if (label is None and previous_label is not None) \
                        or (label is not None and previous_label is None) \
                        or (label is not None and previous_label is not None and (
                            label[2:] != previous_label[2:] or label.startswith('B-') or label.startswith('S-')
                        )):
                    if current_entity_start is not None:

                        entities.append(Entity(
                            id=len(entities),
                            type=self.entity_types[current_entity_type],
                            start=current_entity_start,
                            end=j,
                        ))

                        current_entity_start = None
                        current_entity_type = None

                    if label is not None:

                        current_entity_start = j
                        assert any(label.startswith(f'{prefix}-') for prefix in 'BIS')
                        current_entity_type = label[2:]
                        assert current_entity_type in self.entity_types

            example = InputExample(
                id=f'{split}-{i}',
                tokens=tokens,
                entities=entities,
                relations=[],
            )
            examples.append(example)

        return examples

@register_dataset
class ACE2005NERDataset(NERDataset):
    """
    ACE2005 dataset (NER).

    Downloaded from https://github.com/ShannonAI/mrc-for-flat-nested-ner/
    """
    name = 'ace2005_ner'

    natural_entity_types = {
        'PER': 'person',
        'LOC': 'location',
        'ORG': 'organization',
        'VEH': 'vehicle',
        'GPE': 'geographical entity',
        'WEA': 'weapon',
        'FAC': 'facility',
    }

    num_episodes = 1

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        file_path = os.path.join(self.data_dir(), f'mrc-ner.{split}')
        examples = []

        with open(file_path, 'r') as f:
            data = json.load(f)

            for i, x in enumerate(data):
                tokens = x['context'].split()
                entities = []

                if 'label' not in x:
                    x['label'] = {
                        x['entity_label']:x['span_position'],
                    }

                for entity_type, l in x['label'].items():
                    for start_end in l:
                        start, end = map(int, start_end.split(';'))
                        end += 1

                        entities.append(Entity(
                            id=len(entities),
                            type=self.entity_types[entity_type],
                            start=start,
                            end=end,
                        ))

                example = InputExample(
                    id=f'{split}-{i}',
                    tokens=tokens,
                    entities=entities,
                    relations=[],
                )
                examples.append(example)

        merged_examples = {' '.join(e.tokens):[] for e in examples}
        merged_tokens = {' '.join(e.tokens):e.tokens for e in examples}
        for e in examples:
            merged_examples[' '.join(e.tokens)].extend(e.entities)
        final_examples = []
        for i, t in enumerate(merged_examples):
            final_examples.append(
                InputExample(
                    id=f'{split}-{i}',
                    tokens=merged_tokens[t],
                    entities=merged_examples[t],
                    relations=[],
                )
            )

        logging.info(f"Loaded {len(final_examples)} sentences for split {split} of {self.name}")
        return final_examples


@register_dataset
class GENIADataset(NERDataset):
    """
    GENIA dataset (NER).
    """
    name = 'genia'

    natural_entity_types = {
        'DNA': 'DNA',
        'RNA': 'RNA',
        'cell_line': 'cell line',
        'cell_type': 'cell type',
        'protein': 'protein',
    }

    num_episodes = 1

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        file_path = os.path.join(self.data_dir(), f'mrc-ner.{split}')
        examples = []

        with open(file_path, 'r') as f:
            data = json.load(f)

            for i, x in enumerate(data):
                tokens = x['context'].split()
                entities = []

                if 'label' not in x:
                    x['label'] = {
                        x['entity_label']:x['span_position'],
                    }

                for entity_type, l in x['label'].items():
                    for start_end in l:
                        start, end = map(int, start_end.split(';'))
                        end += 1

                        entities.append(Entity(
                            id=len(entities),
                            type=self.entity_types[entity_type],
                            start=start,
                            end=end,
                        ))

                example = InputExample(
                    id=f'{split}-{i}',
                    tokens=tokens,
                    entities=entities,
                    relations=[],
                )
                examples.append(example)

        merged_examples = {' '.join(e.tokens):[] for e in examples}
        merged_tokens = {' '.join(e.tokens):e.tokens for e in examples}
        for e in examples:
            merged_examples[' '.join(e.tokens)].extend(e.entities)
        final_examples = []
        for i, t in enumerate(merged_examples):
            final_examples.append(
                InputExample(
                    id=f'{split}-{i}',
                    tokens=merged_tokens[t],
                    entities=merged_examples[t],
                    relations=[],
                )
            )

        logging.info(f"Loaded {len(final_examples)} sentences for split {split} of {self.name}")
        return final_examples

@register_dataset
class ACE2005EventTriggerDataset(NERDataset):
    """
    ACE 2005 dataset (event extraction), trigger extraction component.
    """
    name = 'ace2005event_trigger'
    data_name = 'ace2005event_trigger'

    relation_schemas = json.load(open("./ace2005event_types.json"))

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        self.entity_types = {name:EntityType(short=entity['verbose'],natural=entity['verbose']) for name,entity in self.relation_schemas['entities'].items()}
        self.relation_types = {name:EntityType(short=relation['verbose'],natural=relation['verbose']) for name,relation in self.relation_schemas['relations'].items()}

        examples = []

        file_path = os.path.join(self.data_dir(), f'ace2005event_{split}.json')

        with open(file_path, 'r') as f:
            data = json.load(f)
            logging.info(f"Loaded {len(data)} sentences for split {split} of {self.name}")

            """

            "ner": [
                [...],
                [...],
                [[26, 26, "LOC"], [14, 14, "PER"], ...], #the boundary positions are indexed in the document level
                ...,
            ],
            """

            for i, x in enumerate(data):
                offset = 0
                for each_example in range(len(x['sentences'])):
                    triggers = [
                        Entity(
                            id=j, type=self.entity_types[y[0][1].replace('.',':')],
                            start=y[0][0]-offset, end=y[0][0]-offset+1
                        )
                        for j, y in enumerate(x['events'][each_example])
                    ]

                    tokens = x['sentences'][each_example]
                    offset += len(tokens)
                    example = InputExample(
                        id=f'{split}-{i}',
                        tokens=tokens,
                        entities=triggers,
                        relations=[],
                    )
                    examples.append(example)

        return examples

@register_dataset
class ACE2005EventArgumentDataset(NERDataset):
    """
    ACE 2005 dataset (event extraction), argument extraction component.
    """
    name = 'ace2005event_argument'
    data_name = 'ace2005event_argument'

    default_output_format = 'event_argument'
    relation_schemas = json.load(open("./ace2005event_types.json"))

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        self._relation_types = {name:EntityType(short=relation['verbose'],natural=relation['verbose']) for name,relation in self.relation_schemas['relations'].items()}
        self._entity_types = {name:EntityType(short=entity['verbose'],natural=entity['verbose']) for name,entity in self.relation_schemas['entities'].items()}
        self.entity_types = self._entity_types; self.relation_types = None
        assert (TRIGGER_HYPS_FILE is not None) and (TRIGGER_REFS_FILE is not None), "For argument prediction, triggers must be given!"
        self.pred_triples = [[tuple(triple.split(' ; ')) for triple in line.strip().strip('( ').strip(' )').split(' ) ( ')] for line in open(TRIGGER_HYPS_FILE)]
        self.pred_triggers = set([triple[0] for triple in FlattenList(self.pred_triples) if len(triple)==3 and triple[1]=='instance of'])
        self.true_triples = [[tuple(triple.split(' ; ')) for triple in line.strip().strip('( ').strip(' )').split(' ) ( ')] for line in open(TRIGGER_REFS_FILE)]
        self.true_triggers = set([triple[0] for triple in FlattenList(self.true_triples) if len(triple)==3 and triple[1]=='instance of'])
        self.corr_triggers = self.pred_triggers & self.true_triggers
        self.pred_lookup = {fix(trigger) for trigger in self.pred_triggers}
        self.true_lookup = {fix(trigger) for trigger in self.true_triggers}
        self.corr_lookup = {fix(trigger) for trigger in self.corr_triggers}

        examples = []
        name = self.name if self.data_name is None else self.data_name
        file_path = os.path.join(self.data_dir(), f'ace2005event_{split}.json')

        with open(file_path, 'r') as f:
            data = json.load(f)
            logging.info(f"Loaded {len(data)} sentences for split {split} of {self.name}")

            for i, x in enumerate(data):
                offset = 0
                for each_example in range(len(x['sentences'])):
                    tokens = x['sentences'][each_example]
                    for j, y in enumerate(x['events'][each_example]):
                        trigger = Entity(
                            id=j, type=self._entity_types[y[0][1].replace('.',':')],
                            start=y[0][0]-offset, end=y[0][0]-offset+1
                        )
                        entities = []
                        for target in y[1:]:
                            entities.append(
                                Entity(
                                    id=j, type=self._relation_types[target[2]],
                                    start=target[0]-offset, end=target[1]-offset+1
                                )
                            )

                        sentence = copy.deepcopy(tokens)

                        for entity in entities:
                            entity.start += entity.start>=trigger.end
                            entity.end += entity.end>trigger.end
                        sentence[trigger.end  :trigger.end  ] = [']']
                        for entity in entities:
                            entity.start += entity.start>=trigger.start
                            entity.end += entity.end>trigger.start
                        sentence[trigger.start:trigger.start] = ['[']
                        trigger.start += 1
                        trigger.end += 1
                        example = InputExample(
                            id=f'{split}-{i}',
                            tokens=sentence,
                            triggers=[trigger],
                            entities=entities,
                            relations=[],
                        )
                        examples.append(example)
                    offset += len(tokens)

        return examples

    def parse_argument(self, example, arguments, gt=False):
        assert(len(example.triggers)==1); trigger = example.triggers[0]
        if (not gt) and (fix(' '.join(example.tokens[trigger.start:trigger.end])) not in self.corr_lookup):
            return set(), set()
        valid_relations = set(r.natural for r in self._relation_types.values())
        valid_relations.remove('person'); valid_relations.add('human')
        events = [(arg, arg_type) for arg, _, arg_type in arguments if arg_type in valid_relations]
        events_no_type = [arg for arg, _, arg_type in arguments if arg_type in valid_relations]
        return set(events), set(events_no_type)

    def evaluate_example(self, example: InputExample, output_sentence: str, model=None, tokenizer=None, mode: str = 'default') -> Counter:
        """
        Evaluate an output sentence on a single example of this dataset.
        """


        _, pred_entities, _, _, true_entities, _, wrong_reconstruction, _, _, _ = self.output_format.run_inference(
            example,
            output_sentence,
            entity_types=self.entity_types,
            relation_types=self.relation_types,
            mode = mode,
        )

        true_relations, true_relations_no_type = self.parse_argument(example, true_entities, gt=True)
        pred_relations, pred_relations_no_type = self.parse_argument(example, pred_entities)

        correct_relations = pred_relations & true_relations
        correct_relations_no_type = pred_relations_no_type & true_relations_no_type

        return Counter({
            'num_sentences': 1,
            'wrong_reconstructions': 1 if wrong_reconstruction else 0,
            'gt_relations': len(true_relations),
            'predicted_relations': len(pred_relations),
            'correct_relations': len(correct_relations),
            'gt_relations_no_type': len(true_relations_no_type),
            'predicted_relations_no_type': len(pred_relations_no_type),
            'correct_relations_no_type': len(correct_relations_no_type),
        })




















        return Counter()





























    def evaluate_dataset(self, data_args: DataTrainingArguments, model, device, batch_size: int, macro: bool = False, mode: str = 'default', external: str = None) \
            -> Dict[str, float]:
        """
        Evaluate model on this dataset.
        """
        results = Counter()
        for example, output_sentence in (
            self.generate_output_sentences(data_args, model, device, batch_size)
            if external is None else self.generate_external_pairs(external)
        ):
            new_result = self.evaluate_example(
                    example=example,
                    output_sentence=output_sentence,
                    model=model,
                    tokenizer=self.tokenizer,
                    mode=mode
            )
            results += new_result

        return self._evaluate_dataset_calculate_results(results)

    def _evaluate_dataset_calculate_results(self, results):

        relation_precision, relation_recall, relation_f1 = get_precision_recall_f1(
            num_correct=results['correct_relations'],
            num_predicted=results['predicted_relations'],
            num_gt=results['gt_relations'],
        )

        relation_precision_no_type, relation_recall_no_type, relation_f1_no_type = get_precision_recall_f1(
            num_correct=results['correct_relations_no_type'],
            num_predicted=results['predicted_relations_no_type'],
            num_gt=results['gt_relations_no_type'],
        )

        res = {
            'relation_precision': relation_precision,
            'relation_recall': relation_recall,
            'relation_f1': relation_f1,
            'relation_precision_no_type': relation_precision_no_type,
            'relation_recall_no_type': relation_recall_no_type,
            'relation_f1_no_type': relation_f1_no_type,
            'num_gt_triggers': results['gt_entities'],
            'num_pred_triggers': results['predicted_entities'],
            'num_gt_relations': results['gt_relations'],
            'num_pred_relations': results['predicted_relations'],
        }

        return res


@register_dataset
class ACE2005EventDataset(ACE2005EventArgumentDataset):
    """
    ACE 2005 dataset (event extraction), for evaluation only.
    """
    name = 'ace2005event'
    task_descriptor = 'ace2005event_trigger'
    default_input_format = 'plain'
    default_output_format = 'joint_er'
    argument_input_format = 'ace2005_event_with_trigger'
    argument_output_format = 'ace2005_event'

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        examples = []
        name = self.name if self.data_name is None else self.data_name
        file_path = os.path.join(self.data_dir(), f'{name}_{split}.json')

        with open(file_path, 'r') as f:
            data = json.load(f)
            logging.info(f"Loaded {len(data)} sentences for split {split} of {self.name}")

            for i, x in enumerate(data):
                entities = [
                    Entity(id=j, type=self.entity_types[y['type']], start=y['start'], end=y['end'])
                    for j, y in enumerate(x['entities'])
                ]

                triggers = [
                    Entity(id=j, type=self.entity_types[y['type']], start=y['start'], end=y['end'])
                    for j, y in enumerate(x['triggers'])
                ]

                relations = [

                    Relation(
                        type=self.relation_types[y['type']], head=entities[y['head']], tail=triggers[y['tail']]
                    )
                    for y in x['relations']
                ]

                tokens = x['tokens']

                example = InputExample(
                    id=f'{split}-{i}',
                    tokens=tokens,
                    entities=entities,
                    triggers=triggers,
                    relations=relations,
                )
                examples.append(example)

        return examples

    def evaluate_argument(self, output_format, example_argument_single_trigger: InputExample, example: InputExample,
                          argument_output_sentence: str) -> Tuple[Set[tuple], Set[tuple], Set[tuple]]:
        """
        Perform argument prediction.
        """
        predicted_entities, predicted_relations, wrong_reconstruction = \
            output_format.run_inference(example_argument_single_trigger,
                                        argument_output_sentence,
                                        entity_types=self.entity_types,
                                        relation_types=self.relation_types)




        def filter_relation_tuple(relation_tuple):
            return relation_tuple[0], relation_tuple[1][1:], relation_tuple[2]

        gt_relations = set(filter_relation_tuple(relation.to_tuple()) for relation in example.relations)


        filtered_predicted_relations = set()
        for relation in predicted_relations:
            if relation[2][0] in self.relation_schemas and relation[0] in self.relation_schemas[relation[2][0]]:
                filtered_predicted_relations.add(filter_relation_tuple(relation))

        predicted_relations = filtered_predicted_relations


        correct_relations = predicted_relations & gt_relations

        return predicted_relations, gt_relations, correct_relations

    def evaluate_dataset(self, data_args: DataTrainingArguments, model, device, batch_size: int, macro: bool = False) \
            -> Dict[str, float]:
        """
        Evaluate model on this dataset.
        """
        results = Counter()

        for example, trigger_output_sentence in self.generate_output_sentences(data_args, model, device, batch_size):

            trigger_output_format = self.output_format
            predicted_triggers = \
                trigger_output_format.run_inference(
                    example,
                    trigger_output_sentence,
                    entity_types=self.entity_types,
                    relation_types=self.relation_types,
                )[0]
            gt_triggers = set(trigger.to_tuple() for trigger in example.triggers)
            correct_triggers = predicted_triggers & gt_triggers
            predicted_triggers_notype = set()
            gt_triggers_notype = set()

            for trig in predicted_triggers:
                trig_list = list(trig)
                trig_list[0] = 'TYPE'
                predicted_triggers_notype.add(tuple(trig_list))
            for trig in gt_triggers:
                trig_list = list(trig)
                trig_list[0] = 'TYPE'
                gt_triggers_notype.add(tuple(trig_list))
            correct_triggers_notype = predicted_triggers_notype & gt_triggers_notype


            all_gt_relations, all_predicted_relations, all_correct_relations = set(), set(), set()
            for trigger in predicted_triggers:
                example_argument_single_trigger = copy.deepcopy(example)
                trigger_type = None
                for trigger_type in self.entity_types:
                    if self.entity_types[trigger_type].natural == trigger[0]: break
                example_argument_single_trigger.triggers = [
                    Entity(type=self.entity_types[trigger_type], start=trigger[1], end=trigger[2])]

                argument_input_format = INPUT_FORMATS[self.argument_input_format]()
                argument_output_format = OUTPUT_FORMATS[self.argument_output_format]()
                example_input = argument_input_format.format_input(example_argument_single_trigger, multitask=True,
                                                                   task_descriptor=ACE2005EventArgumentDataset.name)
                example_input_ids = self.tokenizer.batch_encode_plus(
                    [example_input],
                    max_length=data_args.max_seq_length,
                    return_tensors='pt',
                    padding='max_length',
                    truncation=True
                )
                argument_output = model.generate(
                    example_input_ids['input_ids'].to(device),
                    max_length=data_args.max_output_seq_length_eval,
                    num_beams=data_args.num_beams,
                )[0]
                argument_output_sentence = self.tokenizer.decode(argument_output, skip_special_tokens=True,
                                                                 clean_up_tokenization_spaces=False)

                gt_relations, predicted_relations, correct_relations = \
                    self.evaluate_argument(argument_output_format, example_argument_single_trigger, example,
                                           argument_output_sentence)
                all_gt_relations = all_gt_relations.union(gt_relations)
                all_predicted_relations = all_predicted_relations.union(predicted_relations)
                all_correct_relations = all_correct_relations.union(correct_relations)

            all_predicted_relations_notype = set()
            all_gt_relations_notype = set()
            for rel in all_predicted_relations:
                rel_list = list(rel)
                rel_list[0] = 'TYPE'
                all_predicted_relations_notype.add(tuple(rel_list))
            for rel in all_gt_relations:
                rel_list = list(rel)
                rel_list[0] = 'TYPE'
                all_gt_relations_notype.add(tuple(rel_list))

            all_correct_relations_notype = all_predicted_relations_notype & all_gt_relations_notype
            res = Counter({
                'num_sentences': 1,
                'gt_triggers': len(gt_triggers),
                'predicted_triggers': len(predicted_triggers),
                'correct_triggers': len(correct_triggers),
                'correct_triggers_notype': len(correct_triggers_notype),
                'predicted_relations': len(all_predicted_relations),
                'gt_relations': len(all_gt_relations),
                'correct_relations': len(all_correct_relations),
                'correct_relations_notype': len(all_correct_relations_notype)
            })

            results += res

        trigger_precision, trigger_recall, trigger_f1 = get_precision_recall_f1(
            num_correct=results['correct_triggers'],
            num_predicted=results['predicted_triggers'],
            num_gt=results['gt_triggers'],
        )
        trigger_precision_notype, trigger_recall_notype, trigger_f1_notype = get_precision_recall_f1(
            num_correct=results['correct_triggers_notype'],
            num_predicted=results['predicted_triggers'],
            num_gt=results['gt_triggers'],
        )
        relation_precision, relation_recall, relation_f1 = get_precision_recall_f1(
            num_correct=results['correct_relations'],
            num_predicted=results['predicted_relations'],
            num_gt=results['gt_relations'],
        )
        relation_precision_notype, relation_recall_notype, relation_f1_notype = get_precision_recall_f1(
            num_correct=results['correct_relations_notype'],
            num_predicted=results['predicted_relations'],
            num_gt=results['gt_relations'],
        )

        full_results = {
            'relation_precision': relation_precision,
            'relation_recall': relation_recall,
            'relation_f1': relation_f1,
            'relation_precision_notype': relation_precision_notype,
            'relation_recall_notype': relation_recall_notype,
            'relation_f1_notype': relation_f1_notype,
            'trigger_precision': trigger_precision,
            'trigger_recall': trigger_recall,
            'trigger_f1': trigger_f1,
            'trigger_precision_notype': trigger_precision_notype,
            'trigger_recall_notype': trigger_recall_notype,
            'trigger_f1_notype': trigger_f1_notype,
        }

        return full_results


@register_dataset
class CoNLL12CorefDataset(NERDataset):
    """
    CoNLL2012 dataset (coreference resolution).
    """
    name = 'conll12_coref'
    default_output_format = 'coref'

    documents = None

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        file_path = os.path.join(self.data_dir(), f'{split} original.jsonlines')

        self.documents = {}
        examples = []

        if self.is_eval:
            chunk_size = self.data_args.chunk_size_eval
            chunk_overlap = self.data_args.chunk_overlap_eval
        else:
            chunk_size = self.data_args.chunk_size
            chunk_overlap = self.data_args.chunk_overlap

        chunk_size = 512

        with open(file_path, 'r') as f:
            for i, l in enumerate(f):
                raw_document = json.loads(l)

                document_id = f'{split}-{i}'

                tokens_data = raw_document['sentences']

                tokens = []
                for each in tokens_data:
                    for each_word in each:
                        tokens.append(each_word)
                tokens_start_char = [each[0] for each in raw_document['constituents']]
                tokens_end_char = [each[1] for each in raw_document['constituents']]


                groups = []
                for raw_group in raw_document['clusters']:
                    mentions = []
                    for raw_mention in raw_group:
                        mentions.append(Entity(start=raw_mention[0], end=raw_mention[1]+1))
                    groups.append(mentions)


                chunks = []
                pos = 0
                chunk_id = 0
                while pos < len(tokens):

                    chunk_tokens = tokens[pos:pos+chunk_size]

                    chunk_groups = []
                    for group in groups:
                        mentions = [
                            Entity(start=mention.start-pos, end=mention.end-pos, type=mention.type)
                            for mention in group
                            if mention.start >= pos and mention.end <= pos + chunk_size
                        ]
                        if len(mentions) >= 2:
                            chunk_groups.append(mentions)

                    example = InputExample(
                        id=f'{split}-{i}-{chunk_id}',
                        tokens=chunk_tokens,
                        offset=pos,
                        groups=chunk_groups,
                        document_id=document_id,
                        chunk_id=chunk_id,
                    )
                    if len(chunk_groups)>1 or split=='test':
                        examples.append(example)
                    chunks.append(example)

                    if pos + chunk_size >= len(tokens):

                        break

                    pos += chunk_size - chunk_overlap
                    chunk_id += 1

                self.documents[document_id] = CorefDocument(
                    id=document_id,
                    tokens=tokens,
                    groups=groups,
                    chunks=chunks,
                    chunk_centers=[example.offset + len(example.tokens) // 2 for example in chunks]
                )

        logging.info(f"Loaded {len(self.documents)} documents split in {len(examples)} chunks"
                     f" for split {split} of {self.name}")

        return examples

    @staticmethod
    def get_document_predictions(chunk_data: List[List[tuple]]) -> List[List[Tuple[int, int]]]:
        """
        Aggregate predictions for each chunk into document-level predictions.
        """
        all_edges = set(x for l in chunk_data for x in l)

        graph = nx.Graph()
        graph.add_edges_from(all_edges)

        processed_groups = []
        for component in nx.connected_components(graph):
            processed_group = []
            for start, end in sorted(component, key=lambda x: (x[0], -x[1])):

                if len(processed_group) == 0 or start >= processed_group[-1][1]:
                    processed_group.append((start, end))

            processed_groups.append(processed_group)

        return [[(start, end) for start, end in group] for group in processed_groups]

    def evaluate_dataset(self, data_args, model, device, batch_size=8, macro=False, by_relation_type=False, mode='default', external=None) \
            -> Dict[str, float]:
        """
        Evaluate model on this dataset.
        """
        documents_to_chunk_data = defaultdict(list)
        predictions = {}

        true_groups = []; pred_groups = []
        for example, output_sentence in (
            self.generate_output_sentences(data_args, model, device, batch_size)
            if external is None else self.generate_external_pairs(external)
        ):
            document_id = example.document_id

            true_group, pred_group = self.output_format.run_inference(
                example=example,
                output_sentence=output_sentence,
                mode=mode,
            )
            true_groups.append(true_group)
            pred_groups.append(pred_group)

        metrics = CorefAllMetrics().get_all_metrics(true_groups, pred_groups)
        return {
            f'{metric_name}_{x}': v
            for metric_name, metric_values in metrics['micro'].items()
            for x, v in metric_values.items()
        }


class RelationClassificationDataset(JointERDataset):
    """
    Base class for relation classification datasets, implementing NLL inference.
    """

    num_episodes = 1

    def preprocess_for_glm_single(self, examples, mode, dataset):
        source = []; target = []
        for example in examples:
            source.append(self.output_format.SOURCE_FORMATS['rc'].format(dataset.split('_')[0],' '.join(example.tokens),TASK_MAPPING[dataset]))
            target.append(self.output_format.format_output(example,mode))
        return source, target

    def nll_inference(self, example: InputExample, relation_types: List[RelationType], model=None,
                      tokenizer=None) -> RelationType:
        """
        Run inference on a single example of this dataset, searching for the relation which maximizes the likelihood
        of the corresponding output sentence.
        """
        formatted_input = [self.input_format.format_input(example)]
        scores = []

        x = tokenizer.batch_encode_plus(
            formatted_input,
            max_length=self.max_input_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )

        for relation_type in relation_types:

            candidate_example = copy.deepcopy(example)
            candidate_example.relations[0].type = relation_type


            formatted_output = [self.output_format.format_output(candidate_example)]
            y = tokenizer.batch_encode_plus(
                formatted_output,
                max_length=self.max_output_length,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
            )


            res = model(
                input_ids=x['input_ids'].to(model.device),
                attention_mask=x['attention_mask'].to(model.device),
                labels=y['input_ids'].to(model.device)
            )
            scores.append(res[0].cpu().detach())

        scores = np.array(scores)
        min_idx = scores.argmin()

        return relation_types[min_idx]

    def load_data(self, mode: str, seed: int = None, glm: bool = False) -> List[InputExample]:
        """
        Load all data, where 'mode' is a list of comma-separated splits to use.
        """
        examples = []

        if isinstance(mode, str):
            splits = mode.split(',')
        else:
            assert isinstance(mode, (list, tuple))
            splits = mode


        if glm:
            for split in splits:
                for i in range(self.num_episodes):
                    examples += self.load_data_single_split(split, seed=i)
        else:
            for split in splits:
                examples += self.load_data_single_split(split, seed=seed)

        return examples

    def evaluate_example(self, example: InputExample, output_sentence: str, model=None, tokenizer=None, mode='default') -> Counter:
        """
        Evaluate an output sentence on a single example of this dataset.
        """

        predicted_relations, true_relations, wrong_reconstruction, format_error = self.output_format.run_inference(
            example,
            output_sentence,
            entity_types=self.entity_types,
            relation_types=self.relation_types,
            mode = mode,
        )


        correct_relations = predicted_relations & true_relations

        res = Counter({
            'num_sentences': 1,
            'wrong_reconstructions': 1 if wrong_reconstruction else 0,
            'format_error': 1 if format_error else 0,
            'true_relations': len(true_relations),
            'predicted_relations': len(predicted_relations),
            'correct_relations': len(correct_relations),
        })

        return res

    def _evaluate_dataset_calculate_results(self, results):

        relation_precision, relation_recall, relation_f1 = get_precision_recall_f1(
            num_correct=results['correct_relations'],
            num_predicted=results['predicted_relations'],
            num_gt=results['true_relations'],
        )

        res = {
            'wrong_reconstruction': results['wrong_reconstructions'] / results['num_sentences'],
            'format_error': results['format_error'] / results['num_sentences'],
            'relation_precision': relation_precision,
            'relation_recall': relation_recall,
            'relation_f1': relation_f1,
        }

        return res

    def evaluate_dataset(self, data_args: DataTrainingArguments, model, device, batch_size: int, macro: bool = False, mode: str = 'default', external: str = None) \
            -> Dict[str, float]:
        results = Counter()

        for example, output_sentence in (
            self.generate_output_sentences(data_args, model, device, batch_size)
            if external is None else self.generate_external_pairs(external)
        ):
            new_result = self.evaluate_example(
                    example=example,
                    output_sentence=output_sentence,
                    model=model,
                    tokenizer=self.tokenizer,
                    mode=mode
            )
            results += new_result

        return self._evaluate_dataset_calculate_results(results)



@register_dataset
class FewRelFull(RelationClassificationDataset):
    """
    Full FewRel dataset (relation classification), not episodic.

    Data was downloaded from https://github.com/thunlp/FewRel/tree/master/data
    """
    name = 'FewRel'
    data_name = 'FewRel'

    natural_entity_types = {
        'head': 'head',
        'tail': 'tail',
    }

    def load_schema(self):
        """
        Load relation types from the pid2name.json file provided with the dataset.
        """
        super().load_schema()

        with open(os.path.join(self.data_dir(), 'pid2name.json'), 'r') as f:
            data = json.load(f)
            self.relation_types = {
                short: RelationType(short=short, natural=description[0])
                for short, description in data.items()
            }

    def load_data_by_relation_type(self, split: str) -> Dict[str, List[InputExample]]:
        """
        Load data for a single split (train or dev) by relation type.

        This is useful for episodic training/evaluation, where we sample N classes at each episode.
        """
        examples_by_type = {}
        file_path = os.path.join(self.data_dir(), f'{split}.json')

        with open(file_path, 'r') as f:
            data = json.load(f)
            i = 0

            for type_id in data:
                assert type_id in self.relation_types
                relation_type = self.relation_types[type_id]
                examples = []

                for idx, _data in enumerate(data[type_id]):
                    tokens = _data['tokens']
                    head_entity = _data['h'][2][0]
                    tail_entity = _data['t'][2][0]

                    if len(head_entity) == 1:
                        head_entity = [head_entity[0], head_entity[0]]

                    if len(tail_entity) == 1:
                        tail_entity = [tail_entity[0], tail_entity[0]]

                    head_entity = Entity(id=None, type=self.entity_types['head'],
                                         start=head_entity[0], end=head_entity[1] + 1)

                    tail_entity = Entity(id=None, type=self.entity_types['tail'],
                                         start=tail_entity[0], end=tail_entity[1] + 1)

                    entities = [head_entity, tail_entity]

                    relations = [
                        Relation(
                            type=relation_type, head=head_entity, tail=tail_entity
                        )
                    ]

                    example = InputExample(
                        id=f'{split}-{i}',
                        tokens=tokens,
                        entities=entities,
                        relations=relations,
                    )
                    examples.append(example)
                    i += 1

                examples_by_type[type_id] = examples

        return examples_by_type

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        examples_by_type = self.load_data_by_relation_type(split=split)
        examples = [example for x in examples_by_type.values() for example in x]
        return examples


@register_dataset
class FewRelEpisodic(FewRelFull):
    """
    Full FewRel dataset (relation classification), episodic.

    Episodic fine-tuning should happen after meta-training on the FewRelFull dataset.
    """
    name = 'FewRelEpisodic'
    data_name = 'FewRelEpisodic'

    default_input_format = 'rc_input'
    default_output_format = 'rc_output'

    way = 10
    shot = 5
    epoch_per_divide = 500

    target_relation_types = None
    def glm_load_data_single_split(self, split: str, seed: int = 0) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        examples_by_type = self.load_data_by_relation_type(split='dev')


        num_ways, num_shots, num_queries = self.data_args.num_ways, self.data_args.num_shots, self.data_args.num_query

        self.seed = seed
        random.seed(seed)


        self.target_relation_types = {
            type_id: self.relation_types[type_id]
            for type_id in random.sample(list(examples_by_type.keys()), num_ways)
        }

        logging.info(f'Target relation types for this few-shot episode: '
                     f'{[relation.natural for relation in self.target_relation_types.values()]}')

        support = []
        query = []


        for i, type_id in enumerate(self.target_relation_types):
            random.seed(seed + i)
            sampled_examples = random.sample(examples_by_type[type_id], num_shots + num_queries)
            support += sampled_examples[:num_shots]
            query += sampled_examples[num_shots:]

        return support, query


    def evaluate_example(self, example: InputExample, output_sentence: str, model=None, tokenizer=None) -> Counter:
        """
        Evaluate a single example of this dataset, using NLL inference.
        """
        predicted_relation = self.nll_inference(
            example=example,
            relation_types=list(self.target_relation_types.values()),
            model=model,
            tokenizer=tokenizer,
        )

        predicted_relations = {predicted_relation}
        gt_relations = set(relation.type for relation in example.relations)
        correct_relations = gt_relations & predicted_relations

        return Counter({
            'num_sentences': 1,
            'gt_relations': len(gt_relations),
            'predicted_relations': len(predicted_relations),
            'correct_relations': len(correct_relations),
        })

    def preprocess_for_glm(self, mode, dataset, fewshot=-1, debug=False):
        dataset_name = self.name; DATA_DIR = self.data_dir()+"/"

        '''
        finetune the model on the support set to learn the new task and evaluate on the query set
        '''

        support, query = self.glm_load_data_single_split('dev')

        support = support * self.epoch_per_divide

        source,target = self.preprocess_for_glm_single(support, mode, dataset)

        print("self.seed: ", self.seed)

        with open(DATA_DIR+"train.source"+"_"+str(self.way)+'_'+str(self.shot)+'_'+str(self.seed),"w") as f:
            for line in source:
                print(line, file=f)
        with open(DATA_DIR+"train.target"+"_"+str(self.way)+'_'+str(self.shot)+'_'+str(self.seed),"w") as f:
            for line in target:
                print(line, file=f)

        source,target = self.preprocess_for_glm_single(query, mode, dataset)

        with open(DATA_DIR+"val.source"+"_"+str(self.way)+'_'+str(self.shot)+'_'+str(self.seed),"w") as f:
            for line in source:
                print(line, file=f)
        with open(DATA_DIR+"val.target"+"_"+str(self.way)+'_'+str(self.shot)+'_'+str(self.seed),"w") as f:
            for line in target:
                print(line, file=f)

        with open(DATA_DIR+"test.source"+"_"+str(self.way)+'_'+str(self.shot)+'_'+str(self.seed),"w") as f:
            for line in source:
                print(line, file=f)
        with open(DATA_DIR+"test.target"+"_"+str(self.way)+'_'+str(self.shot)+'_'+str(self.seed),"w") as f:
            for line in target:
                print(line, file=f)

@register_dataset
class TACRED(RelationClassificationDataset):
    name = 'tacred'

    NO_RELATION = 'no relation'


    @staticmethod
    def to_natural(t: str) -> str:
        """
        Convert entity or relation type to a natural text.
        """
        t = t.split(":")
        assert len(t) <= 2, "Unexpected format {}".format(t)
        t = t[1] if len(t) == 2 else t[0]
        t = t.lower()
        t = t.replace("_", " ")
        t = t.replace("/", " ")
        t = t.replace("stateorprovince", "state or province")

        return t

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        file_path = os.path.join(self.data_dir(), f'json/{split}.json')
        examples = []


        self.entity_types = {}
        self.relation_types = {}

        with open(file_path, 'r') as f:
            data = json.load(f)
            logging.info(f"Loaded {len(data)} sentences for split {split} of {self.name}")
            i = 0
            for idx, obj in enumerate(data):
                words = obj['token']
                head_start, head_end, head_type = obj['subj_start'], obj['subj_end'] + 1, obj['subj_type']
                tail_start, tail_end, tail_type = obj['obj_start'], obj['obj_end'] + 1, obj['obj_type']
                relation = obj['relation']

                if head_type not in self.entity_types:
                    self.entity_types[head_type] = EntityType(short=head_type, natural=self.to_natural(head_type))
                if tail_type not in self.entity_types:
                    self.entity_types[tail_type] = EntityType(short=tail_type, natural=self.to_natural(tail_type))

                head_entity = Entity(
                    id=None,
                    type=self.entity_types[head_type],
                    start=head_start,
                    end=head_end
                )
                tail_entity = Entity(
                    id=None,
                    type=self.entity_types[tail_type],
                    start=tail_start,
                    end=tail_end
                )

                entities = [
                    head_entity, tail_entity
                ]

                if relation not in self.relation_types:
                    self.relation_types[relation] = RelationType(short=relation, natural=self.to_natural(relation))

                relations = [
                    Relation(type=self.relation_types[relation], head=head_entity, tail=tail_entity)
                ]

                example = InputExample(
                    id=f'{split}-{i}',
                    tokens=words,
                    entities=entities,
                    relations=relations,
                )
                i += 1
                examples.append(example)

        self.relation_types = {
            relation.type.short: relation.type
            for example in examples for relation in example.relations
        }

        return examples
    def evaluate_example(self, example: InputExample, output_sentence: str, model=None, tokenizer=None, mode='default') -> Counter:
        """
        Evaluate an output sentence on a single example of this dataset.
        """

        predicted_relations, true_relations, wrong_reconstruction, format_error = self.output_format.run_inference(
            example,
            output_sentence,
            entity_types=self.entity_types,
            relation_types=self.relation_types,
            mode = mode,
        )


        correct_relations = predicted_relations & true_relations

        try:
            if list(true_relations)[0] == self.NO_RELATION and list(predicted_relations)[0] == self.NO_RELATION:
                return Counter({
                'num_sentences': 1,
                'wrong_reconstructions': 0,
                'format_error': 0,
                'true_relations': 1,
                'predicted_relations': 1,
                'correct_relations': 1,
                })

            elif list(true_relations)[0] == self.NO_RELATION:
                return Counter({
                'num_sentences': 1,
                'wrong_reconstructions': 0,
                'format_error': 0,
                'true_relations': 1,
                'predicted_relations': 1,
                'correct_relations': 0,
                })
            elif list(predicted_relations)[0] == self.NO_RELATION:
                return Counter({
                'num_sentences': 1,
                'wrong_reconstructions': 0,
                'format_error': 0,
                'true_relations': 1,
                'predicted_relations': 1,
                'correct_relations': 0,
                })
            else:
                return Counter({
                'num_sentences': 1,
                'wrong_reconstructions': 1 if wrong_reconstruction else 0,
                'format_error': 1 if format_error else 0,
                'true_relations': 1,
                'predicted_relations': 1,
                'correct_relations': 1 if len(correct_relations) == 1 else 0,
                })

        except:
            return Counter({
                'num_sentences': 1,
                'wrong_reconstructions': 1,
                'format_error': 1,
                'true_relations': 1,
                'predicted_relations': 1,
                'correct_relations': 0,
                })


    def _evaluate_dataset_calculate_results(self, results):

        relation_precision, relation_recall, relation_f1 = get_precision_recall_f1(
            num_correct=results['correct_relations'],
            num_predicted=results['predicted_relations'],
            num_gt=results['true_relations'],
        )

        res = {
            'wrong_reconstruction': results['wrong_reconstructions'] / results['num_sentences'],
            'format_error': results['format_error'] / results['num_sentences'],
            'relation_precision': relation_precision,
            'relation_recall': relation_recall,
            'relation_f1': relation_f1,
        }

        return res

    def evaluate_dataset(self, data_args: DataTrainingArguments, model, device, batch_size: int, macro: bool = False, mode: str = 'v0.3.0', external: str = None) \
            -> Dict[str, float]:
        results = Counter()

        for example, output_sentence in (
            self.generate_output_sentences(data_args, model, device, batch_size)
            if external is None else self.generate_external_pairs(external)
        ):
            new_result = self.evaluate_example(
                    example=example,
                    output_sentence=output_sentence,
                    model=model,
                    tokenizer=self.tokenizer,
                    mode=mode
            )
            results += new_result

        return self._evaluate_dataset_calculate_results(results)



@register_dataset
class CONLL05SRL(NERDataset):
    name = 'conll05_srl'
    natural_entity_types = {
        'V': 'verb',
        'A0': 'first argument',
        'A1': 'second argument',
        'A2': 'third argument',
        'A3': 'fourth argument',
        'AM-MOD': 'modal',
        'AM-NEG': 'negation',
    }

    default_input_format = 'srl_input'

    def convert_bio_to_entities(self, bio_tag: List[str]) -> Tuple[List[Entity], Entity]:
        entities = []
        current_entity = None
        predicate = None
        for ii, el in enumerate(bio_tag):
            if el.startswith('B-'):
                tag_type = el[2:]
                if tag_type in self.natural_entity_types:
                    current_entity = Entity(
                        type=EntityType(
                            short=tag_type,
                            natural=self.natural_entity_types[tag_type]
                        ),
                        start=ii,
                        end=ii+1,
                    )
                    if tag_type == 'V':
                        predicate = current_entity
                    else:
                        entities.append(current_entity)
                else:
                    current_entity = None
            elif el.startswith('I-'):
                if current_entity is not None:
                    current_entity.end = ii + 1
        return entities, predicate

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        file_path = os.path.join(self.data_dir(), f'conll05.{split}.txt')
        examples = []
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                line = line.split('|||')

                sentence, tag = line[0].strip(), line[1].strip()
                sentence = sentence.split()[1:]
                tag = tag.split()
                arguments, predicate = self.convert_bio_to_entities(tag)
                for argument in arguments:
                    argument.start += argument.start>=predicate.end
                    argument.end += argument.end>predicate.end
                sentence[predicate.end  :predicate.end  ] = [']']
                for argument in arguments:
                    argument.start += argument.start>=predicate.start
                    argument.end += argument.end>predicate.start
                sentence[predicate.start:predicate.start] = ['[']
                predicate.start += 1
                predicate.end += 1
                example = InputExample(
                    id=f'{split}-{i}',
                    tokens=sentence,
                    entities=arguments,
                    relations=[],
                    sentence_level_entities=[predicate]
                )
                examples.append(example)

        self.entity_types = {
            entity.type.natural: entity.type
            for example in examples for entity in example.entities
        }

        return examples


@register_dataset
class CONLL12SRL(CONLL05SRL):
    name = 'conll12_srl'
    natural_entity_types = {
        'V': 'verb',
        'ARG0': 'first argument',
        'ARG1': 'second argument',
        'ARG2': 'third argument',
        'ARG3': 'fourth argument',
        'ARG4': 'fifth argument',
        'ARG5': 'sixth argument',
        'ARGM-MOD': 'modal',
        'ARGM-NEG': 'negation',
        'ARGM-ADV': 'general-purpose',
        'ARGM-CAU': 'cause',
        'ARGM-DIR': 'direction',
        'ARGM-DIS': 'discourse marker',
        'ARGM-EXT': 'extent',
        'ARGM-LOC': 'location',
        'ARGM-MNR': 'manner',
        'ARGM-PNC': 'purpose',
        'ARGM-PRD': 'predication',
        'ARGM-REC': 'reciprocal',
        'ARGM-TMP': 'temporal',
        'R-ARG0': 'reference to first argument',
        'R-ARG1': 'reference to second argument',
        'R-ARG2': 'reference to third argument',
        'R-ARG3': 'reference to fourth argument',
        'R-ARG4': 'reference to fifth argument',
        'R-ARG5': 'reference to sixth argument',
        'R-ARGM-MOD': 'reference to modal',
        'R-ARGM-NEG': 'reference to negation',
        'R-ARGM-ADV': 'reference to general-purpose',
        'R-ARGM-CAU': 'reference to cause',
        'R-ARGM-DIR': 'reference to direction',
        'R-ARGM-DIS': 'reference to discourse marker',
        'R-ARGM-EXT': 'reference to extent',
        'R-ARGM-LOC': 'reference to location',
        'R-ARGM-MNR': 'reference to manner',
        'R-ARGM-PNC': 'reference to purpose',
        'R-ARGM-PRD': 'reference to predication',
        'R-ARGM-REC': 'reference to reciprocal',
        'R-ARGM-TMP': 'reference to temporal',
    }

    def convert_bio_to_entities(self, bio_tag: List[str]) -> Tuple[List[Entity], Entity]:
        entities = []
        current_entity = None
        predicate = None
        for ii, el in enumerate(bio_tag):
            if el.startswith('B-'):
                tag_type = el[2:]
                if tag_type in self.natural_entity_types:
                    current_entity = Entity(
                        type=EntityType(
                            short=tag_type,
                            natural=self.natural_entity_types[tag_type]
                        ),
                        start=ii,
                        end=ii+1,
                    )
                    if tag_type == 'V':
                        predicate = current_entity
                    else:
                        entities.append(current_entity)
                else:
                    current_entity = None

            elif el.startswith('I-'):
                if current_entity is not None:
                    current_entity.end = ii + 1
        return entities, predicate

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        file_path = os.path.join(self.data_dir(), f'conll2012.{split}.txt')
        examples = []
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                line = line.split('|||')

                sentence, tag = line[0].strip(), line[1].strip()
                sentence = sentence.split()[1:]
                tag = tag.split()
                arguments, predicate = self.convert_bio_to_entities(tag)
                for argument in arguments:
                    argument.start += argument.start>=predicate.end
                    argument.end += argument.end>predicate.end
                sentence[predicate.end  :predicate.end  ] = [']']
                for argument in arguments:
                    argument.start += argument.start>=predicate.start
                    argument.end += argument.end>predicate.start
                sentence[predicate.start:predicate.start] = ['[']
                predicate.start += 1
                predicate.end += 1
                example = InputExample(
                    id=f'{split}-{i}',
                    tokens=sentence,
                    entities=arguments,
                    relations=[],
                    sentence_level_entities=[predicate]
                )
                examples.append(example)

        self.entity_types = {
            entity.type.natural: entity.type
            for example in examples for entity in example.entities
        }


















        return examples


@register_dataset
class CONLL05SRLBrown(CONLL05SRL):
    name = 'conll05_srl_brown'

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        if split!='test':
            tmp = self.data_name; self.data_name = "conll05_srl"
            result = super(CONLL05SRLBrown, self).load_data_single_split(split, seed)
            self.data_name = tmp; return result
        file_path = os.path.join(self.data_dir(), f'conll05.{split}.txt')
        examples = []
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                line = line.split('|||')

                sentence, tag = line[0].strip(), line[1].strip()
                sentence = sentence.split()[1:]
                tag = tag.split()
                arguments, predicate = self.convert_bio_to_entities(tag)
                for argument in arguments:
                    argument.start += argument.start>=predicate.end
                    argument.end += argument.end>predicate.end
                sentence[predicate.end  :predicate.end  ] = [']']
                for argument in arguments:
                    argument.start += argument.start>=predicate.start
                    argument.end += argument.end>predicate.start
                sentence[predicate.start:predicate.start] = ['[']
                example = InputExample(
                    id=f'{split}-{i}',
                    tokens=sentence,
                    entities=arguments,
                    relations=[],
                    sentence_level_entities=[predicate]
                )
                examples.append(example)

        self.entity_types = {
            entity.type.natural: entity.type
            for example in examples for entity in example.entities
        }


















        return examples

@register_dataset
class CONLL05SRLWSJ(CONLL05SRLBrown):
    name = 'conll05_srl_wsj'

@register_dataset
class MultiWoz(BaseDataset):
    """
    MultiWoz dataset (Dialogue State Tracking).

    Data was downloaded from https://github.com/budzianowski/multiwoz/blob/master/data/MultiWOZ_2.1.zip, and the
    pre-processing script from https://github.com/jasonwu0731/trade-dst was used, as suggested in the official
    MultiWoz repository https://github.com/budzianowski/multiwoz.
    """
    name = 'multi_woz'
    data_name = 'multi_woz'
    default_input_format = 'plain'
    default_output_format = 'multi_woz'

    num_episodes = 1

    def truncate_first_n_tokens(self, examples, max_seq_length, delimiter='▁'):
        output = []
        for x in examples:
            tokens = self.tokenizer.tokenize(x)
            if len(tokens) > max_seq_length:
                x = ''.join(tokens[-1 * max_seq_length + 1:]).replace(delimiter, ' ')
                assert self.tokenizer.tokenize(''.join(tokens).replace(delimiter, ' ')) == tokens
            output.append(x)
        return output

    def load_data(self, mode: str, seed: int = None, glm: bool = False) -> List[InputExample]:
        """
        Load all data, where 'mode' is a list of comma-separated splits to use.
        """
        examples = []

        if isinstance(mode, str):
            splits = mode.split(',')
        else:
            assert isinstance(mode, (list, tuple))
            splits = mode


        if glm:
            for split in splits:
                for i in range(self.num_episodes):
                    examples += self.load_data_single_split(split, seed=i)
        else:
            for split in splits:
                examples += self.load_data_single_split(split, seed=seed)

        return examples

    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        examples = []
        file_path = os.path.join(self.data_dir(), f'multi_woz_2.1_{split}_5_domain.json')

        with open(file_path, 'r') as f:
            data = json.load(f)
            num_examples = len(data["examples"])
            logging.info(f"Loaded {num_examples} sentences for split {split} of {self.name}")

            for i, x in enumerate(data["examples"]):
                turn_id = x["turn_id"]
                conv_id = x["ID"]
                dialog = x["dialog_history"]
                tokens = dialog.split(" ")
                belief = x["turn_belief"]
                uid = "{0}-{1}".format(turn_id, conv_id)

                example = InputExample(
                    id=uid,
                    tokens=tokens,
                    belief_state=belief,
                    utterance_tokens=x["turn_uttr"],
                )
                examples.append(example)
        return examples

    def compute_features(self, max_input_length: int, max_output_length: int, multitask: bool = False):
        input_sentences = [self.input_format.format_input(example, multitask=multitask) for example in self.examples]
        output_sentences = [self.output_format.format_output(example) for example in self.examples]

        input_sentences = self.truncate_first_n_tokens(examples=input_sentences,
                                                       max_seq_length=max_input_length)
        output_sentences = self.truncate_first_n_tokens(examples=output_sentences,
                                                        max_seq_length=max_output_length)

        input_tok = self.tokenizer.batch_encode_plus(
            input_sentences,
            max_length=max_input_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )
        output_tok = self.tokenizer.batch_encode_plus(
            output_sentences,
            max_length=max_output_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )
        self._warn_max_sequence_length(max_input_length, input_sentences, "input")
        self._warn_max_sequence_length(max_output_length, output_sentences, "output")

        assert input_tok.input_ids.size(0) == output_tok.input_ids.size(0), print(
            f'Size does not match: len(sentences_tok.input_ids)={len(input_tok.input_ids)}, '
            f'len(labels_tok.input_ids)={len(output_tok.input_ids)}'
        )
        features = []
        for sentence_input_ids, att_mask, label_input_ids in zip(input_tok.input_ids, input_tok.attention_mask,
                                                                 output_tok.input_ids):
            features.append(InputFeatures(
                input_ids=sentence_input_ids.tolist(),
                attention_mask=att_mask.tolist(),
                label_ids=label_input_ids.tolist()
            ))

        return features

    def evaluate(self, example: InputExample, output_sentence: str):
        """
        Evaluate a single example.
        """
        gold_belief_set = set(example.belief_state)
        pred_belief_set = \
            self.output_format.run_inference(
                example,
                output_sentence,
            )


        pred_belief_set = {elem.replace("-", " ") for elem in pred_belief_set}
        gold_belief_set = {elem.replace("-", " ") for elem in gold_belief_set}
        correct_belief_state = gold_belief_set == pred_belief_set

        return {
            'num_sentences': 1,
            'correct_state': int(correct_belief_state),
            'raw_gold_state': None,
            'raw_pred_state': output_sentence,
            'list_pred_state': list(pred_belief_set),
            'list_gold_state': list(gold_belief_set)
        }

    def evaluate_dataset(self, data_args: DataTrainingArguments, model, device, batch_size: int, macro: bool = False) \
            -> Dict[str, float]:
        """
        Evaluate model on this dataset.
        """

        def compute_accuracy(results_dict):
            num_examples = float(sum(results_dict["num_sentences"]))
            return sum(results_dict["correct_state"]) / num_examples

        results = {
            'num_sentences': [],
            'correct_state': [],
            'raw_gold_state': [],
            'raw_pred_state': [],
            'list_pred_state': [],
            'list_gold_state': []
        }

        for example, output_sentence in self.generate_output_sentences(data_args, model, device, batch_size):
            new_result = self.evaluate(
                example=example,
                output_sentence=output_sentence,
            )

            for k, v in new_result.items():
                results[k].append(v)

        return {
            'joint_accuracy': compute_accuracy(results),
        }


    def preprocess_for_glm(self, mode, dataset, fewshot=-1, debug=False):
        dataset_name = self.name; DATA_DIR = self.data_dir()+"/"
        source,target = self.preprocess_for_glm_single(self.load_data_single_split('train'), mode, dataset)
        S, T = self.preprocess_for_glm_single(self.load_data_single_split('dev'), mode, dataset)
        source.extend(S); target.extend(T)
        ind = [i for i in range(len(source))]
        if fewshot>0:
            np.random.seed(0)
            np.random.shuffle(ind)
            ind = ind[:fewshot]
        source, target = [source[i] for i in ind], [target[i] for i in ind]
        with open(DATA_DIR+"train.source","w") as f:
            for line in source:
                print((self.prompt if mode=="fewshot" else "")+line, file=f)
        with open(DATA_DIR+"train.target","w") as f:
            for line in target:
                print(line, file=f)

        source,target = self.preprocess_for_glm_single(self.load_data_single_split('dev'), mode, dataset)
        ind = [i for i in range(len(source))]
        if debug:
            np.random.seed(0)
            np.random.shuffle(ind)
            ind = ind[:8]
        source, target = [source[i] for i in ind], [target[i] for i in ind]
        with open(DATA_DIR+"val.source","w") as f:
            for line in source:
                print((self.prompt if mode=="fewshot" else "")+line, file=f)
        with open(DATA_DIR+"val.target","w") as f:
            for line in target:
                print(line, file=f)

        source,target = self.preprocess_for_glm_single(self.load_data_single_split('test'), mode, dataset)
        ind = [i for i in range(len(source))]
        if debug:
            np.random.seed(0)
            np.random.shuffle(ind)
            ind = ind[:8]
        source, target = [source[i] for i in ind], [target[i] for i in ind]
        with open(DATA_DIR+"test.source","w") as f:
            for line in source:
                print((self.prompt if mode=="fewshot" else "")+line, file=f)
        with open(DATA_DIR+"test.target","w") as f:
            for line in target:
                print(line, file=f)

    def preprocess_for_glm_single(self, examples, mode, dataset):
        source = []; target = []
        for example in examples:
            source.append("id atis sentence: "+' '.join(example.tokens).replace('\n', '').replace('\r', ''))
            target.append(self.output_format.format_output(example).replace('\n', '').replace('\r', ''))
        return source, target


@register_dataset
class SnipsDataset(NERDataset):
    name = 'snips'

    num_episodes = 1

    default_input_format = 'plain'

    default_output_format = 'intent_detection'

    natural_entity_types = {
        'entity_name': 'entity name',
        'playlist_owner': 'playlist owner',
        'playlist': 'playlist',
        'music_item': 'music item',
        'artist': 'artist',
        'party_size_description': 'party size description',
        'party_size_number': 'party size number',
        'restaurant_type': 'restaurant type',
        'spatial_relation': 'spatial relation',
        'state': 'state',
        'cuisine': 'cuisine',
        'poi': 'poi',
        'country': 'country',
        'city': 'city',
        'timeRange': 'time range',
        'facility': 'facility',
        'served_dish': 'served dish',
        'condition_description': 'condition description',
        'geographic_poi': 'geographic poi',
        'condition_temperature': 'condition temperature',
        'current_location': 'current location',
        'album': 'album',
        'service': 'service',
        'sort': 'sort',
        'track': 'track',
        'year': 'year',
        'object_name': 'object name',
        'rating_value': 'rating value',
        'best_rating': 'best rating',
        'rating_unit': 'rating unit',
        'object_select': 'object select',
        'object_part_of_series_type': 'object part of series type',
        'movie_name': 'movie name',
        'location_name': 'location name',
        'object_location_type': 'object location type',
        'movie_type': 'movie type'
    }

    natural_intent_types = {
        'AddToPlaylist': 'add to playlist',
        'BookRestaurant': 'book restaurant',
        'GetWeather': 'get weather',
        'PlayMusic': 'play music',
        'RateBook': 'rate book',
        'SearchCreativeWork': 'search creative work',
        'SearchScreeningEvent': 'search screening event'
    }

    def convert_bio_to_entities(self, bio_tag: List[str]) -> Tuple[List[Entity], Entity]:
        entities = []
        current_entity = None
        for ii, el in enumerate(bio_tag):
            if el.startswith('B-'):
                tag_type = el[2:]
                current_entity = Entity(
                    type=EntityType(
                        short=tag_type,
                        natural=self.natural_entity_types[tag_type]
                        if tag_type in self.natural_entity_types else tag_type
                    ),
                    start=ii,
                    end=ii+1,
                )
                entities.append(current_entity)
            elif el.startswith('I-'):
                current_entity.end = ii + 1
        return entities


    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        file_path = os.path.join(self.data_dir(), f'{split}/')
        examples = []

        with open(os.path.join(file_path, "seq.in"), 'r') as fin, open(os.path.join(file_path, "seq.out"), 'r') as fout, open(os.path.join(file_path, "label"), mode = 'r', encoding='utf-8') as flabel:
            label_data = flabel.readlines()
            label_data = [x.strip() for x in label_data]
            label_kinds = set(label_data)
            in_data = fin.readlines()
            in_data = [x.strip() for x in in_data]
            out_data = fout.readlines()
            out_data = [x.strip() for x in out_data]

            for id, (utterance, slot_labels, short_intent) in enumerate(zip(in_data, out_data, label_data)):



                tokens = utterance.split()
                slot_labels = slot_labels.split()
                entities = self.convert_bio_to_entities(slot_labels)

                intent = Intent(
                    short=short_intent,
                    natural=self.natural_intent_types[short_intent]
                    if short_intent in self.natural_intent_types.keys() else short_intent
                )

                example = InputExample(
                    id=f'{split}-{id}',
                    tokens=tokens,
                    intent=intent,
                    entities=entities,
                )

                examples.append(example)

        self.entity_types = {
            entity.type.natural: entity.type
            for example in examples for entity in example.entities
        }
        self.intents = {
            example.intent.natural: example.intent
            for example in examples
        }

        return examples


    def evaluate_example(self, example: InputExample, output_sentence: str, model=None, tokenizer=None) -> Counter:
        """
        Evaluate an output sentence on a single example of this dataset.
        """

        res = self.output_format.run_inference(
            example,
            output_sentence,
            entity_types=self.entity_types,
        )
        predicted_intent, predicted_entities, wrong_reconstruction, label_error, format_error = res

        predicted_entities_no_type = set([entity[1:] for entity in predicted_entities])


        gt_entities = set(entity.to_tuple() for entity in example.entities)
        gt_entities_no_type = set([entity[1:] for entity in gt_entities])


        correct_entities = predicted_entities & gt_entities
        correct_entities_no_type = gt_entities_no_type & predicted_entities_no_type


        gt_intent = example.intent




        correct_intent = int(predicted_intent == gt_intent.natural)


        assert len(correct_entities) <= len(predicted_entities)
        assert len(correct_entities) <= len(gt_entities)
        assert len(correct_entities_no_type) <= len(predicted_entities_no_type)
        assert len(correct_entities_no_type) <= len(gt_entities_no_type)

        res = Counter({
            'num_sentences': 1,
            'wrong_reconstructions': 1 if wrong_reconstruction else 0,
            'label_error': 1 if label_error else 0,
            'format_error': 1 if format_error else 0,
            'predicted_intent': 1 if len(predicted_intent) > 0 else 0,
            'gt_intent': 1,
            'correct_intent': correct_intent,
            'gt_entities': len(gt_entities),
            'predicted_entities': len(predicted_entities),
            'correct_entities': len(correct_entities),
            'gt_entities_no_type': len(gt_entities_no_type),
            'predicted_entities_no_type': len(predicted_entities_no_type),
            'correct_entities_no_type': len(correct_entities_no_type),
        })

        if self.intents is not None:
            for intent_type in self.intents.values():
                predicted = int(predicted_intent == intent_type.natural)
                gt = int(gt_intent.natural == intent_type.natural)
                correct = int(predicted_intent == gt_intent.natural)
                res['predicted_intent', intent_type.natural] = predicted
                res['gt_intent', intent_type.natural] = gt
                res['correct_intent', intent_type.natural] = correct


        if self.entity_types is not None:
            for entity_type in self.entity_types.values():
                predicted = set(entity for entity in predicted_entities if entity[0] == entity_type.natural)
                gt = set(entity for entity in gt_entities if entity[0] == entity_type.natural)
                correct = predicted & gt
                res['predicted_entities', entity_type.natural] = len(predicted)
                res['gt_entities', entity_type.natural] = len(gt)
                res['correct_entities', entity_type.natural] = len(correct)

        return res

    def evaluate_dataset(self, data_args: DataTrainingArguments, model, device, batch_size: int, macro: bool = False) \
            -> Dict[str, float]:
        """
        Evaluate model on this dataset.
        """
        results = Counter()

        for example, output_sentence in self.generate_output_sentences(data_args, model, device, batch_size):
            new_result = self.evaluate_example(
                    example=example,
                    output_sentence=output_sentence,
                    model=model,
                    tokenizer=self.tokenizer,
                )
            results += new_result

        entity_precision, entity_recall, entity_f1 = get_precision_recall_f1(
            num_correct=results['correct_entities'],
            num_predicted=results['predicted_entities'],
            num_gt=results['gt_entities'],
        )

        entity_precision_no_type, entity_recall_no_type, entity_f1_no_type = get_precision_recall_f1(
            num_correct=results['correct_entities_no_type'],
            num_predicted=results['predicted_entities_no_type'],
            num_gt=results['gt_entities_no_type'],
        )

        entity_precision_by_type = []
        entity_recall_by_type = []
        entity_f1_by_type = []

        if macro:

            for entity_type in self.entity_types.values():
                precision, recall, f1 = get_precision_recall_f1(
                    num_correct=results['correct_entities', entity_type.natural],
                    num_predicted=results['predicted_entities', entity_type.natural],
                    num_gt=results['gt_entities', entity_type.natural],
                )
                entity_precision_by_type.append(precision)
                entity_recall_by_type.append(recall)
                entity_f1_by_type.append(f1)

        intent_precision, intent_recall, intent_f1 = get_precision_recall_f1(
            num_correct=results['correct_intent'],
            num_predicted=results['predicted_intent'],
            num_gt=results['gt_intent']
        )

        res = {
            'wrong_reconstruction': results['wrong_reconstructions'] / results['num_sentences'],
            'label_error': results['label_error'] / results['num_sentences'],
            'format_error': results['format_error'] / results['num_sentences'],
            'intent_precision': intent_precision,
            'intent_recall': intent_recall,
            'intent_f1': intent_f1,
            'entity_precision': entity_precision,
            'entity_recall': entity_recall,
            'entity_f1': entity_f1,
            'entity_precision_no_type': entity_precision_no_type,
            'entity_recall_no_type': entity_recall_no_type,
            'entity_f1_no_type': entity_f1_no_type,
        }

        if macro:
            res.update({
                'entity_macro_precision': np.mean(np.array(entity_precision_by_type)),
                'entity_macro_recall': np.mean(np.array(entity_recall_by_type)),
                'entity_macro_f1': np.mean(np.array(entity_f1_by_type)),
            })

        return res


    def preprocess_for_glm(self, mode, dataset, fewshot=-1, debug=False):
        dataset_name = self.name; DATA_DIR = self.data_dir()+"/"
        source,target = self.preprocess_for_glm_single(self.load_data_single_split('train'), mode, dataset)


        ind = [i for i in range(len(source))]
        if fewshot>0:
            np.random.seed(0)
            np.random.shuffle(ind)
            ind = ind[:fewshot]
        source, target = [source[i] for i in ind], [target[i] for i in ind]
        with open(DATA_DIR+"train.source","w") as f:
            for line in source:
                print((self.prompt if mode=="fewshot" else "")+line, file=f)
        with open(DATA_DIR+"train.target","w") as f:
            for line in target:
                print(line, file=f)



        source,target = self.preprocess_for_glm_single(self.load_data_single_split('dev'), mode, dataset)
        ind = [i for i in range(len(source))]
        if debug:
            np.random.seed(0)
            np.random.shuffle(ind)
            ind = ind[:8]
        source, target = [source[i] for i in ind], [target[i] for i in ind]
        with open(DATA_DIR+"val.source","w") as f:
            for line in source:
                print((self.prompt if mode=="fewshot" else "")+line, file=f)
        with open(DATA_DIR+"val.target","w") as f:
            for line in target:
                print(line, file=f)

        source,target = self.preprocess_for_glm_single(self.load_data_single_split('test'), mode, dataset)
        ind = [i for i in range(len(source))]
        if debug:
            np.random.seed(0)
            np.random.shuffle(ind)
            ind = ind[:8]
        source, target = [source[i] for i in ind], [target[i] for i in ind]
        with open(DATA_DIR+"test.source","w") as f:
            for line in source:
                print((self.prompt if mode=="fewshot" else "")+line, file=f)
        with open(DATA_DIR+"test.target","w") as f:
            for line in target:
                print(line, file=f)

    def preprocess_for_glm_single(self, examples, mode, dataset):

        source = []; target = []
        for example in examples:
            source.append(self.output_format.SOURCE_FORMATS['id'].format(dataset.split('_')[0],' '.join(example.tokens),TASK_MAPPING[dataset]))
            target.append(self.output_format.format_output(example).replace('\n', '').replace('\r', ''))
        return source, target

@register_dataset
class ATISDataset(SnipsDataset):
    name = 'atis'

    natural_entity_types = {
        'fromloc': 'from', 'toloc': 'to',
        'city_name': 'city', 'state_code': 'state code', 'state_name': 'state name',
        'country_name': 'country name',
        'airport_code': 'airport code', 'airport_name': 'airport name',
        'depart_date': 'depart date', 'arrive_date': 'arrive date',
        'depart_time': 'depart time', 'arrive_time': 'arrive time',
        'return_date': 'return date', 'return_time': 'return time',
        'day_number': 'day number', 'day_name': 'day name', 'days_code': 'days code',
        'month_name': 'month', 'year': 'year',
        'date_relative': 'relative date', 'today_relative': 'relative today',
        'period_of_day': 'period of day', 'time_relative': 'relative time',
        'time': 'time', 'start_time': 'start time', 'end_time': 'end time',
        'cost_relative': 'relative cost',
        'airline_name': 'airline name', 'airline_code': 'airline code',
        'class_type': 'class type',
        'round_trip': 'round trip',
        'fare_basis_code': 'fare basis code',
        'fare_amount': 'fare amount',
        'meal': 'meal', 'meal_code': 'meal code', 'meal_description': 'meal description',
        'flight_mod': 'flight modify', 'mod': 'modify', 'period_mod': 'period modify',
        'stoploc': 'stop location',
        'connect': 'connect',
        'flight_number': 'flight number', 'flight_time': 'flight time', 'flight_stop': 'flight stop',
        'flight_days': 'flight days',
        'aircraft_code': 'aircraft code',
        'or': 'or',
        'restriction_code': 'restriction code',
        'transport_type': 'transport type',
        'economy': 'economy'
    }

    natural_intent_types = {
        'atis_flight': 'flight',
        'atis_airfare': 'airfare',
        'atis_airline': 'airline',
        'atis_aircraft': 'aircraft',
        'atis_flight#atis_airfare': 'flight and airfare',
        'atis_abbreviation': 'abbreviation',
        'atis_ground_service': 'ground service',
        'atis_meal': 'meal',
        'atis_restriction': 'restriction',
        'atis_quantity': 'quantity',
        'atis_aircraft#atis_flight#atis_flight_no': 'aircraft and flight and flight number',
        'atis_airport': 'airport',
        'atis_ground_fare': 'ground fare',
        'atis_airline#atis_flight_no': 'airline and flight number',
        'atis_flight_time': 'flight time',
        'atis_flight_no': 'flight number',
        'atis_distance': 'distance',
        'atis_city': 'city',
        'atis_capacity': 'capacity',
        'atis_cheapest': 'cheapest',
        'atis_ground_service#atis_ground_fare': 'ground service and ground fare',
        'atis_day_name': 'day name',
        'atis_airfare#atis_flight': 'airfare and flight',
        'atis_flight#atis_airline': 'flight number and airline',
        'atis_flight_no#atis_airline': 'flight number and airline'
    }

    def convert_bio_to_entities(self, bio_tag: List[str]) -> Tuple[List[Entity], Entity]:
        entities = []
        current_entity = None
        for ii, el in enumerate(bio_tag):
            if el.startswith('B-'):
                tag_type = el[2:]
                if '.' in tag_type:
                    natural = ' '.join([self.natural_entity_types[tag_part]
                            if tag_part in self.natural_entity_types else tag_part
                            for tag_part in tag_type.split('.')])
                else:
                    natural = self.natural_entity_types[tag_type] if tag_type in self.natural_entity_types else tag_type
                current_entity = Entity(
                    type=EntityType(
                        short=tag_type,
                        natural=natural
                    ),
                    start=ii,
                    end=ii+1,
                )
                entities.append(current_entity)
            elif el.startswith('I-'):
                current_entity.end = ii + 1
        return entities
