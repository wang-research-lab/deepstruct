# Adapted from https://github.com/amazon-science/tanl
import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, List, Dict
import numpy as np

from input_example import InputFeatures, EntityType, RelationType, Entity, Relation, Intent, InputExample, CorefDocument
from utils import augment_sentence, get_span


OUTPUT_FORMATS = {}


def register_output_format(format_class):
    OUTPUT_FORMATS[format_class.name] = format_class
    return format_class

def fix(string):
    return string.strip('[').strip(']') \
                 .replace(' ','') \
                 .replace('.','') \
                 .replace(',','') \
                 .replace('(','') \
                 .replace(')','').lower()

def oie_fix(string):
    return string.strip('[').strip(']') \
                 .replace(' ','') \
                 .replace('.','') \
                 .replace(',','') \
                 .replace('(','') \
                 .replace(')','').lower()

class BaseOutputFormat(ABC):
    name = None

    BEGIN_TUPLE_TOKEN = '('
    END_TUPLE_TOKEN = ')'
    SEP_INNER_TUPLE_TOKEN = ';'
    SEP_BETWEEN_TOKEN = '</s>'
    INSTANCE_OF = 'instance of'

    GLM_SEP_INNER_TUPLE_TOKEN = ';'
    GLM_SEP_BETWEEN_TOKEN = '[SEP]'
    GLM_SEP_ORG_BETWEEN_TOKEN = ') ('
    GLM_INSTANCE_OF = 'instance of'

    MODES = [
        'default',
        'multi',
        'empha',
        'task',
    ]
    TRIPLE_TOKENS = {
        'default':['( ',' )'],
        'multi'  :['( ',' )'],
        'empha'  :['( ',' )'],
        'task'   :['( ',' )'],
    }
    SEP_INTRA_TOKENS = {
        'default':' ; ',
        'multi'  :' ; ',
        'empha'  :' ; ',
        'task'   :' ; ',
    }
    SEP_INTER_TOKENS = {
        'default':' ',
        'multi'  :' ',
        'empha'  :' ',
        'task'   :' ',
    }
    SEP_HYPER_TOKENS = {
        'default':' ',
        'multi'  :' ',
        'empha'  :' ',
        'task'   :' ',
    }
    INSTANCE_OF_TOKENS = {
        'default':'instance of',
        'multi'  :'instance of',
        'empha'  :'instance of',
        'task'   :'instance of',
    }
    SENTENSE_TOKENS = {
        'default':[   ''   ,''      ],
        'multi'  :[   ''   ,''      ],
        'empha'  :[   ''   ,''      ],
        'task'   :[   ''   ,''      ],
    }
    SOURCE_FORMATS = {
        'default':"Sentence : {1}",
        'multi'  :"Dataset : {0} Task : {2} Sentence : {1}",
        'empha'  :"Dataset : {0} Task : {2} Sentence : {1}",
        'task'   :"Task : {2} Sentence : {1}",
        'oie'    :"Dataset : {0} Task : {2} oie {0} sentence : {1}",
        'rc'     :"Dataset : {0} Task : {2} rc {0} sentence : {1}",
        'id'     :"Dataset : {0} Task : {2} id {0} sentence : {1}",
    }
    EMPHASIS_TOKENS = {
        'default':[ '' , '' ],
        'multi'  :[ '' , '' ],
        'empha'  :['[ ',' ]'],
        'task'   :[ '' , '' ],
    }

    BEGIN_ENTITY_TOKEN = '['
    END_ENTITY_TOKEN = ']'
    SEPARATOR_TOKEN = '|'
    RELATION_SEPARATOR_TOKEN = '='

    @abstractmethod
    def format_output(self, example: InputExample) -> str:
        """
        Format output in augmented natural language.
        """
        raise NotImplementedError

    @abstractmethod
    def run_inference(self, example: InputExample, output_sentence: str):
        """
        Process an output sentence to extract whatever information the task asks for.
        """
        raise NotImplementedError


    def parse_output_sentence(self, output_sentence: str, mode: str = 'default', rel_types=None, tail_types=None):
        assert(mode in self.MODES)
        triples, entities, entities_no_type = [], [], []
        try:

            wrong_reconstruction = False
            text_examples = []
            for triple in output_sentence.strip() \
                                         .strip(self.SENTENSE_TOKENS[mode][0]) \
                                         .strip(self.SENTENSE_TOKENS[mode][1]) \
                                         .strip(self.TRIPLE_TOKENS[mode][0]) \
                                         .strip(self.TRIPLE_TOKENS[mode][1]) \
                                         .replace(self.TRIPLE_TOKENS[mode][1]
                                                 +self.SEP_HYPER_TOKENS[mode]
                                                 +self.TRIPLE_TOKENS[mode][0]
                                                 ,self.TRIPLE_TOKENS[mode][1]
                                                 +self.SEP_INTER_TOKENS[mode]
                                                 +self.TRIPLE_TOKENS[mode][0]) \
                                         .split(self.TRIPLE_TOKENS[mode][1]
                                               +self.SEP_INTER_TOKENS[mode]
                                               +self.TRIPLE_TOKENS[mode][0]) :
                assert(len(triple.split(self.SEP_INTRA_TOKENS[mode]))==3)
                h, r, t = triple.split(self.SEP_INTRA_TOKENS[mode])
                h, r, t = fix(h), r, fix(t)
                if ((rel_types is None) or (r in rel_types)) and ((tail_types is None) or (t in tail_types)):
                    text_examples.append((h,r,t))
                else:
                    print(h, r, t, ((rel_types is None) or (r in rel_types)), ((tail_types is None) or (t in tail_types)))

            for triple in text_examples:
                if triple[1] == self.INSTANCE_OF_TOKENS[mode]:
                    entities.append(triple)
                    entities_no_type.append(triple[0])
                else:
                    triples.append(triple)

            triples = set(triples)
            entities = set(entities)
            entities_no_type = set(entities_no_type)
        except:

            wrong_reconstruction = True
            triples = set()
            entities = set()
            entities_no_type = set()

        return triples, entities, entities_no_type, wrong_reconstruction


@register_output_format
class OIEOutputFormat(BaseOutputFormat):
    """
    Output format for open information extraction.
    """
    name = 'oie'

    def format_output(self, example: InputExample, mode = 'default', uncased=False) -> str:
        assert(mode in self.MODES)
        output_strings = []
        words = example.tokens
        triple_token = self.TRIPLE_TOKENS[mode]
        sep_intra_token = self.SEP_INTRA_TOKENS[mode]
        sep_inter_token = self.SEP_INTER_TOKENS[mode]
        sep_hyper_token = self.SEP_HYPER_TOKENS[mode]
        instance_of_token = self.INSTANCE_OF_TOKENS[mode]
        sentence_token = self.SENTENSE_TOKENS[mode]
        entities = set()

        for relation in example.relations:
            buf_strings = []


            head_mention = ' '.join(words[relation.head.start:relation.head.end])
            tail_mention = ' '.join(words[relation.tail.start:relation.tail.end])
            buf_strings.append(triple_token[0]+head_mention+sep_intra_token+relation.type+sep_intra_token+tail_mention+triple_token[1])


            output_strings.append(sep_inter_token.join([s.lower() for s in buf_strings] if uncased else buf_strings))


        output = sep_hyper_token.join(output_strings).strip()


        output = output.replace("person","human")

        return (sentence_token[0]+output+sentence_token[1]).strip()

    def parse_output_sentence(self, output_sentence: str, mode: str = 'default', rel_types=None, tail_types=None):
        assert(mode in self.MODES)
        triples = []
        try:
            wrong_reconstruction = False
            text_examples = []
            for triple in output_sentence.strip() \
                                         .strip(self.SENTENSE_TOKENS[mode][0]) \
                                         .strip(self.SENTENSE_TOKENS[mode][1]) \
                                         .split(self.SEP_TOKENS[mode]):
                assert(len(triple.split(self.SEP_INTRA_TOKENS[mode]))>=3)
                h, r, t = triple.split(self.SEP_INTRA_TOKENS[mode])
                if len(triple.split(self.SEP_INTRA_TOKENS[mode]))>3:
                    t = self.SEP_INTRA_TOKENS[mode].join(triple.split(self.SEP_INTRA_TOKENS[mode])[2: ])
                h, r, t = oie_fix(h), r, oie_fix(t)
                if ((rel_types is None) or (r in rel_types)) and ((tail_types is None) or (t in tail_types)):
                    text_examples.append((h,r,t))

            for triple in text_examples:
                    triples.append(str(triple))

            triples = set(triples)



        except:
            wrong_reconstruction = True
            triples = set()

        return triples, wrong_reconstruction

    def run_inference(self, example: InputExample, output_sentence: str,
                      entity_types: Dict[str, EntityType] = None, relation_types: Dict[str, RelationType] = None, mode = 'default') \
            -> Tuple[set, set, bool, bool, bool, bool]:

        format_error = False

        relation_types = set(relation_type for relation_type in relation_types.values()) \
            if relation_types is not None else {}


        predicted_relations, wrong_reconstruction = self.parse_output_sentence(output_sentence, mode=mode)


        true_string = self.format_output(example, mode=mode)
        true_relations, _ = self.parse_output_sentence(true_string, mode=mode)








        if wrong_reconstruction:
            format_error = True

        return predicted_relations, true_relations, wrong_reconstruction, format_error


@register_output_format
class JointEROutputFormat(BaseOutputFormat):
    """
    Output format for joint entity and relation extraction.
    """
    name = 'joint_er'


    def format_output(self, example: InputExample, mode = 'default', uncased=False) -> str:
        assert(mode in self.MODES)
        output_strings = []
        words = example.tokens
        triple_token = self.TRIPLE_TOKENS[mode]
        sep_intra_token = self.SEP_INTRA_TOKENS[mode]
        sep_inter_token = self.SEP_INTER_TOKENS[mode]
        sep_hyper_token = self.SEP_HYPER_TOKENS[mode]
        instance_of_token = self.INSTANCE_OF_TOKENS[mode]
        sentence_token = self.SENTENSE_TOKENS[mode]
        emphasis_token = self.EMPHASIS_TOKENS[mode]
        example.entities = sorted(example.entities,key=lambda x:x.start)
        entities = set()























        for entity in example.entities:
            if entity not in entities:
                entity_mention = ' '.join(words[entity.start:entity.end]); entities.add(entity)
                buf_string = triple_token[0]+emphasis_token[0]+entity_mention+emphasis_token[1]+sep_intra_token+instance_of_token+sep_intra_token+entity.type.natural+triple_token[1]
                output_strings.append(buf_string.lower() if uncased else buf_string)


        output = sep_hyper_token.join(output_strings).strip()


        output = output.replace("person","human")

        return (sentence_token[0]+output+sentence_token[1]).strip()


    def run_inference(self, example: InputExample, output_sentence: str,
                      entity_types: Dict[str, EntityType] = None, relation_types: Dict[str, RelationType] = None, mode = 'default') \
            -> Tuple[set, set, bool, bool, bool, bool]:

        label_error = False
        entity_error = False
        format_error = False

        entity_types = set(entity_type.natural for entity_type in entity_types.values())
        relation_types = set(relation_type.natural for relation_type in relation_types.values()) \
            if relation_types is not None else {}


        predicted_relations, predicted_entities, predicted_entities_no_type, wrong_reconstruction = self.parse_output_sentence(output_sentence, mode=mode)


        true_string = self.format_output(example, mode=mode)
        true_relations, true_entities, true_entities_no_type, _ = self.parse_output_sentence(true_string, mode=mode)



















        if wrong_reconstruction:
            format_error = True

        if not predicted_entities == true_entities:
            entity_error = True

        return predicted_relations, predicted_entities, predicted_entities_no_type, true_relations, true_entities, true_entities_no_type, wrong_reconstruction, label_error, entity_error, format_error

@register_output_format
class RelationOnlyOutputFormat(JointEROutputFormat):
    """
    Output format for joint entity and relation extraction.
    """
    name = 're'


    def format_output(self, example: InputExample, mode = 'default', uncased=False) -> str:
        assert(mode in self.MODES)
        output_strings = []
        words = example.tokens
        triple_token = self.TRIPLE_TOKENS[mode]
        sep_intra_token = self.SEP_INTRA_TOKENS[mode]
        sep_inter_token = self.SEP_INTER_TOKENS[mode]
        sep_hyper_token = self.SEP_HYPER_TOKENS[mode]
        instance_of_token = self.INSTANCE_OF_TOKENS[mode]
        sentence_token = self.SENTENSE_TOKENS[mode]
        emphasis_token = self.EMPHASIS_TOKENS[mode]
        example.entities = sorted(example.entities,key=lambda x:x.start)
        entities = set()

        for relation in example.relations:
            buf_strings = []


            head_mention = ' '.join(words[relation.head.start:relation.head.end])
            tail_mention = ' '.join(words[relation.tail.start:relation.tail.end])
            buf_strings.append(triple_token[0]+emphasis_token[0]+head_mention+emphasis_token[1]+sep_intra_token+relation.type.natural+sep_intra_token+emphasis_token[0]+tail_mention+emphasis_token[1]+triple_token[1])












            output_strings.append(sep_inter_token.join([s.lower() for s in buf_strings] if uncased else buf_strings))









        output = sep_hyper_token.join(output_strings).strip()


        output = output.replace("person","human")

        return (sentence_token[0]+output+sentence_token[1]).strip()

@register_output_format
class NEROutputFormat(JointEROutputFormat):
    """
    Output format for joint entity and relation extraction.
    """
    name = 'ner'


    def format_output(self, example: InputExample, mode = 'default', uncased=False) -> str:
        assert(mode in self.MODES)
        output_strings = []
        words = example.tokens
        triple_token = self.TRIPLE_TOKENS[mode]
        sep_intra_token = self.SEP_INTRA_TOKENS[mode]
        sep_inter_token = self.SEP_INTER_TOKENS[mode]
        sep_hyper_token = self.SEP_HYPER_TOKENS[mode]
        instance_of_token = self.INSTANCE_OF_TOKENS[mode]
        sentence_token = self.SENTENSE_TOKENS[mode]
        emphasis_token = self.EMPHASIS_TOKENS[mode]
        example.entities = sorted(example.entities,key=lambda x:x.start)
        entities = set()


        for entity in example.entities:
            if entity not in entities:
                entity_mention = ' '.join(words[entity.start:entity.end]); entities.add(entity)
                buf_string = triple_token[0]+emphasis_token[0]+entity_mention+emphasis_token[1]+sep_intra_token+instance_of_token+sep_intra_token+entity.type.natural+triple_token[1]
                output_strings.append(buf_string.lower() if uncased else buf_string)


        output = sep_hyper_token.join(output_strings).strip()


        output = output.replace("person","human")

        return (sentence_token[0]+output+sentence_token[1]).strip()


    def run_inference(self, example: InputExample, output_sentence: str,
                      entity_types: Dict[str, EntityType] = None, relation_types: Dict[str, RelationType] = None, mode = 'default') \
            -> Tuple[set, set, bool, bool, bool, bool]:

        label_error = False
        entity_error = False
        format_error = False


        if entity_types is not None:
            tail_types = set(e.natural for e in entity_types.values())
            if 'person' in tail_types:
                tail_types.remove('person'); tail_types.add('human')
            tail_types = set(fix(tail_type) for tail_type in tail_types)
        else:
            tail_types = None
        predicted_relations, predicted_entities, predicted_entities_no_type, wrong_reconstruction = self.parse_output_sentence(output_sentence, mode=mode, tail_types=tail_types)


        true_string = self.format_output(example, mode=mode)
        true_relations, true_entities, true_entities_no_type, _ = self.parse_output_sentence(true_string, mode=mode, tail_types=tail_types)



        if wrong_reconstruction:
            format_error = True

        if not predicted_entities == true_entities:
            entity_error = True

        return predicted_relations, predicted_entities, predicted_entities_no_type, true_relations, true_entities, true_entities_no_type, wrong_reconstruction, label_error, entity_error, format_error

@register_output_format
class EventArgumentOutputFormat(NEROutputFormat):
    """
    Output format for joint entity and relation extraction.
    """
    name = 'event_argument'


    def run_inference(self, example: InputExample, output_sentence: str,
                      entity_types: Dict[str, EntityType] = None, relation_types: Dict[str, RelationType] = None, mode = 'default') \
            -> Tuple[set, set, bool, bool, bool, bool]:

        label_error = False
        entity_error = False
        format_error = False


        if relation_types is not None:
            tail_types = set(e.natural for e in entity_types.values())
            if 'person' in tail_types:
                tail_types.remove('person'); tail_types.add('human')
            tail_types = set(fix(tail_type) for tail_type in tail_types)
        else:
            tail_types = None
        predicted_relations, predicted_entities, predicted_entities_no_type, wrong_reconstruction = self.parse_output_sentence(output_sentence, mode=mode, tail_types=tail_types)


        true_string = self.format_output(example, mode=mode)
        true_relations, true_entities, true_entities_no_type, _ = self.parse_output_sentence(true_string, mode=mode, tail_types=tail_types)



        if wrong_reconstruction:
            format_error = True

        if not predicted_entities == true_entities:
            entity_error = True

        return predicted_relations, predicted_entities, predicted_entities_no_type, true_relations, true_entities, true_entities_no_type, wrong_reconstruction, label_error, entity_error, format_error

@register_output_format
class JointICSLFormat(JointEROutputFormat):
    """
    Output format for joint intent classification and slot labeling.
    """
    name = 'joint_icsl'
    BEGIN_INTENT_TOKEN = "(("
    END_INTENT_TOKEN = "))"

    def format_output(self, example: InputExample) -> str:
        """
        Get output in augmented natural language.
        """
        augmentations = []
        for entity in example.entities:
            tags = [(entity.type.natural,)]

            augmentations.append((
                tags,
                entity.start,
                entity.end,
            ))

        augmented_sentence = augment_sentence(example.tokens, augmentations, self.BEGIN_ENTITY_TOKEN, self.SEPARATOR_TOKEN,
                                self.RELATION_SEPARATOR_TOKEN, self.END_ENTITY_TOKEN)

        return (f"(( {example.intent.natural} )) " + augmented_sentence)

    def run_inference(self, example: InputExample, output_sentence: str,
            entity_types: Dict[str, EntityType] = None) -> Tuple[str, set]:
        entity_types = set(entity_type.natural for entity_type in entity_types.values())



        for special_token in [self.BEGIN_INTENT_TOKEN, self.END_INTENT_TOKEN]:
            output_sentence.replace(special_token, ' ' + special_token + ' ')

        output_sentence_tokens = output_sentence.split()

        if self.BEGIN_INTENT_TOKEN in output_sentence_tokens and \
                self.END_INTENT_TOKEN in output_sentence_tokens:
            intent = output_sentence.split(self.BEGIN_INTENT_TOKEN)[1].split(self.END_INTENT_TOKEN)[0].strip()
            output_sentence = output_sentence.split(self.END_INTENT_TOKEN)[1]

        label_error = False
        format_error = False

        if output_sentence.count(self.BEGIN_ENTITY_TOKEN) != output_sentence.count(self.END_ENTITY_TOKEN):

            format_error = True


        raw_predicted_entities, wrong_reconstruction = self.parse_output_sentence(example, output_sentence)


        predicted_entities_by_name = defaultdict(list)
        predicted_entities = set()


        for entity_name, tags, start, end in raw_predicted_entities:
            if len(tags) == 0 or len(tags[0]) > 1:

                format_error = True
                continue

            entity_type = tags[0][0]

            if entity_type in entity_types:
                entity_tuple = (entity_type, start, end)
                predicted_entities.add(entity_tuple)
            else:
                label_error = True

        return intent, predicted_entities, wrong_reconstruction, label_error, format_error


@register_output_format
class EventOutputFormat(JointEROutputFormat):
    """
    Output format for event extraction, where an input example contains exactly one trigger.
    """
    name = 'ace2005_event'

    def format_output(self, example: InputExample) -> str:
        """
        Get output in augmented natural language, similarly to JointEROutputFormat (but we also consider triggers).
        """

        relations_by_entity = {entity: [] for entity in example.entities + example.triggers}
        for relation in example.relations:
            relations_by_entity[relation.head].append((relation.type, relation.tail))

        augmentations = []
        for entity in (example.entities + example.triggers):
            if not relations_by_entity[entity]:
                continue

            tags = [(entity.type.natural,)]
            for relation_type, tail in relations_by_entity[entity]:
                tags.append((relation_type.natural, ' '.join(example.tokens[tail.start:tail.end])))

            augmentations.append((
                tags,
                entity.start,
                entity.end,
            ))

        return augment_sentence(example.tokens, augmentations, self.BEGIN_ENTITY_TOKEN, self.SEPARATOR_TOKEN,
                                self.RELATION_SEPARATOR_TOKEN, self.END_ENTITY_TOKEN)

    def run_inference(self, example: InputExample, output_sentence: str,
                      entity_types: Dict[str, EntityType] = None, relation_types: Dict[str, RelationType] = None) \
            -> Tuple[set, set, bool]:
        """
        Process an output sentence to extract arguments, given as entities and relations.
        """
        entity_types = set(entity_type.natural for entity_type in entity_types.values())
        relation_types = set(relation_type.natural for relation_type in relation_types.values()) \
            if relation_types is not None else {}

        triggers = example.triggers
        assert len(triggers) <= 1
        if len(triggers) == 0:

            return set(), set(), False

        trigger = triggers[0]


        raw_predicted_entities, wrong_reconstruction = self.parse_output_sentence(example, output_sentence)


        predicted_entities = set()
        predicted_relations = set()


        for entity_name, tags, start, end in raw_predicted_entities:
            if len(tags) == 0 or len(tags[0]) > 1:

                continue

            entity_type = tags[0][0]

            if entity_type in entity_types:
                entity_tuple = (entity_type, start, end)
                predicted_entities.add(entity_tuple)


                for tag in tags[1:]:
                    if len(tag) == 2:
                        relation_type, related_entity = tag
                        if relation_type in relation_types:
                            predicted_relations.add(
                                (relation_type, entity_tuple, (trigger.type.natural, trigger.start, trigger.end))
                            )

        return predicted_entities, predicted_relations, wrong_reconstruction

class UFS(object):
    def __init__(self, n):
        self.reset(n)

    def __call__(self, i):
        if i!=self.fa[i]:
            self.fa[i] = self(self.fa[i])
        return self.fa[i]

    def merge(self, a, b):
        fa_a = self(a); fa_b = self(b); self.fa[fa_a] = fa_b

    def reset(self, n=None):
        self.fa = list(range(len(self))) if n is None else list(range(n))

    def get_groups(self):
        groups = defaultdict(list)
        for i in range(len(self)):
            groups[self(i)].append(i)
        return list(groups.values())

    def __len__(self):
        return len(self.fa)

from fuzzywuzzy import fuzz
@register_output_format
class CorefOutputFormat(NEROutputFormat):
    """
    Output format for coreference resolution.
    """
    name = 'coref'

    INSTANCE_OF_TOKENS = {
        'default':'refer to',
        'multi'  :'refer to',
        'empha'  :'refer to',
        'task'   :'refer to',
    }


    def format_output(self, example: InputExample, mode = 'default', uncased=True) -> str:
        assert(mode in self.MODES)
        output_strings = []
        words = example.tokens
        triple_token = self.TRIPLE_TOKENS[mode]
        sep_intra_token = self.SEP_INTRA_TOKENS[mode]
        sep_inter_token = self.SEP_INTER_TOKENS[mode]
        sep_hyper_token = self.SEP_HYPER_TOKENS[mode]
        instance_of_token = self.INSTANCE_OF_TOKENS[mode]
        sentence_token = self.SENTENSE_TOKENS[mode]
        emphasis_token = self.EMPHASIS_TOKENS[mode]
        entities = set()


        for group in example.groups:
            group = sorted(group,key=lambda x:x.start)
            for p, e in zip(group,group[1:]):
                head_mention = ' '.join(words[p.start:p.end])
                tail_mention = ' '.join(words[e.start:e.end])
                buf_string = triple_token[0]+emphasis_token[0]+head_mention+emphasis_token[1]+sep_intra_token+instance_of_token+sep_intra_token+emphasis_token[0]+tail_mention+emphasis_token[1]+triple_token[1]
                output_strings.append((p.start,buf_string.lower() if uncased else buf_string))
        output_strings = list(x[1] for x in sorted(output_strings, key=lambda x:x[0]))

        output = sep_hyper_token.join(output_strings).strip()

        return (sentence_token[0]+output+sentence_token[1]).strip()

    def mention_to_span(self, pred_corefs, tokens):
        sentence = ''.join([fix(token) for token in tokens])
        mapped_pred_corefs = list()
        for head, ref, tail in pred_corefs:
            tail_match = [fuzz.ratio(tail, sentence[r-len(tail):r]) for r in range(len(tail),len(sentence))]
            l = np.argmax(tail_match)
            if tail_match[l]>80:
                tail_span = (l, l+len(tail))
            else:
                tail_span = None
            head_match = [fuzz.ratio(head, sentence[r-len(head):r]) for r in range(len(head),len(sentence))]
            l = np.argmax(head_match)
            if head_match[l]>80:
                head_span = (l, l+len(head))
            else:
                head_span = None
            if (head_span is not None) and (tail_span is not None):
                mapped_pred_corefs.append((head_span, tail_span))
        return mapped_pred_corefs

    def span_to_group(self, refs):
        all_spans = list(set(ref[0] for ref in refs).union(set(ref[1] for ref in refs)))
        spans_ids = {span:i for i,span in enumerate(all_spans)}; ufs = UFS(len(all_spans))
        for head, tail in refs:
            h = spans_ids[head]; t = spans_ids[tail]; ufs.merge(h,t)
        return [[all_spans[i] for i in g] for g in ufs.get_groups() if len(g)>0]

    def run_inference(self, example: InputExample, output_sentence: str, mode='default') \
            -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Process an output sentence to extract coreference relations.
        Return a list of ((start, end), parent) where (start, end) denote an entity span, and parent is either None
        or another (previous) entity span.
        """
        _, pred_corefs, _, _ = self.parse_output_sentence(output_sentence, mode=mode)
        _, true_corefs, _, _ = self.parse_output_sentence(
            self.format_output(example, mode=mode), mode=mode
        )

        pred_spans = self.mention_to_span(pred_corefs, example.tokens)
        pred_groups = self.span_to_group(pred_spans)
        true_spans = self.mention_to_span(true_corefs, example.tokens)
        true_groups = self.span_to_group(true_spans)



        return true_groups, pred_groups


@register_output_format
class RelationClassificationOutputFormat(BaseOutputFormat):
    """
    Output format for relation classification.
    """
    name = 'rc_output'

    def format_output(self, example: InputExample, mode = 'default', uncased=False) -> str:
        assert(mode in self.MODES)
        output_strings = []
        words = example.tokens
        triple_token = self.TRIPLE_TOKENS[mode]
        sep_intra_token = self.SEP_INTRA_TOKENS[mode]
        sep_inter_token = self.SEP_INTER_TOKENS[mode]
        sep_hyper_token = self.SEP_HYPER_TOKENS[mode]
        instance_of_token = self.INSTANCE_OF_TOKENS[mode]
        sentence_token = self.SENTENSE_TOKENS[mode]
        example.entities = sorted(example.entities,key=lambda x:x.start)
        entities = set()

        for relation in example.relations:
            buf_strings = []


            head_mention = ' '.join(words[relation.head.start:relation.head.end])
            tail_mention = ' '.join(words[relation.tail.start:relation.tail.end])
            buf_strings.append(triple_token[0]+head_mention+sep_intra_token+relation.type.natural+sep_intra_token+tail_mention+triple_token[1])


            output_strings.append(sep_inter_token.join([s.lower() for s in buf_strings] if uncased else buf_strings))


        output = sep_hyper_token.join(output_strings).strip()

        return (sentence_token[0]+output+sentence_token[1]).strip()

    def parse_output_sentence(self, output_sentence: str, mode: str = 'default', rel_types=None, tail_types=None):
        assert(mode in self.MODES)
        triples = []
        try:
            wrong_reconstruction = False
            relations = []
            for triple in output_sentence.strip() \
                                         .strip(self.SENTENSE_TOKENS[mode][0]) \
                                         .strip(self.SENTENSE_TOKENS[mode][1]) \
                                         .split(self.SEP_TOKENS[mode]):
                assert(len(triple.split(self.SEP_INTRA_TOKENS[mode]))>=3)
                h, r, t = triple.split(self.SEP_INTRA_TOKENS[mode])
                if len(triple.split(self.SEP_INTRA_TOKENS[mode]))>3:
                    t = self.SEP_INTRA_TOKENS[mode].join(triple.split(self.SEP_INTRA_TOKENS[mode])[2: ])
                h, r, t = fix(h), r, fix(t)
                if ((rel_types is None) or (r in rel_types)) and ((tail_types is None) or (t in tail_types)):
                    relations.append(r)

            relations = set(relations)

        except:
            wrong_reconstruction = True
            relations = set()

        return relations, wrong_reconstruction

    def run_inference(self, example: InputExample, output_sentence: str,
                      entity_types: Dict[str, EntityType] = None, relation_types: Dict[str, RelationType] = None, mode = 'default') \
            -> Tuple[set, set, bool, bool, bool, bool]:

        format_error = False

        relation_types = set(relation_type for relation_type in relation_types.values()) \
            if relation_types is not None else {}


        predicted_relations, wrong_reconstruction = self.parse_output_sentence(output_sentence, mode=mode)


        true_string = self.format_output(example, mode=mode)
        true_relations, _ = self.parse_output_sentence(true_string, mode=mode)

        if wrong_reconstruction:
            format_error = True

        return predicted_relations, true_relations, wrong_reconstruction, format_error


@register_output_format
class MultiWozOutputFormat(BaseOutputFormat):
    """
    Output format for the MultiWoz DST dataset.
    """
    name = 'multi_woz'

    none_slot_value = 'not given'
    domain_ontology = {
        'hotel': [
            'price range',
            'type',
            'parking',
            'book stay',
            'book day',
            'book people',
            'area',
            'stars',
            'internet',
            'name'
        ],
        'train': [
            'destination',
            'day',
            'departure',
            'arrive by',
            'book people',
            'leave at'
        ],
        'attraction': ['type', 'area', 'name'],
        'restaurant': [
            'book people',
            'book day',
            'book time',
            'food',
            'price range',
            'name',
            'area'
        ],
        'taxi': ['leave at', 'destination', 'departure', 'arrive by'],
        'bus': ['people', 'leave at', 'destination', 'day', 'arrive by', 'departure'],
        'hospital': ['department']
    }

    def format_output(self, example: InputExample) -> str:
        """
        Get output in augmented natural language, for example:
        [belief] hotel price range cheap , hotel type hotel , duration two [belief]
        """
        turn_belief = example.belief_state
        domain_to_slots = defaultdict(dict)
        for label in turn_belief:
            domain, slot, value = label.split("-")
            domain_to_slots[domain][slot] = value


        for domain, slot_dict in domain_to_slots.items():
            for slot in self.domain_ontology[domain]:
                if slot not in slot_dict:
                    slot_dict[slot] = self.none_slot_value

        output_list = []
        for domain, slot_dict in sorted(domain_to_slots.items(), key=lambda p: p[0]):
            output_list += [
                f"( [User] ; {domain} {slot} ; {value} )" for slot, value in sorted(slot_dict.items(), key=lambda p: p[0])
            ]
        output = " ".join(output_list)
        return output

    def run_inference(self, example: InputExample, output_sentence: str):
        """
        Process an output sentence to extract the predicted belief.
        """
        start = output_sentence.find("[belief]")
        end = output_sentence.rfind("[belief]")

        label_span = output_sentence[start+len("[belief]"):end]
        belief_set = set([
            slot_value.strip() for slot_value in label_span.split(",")
            if self.none_slot_value not in slot_value
        ])
        return belief_set


@register_output_format
class IDOutputFormat(JointEROutputFormat):
    """
    Output format for joint intent classification and slot labeling.
    """
    name = 'intent_detection'

    def format_output(self, example: InputExample, mode = 'default', uncased=False) -> str:
        assert(mode in self.MODES)
        output_strings = []
        words = example.tokens
        triple_token = self.TRIPLE_TOKENS[mode]
        sep_intra_token = self.SEP_INTRA_TOKENS[mode]
        sep_inter_token = self.SEP_INTER_TOKENS[mode]
        sep_hyper_token = self.SEP_HYPER_TOKENS[mode]
        instance_of_token = self.INSTANCE_OF_TOKENS[mode]
        sentence_token = self.SENTENSE_TOKENS[mode]
        entities = set()

        intent = example.intent
        buf_strings = []

        head_mention = 'intent'
        relation = 'is'
        buf_strings.append(triple_token[0]+head_mention+sep_intra_token+relation+sep_intra_token+intent.natural+triple_token[1])

        output_strings.append(sep_inter_token.join([s for s in buf_strings] if uncased else buf_strings))

        output = sep_hyper_token.join(output_strings).strip()

        return (sentence_token[0]+output+sentence_token[1]).strip()

    def parse_output_sentence(self, output_sentence: str, mode: str = 'default'):
        assert(mode in self.MODES)
        triples = []
        try:
            wrong_reconstruction = False
            relations = []
            for triple in output_sentence.strip() \
                                         .strip(self.SENTENSE_TOKENS[mode][0]) \
                                         .strip(self.SENTENSE_TOKENS[mode][1]) \
                                         .split(self.SEP_TOKENS[mode]):
                assert(len(triple.split(self.SEP_INTRA_TOKENS[mode]))>=3)
                h, r, t = triple.split(self.SEP_INTRA_TOKENS[mode])
                if len(triple.split(self.SEP_INTRA_TOKENS[mode]))>3:
                    t = self.SEP_INTRA_TOKENS[mode].join(triple.split(self.SEP_INTRA_TOKENS[mode])[2: ])
                h, r, t = fix(h), r, fix(t)
                relations.append(r)

            relations = set(relations)

        except:
            wrong_reconstruction = True
            relations = set()

        return relations, wrong_reconstruction

    def run_inference(self, example: InputExample, output_sentence: str,
                      entity_types: Dict[str, EntityType] = None, relation_types: Dict[str, RelationType] = None, mode = 'default') \
            -> Tuple[set, set, bool, bool, bool, bool]:

        format_error = False

        relation_types = set(relation_type for relation_type in relation_types.values()) \
            if relation_types is not None else {}


        predicted_relations, wrong_reconstruction = self.parse_output_sentence(output_sentence, mode=mode)


        true_string = self.format_output(example, mode=mode)
        true_relations, _ = self.parse_output_sentence(true_string, mode=mode)

        print("predicted_output: ", output_sentence)
        print("true_output: ", true_string)
        print("predicted_relations: ", predicted_relations)
        print("true_relations: ", true_relations)

        print("predicted_relations is_correct: ", predicted_relations==true_relations)

        if wrong_reconstruction:
            format_error = True

        return predicted_relations, true_relations, wrong_reconstruction, format_error


@register_output_format
class JointICSLFormat(JointEROutputFormat):
    """
    Output format for joint intent classification and slot labeling.
    """
    name = 'joint_icsl'
    BEGIN_INTENT_TOKEN = "(("
    END_INTENT_TOKEN = "))"

    def format_output(self, example: InputExample) -> str:
        """
        Get output in augmented natural language.
        """
        augmentations = []
        for entity in example.entities:
            tags = [(entity.type.natural,)]

            augmentations.append((
                tags,
                entity.start,
                entity.end,
            ))

        augmented_sentence = augment_sentence(example.tokens, augmentations, self.BEGIN_ENTITY_TOKEN, self.SEPARATOR_TOKEN,
                                self.RELATION_SEPARATOR_TOKEN, self.END_ENTITY_TOKEN)

        return (f"(( {example.intent.natural} )) " + augmented_sentence)

    def run_inference(self, example: InputExample, output_sentence: str,
            entity_types: Dict[str, EntityType] = None) -> Tuple[str, set]:
        entity_types = set(entity_type.natural for entity_type in entity_types.values())



        for special_token in [self.BEGIN_INTENT_TOKEN, self.END_INTENT_TOKEN]:
            output_sentence.replace(special_token, ' ' + special_token + ' ')

        output_sentence_tokens = output_sentence.split()

        if self.BEGIN_INTENT_TOKEN in output_sentence_tokens and \
                self.END_INTENT_TOKEN in output_sentence_tokens:
            intent = output_sentence.split(self.BEGIN_INTENT_TOKEN)[1].split(self.END_INTENT_TOKEN)[0].strip()
            output_sentence = output_sentence.split(self.END_INTENT_TOKEN)[1]

        label_error = False
        format_error = False

        if output_sentence.count(self.BEGIN_ENTITY_TOKEN) != output_sentence.count(self.END_ENTITY_TOKEN):

            format_error = True


        raw_predicted_entities, wrong_reconstruction = self.parse_output_sentence(example, output_sentence)


        predicted_entities_by_name = defaultdict(list)
        predicted_entities = set()


        for entity_name, tags, start, end in raw_predicted_entities:
            if len(tags) == 0 or len(tags[0]) > 1:

                format_error = True
                continue

            entity_type = tags[0][0]

            if entity_type in entity_types:
                entity_tuple = (entity_type, start, end)
                predicted_entities.add(entity_tuple)
            else:
                label_error = True

        return intent, predicted_entities, wrong_reconstruction, label_error, format_error
