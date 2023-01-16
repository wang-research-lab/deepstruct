# Adapted from https://github.com/amazon-science/tanl
from abc import ABC, abstractmethod
import copy

from input_example import InputExample
from utils import augment_sentence, get_span

INPUT_FORMATS = {}


def register_input_format(format_class):
    INPUT_FORMATS[format_class.name] = format_class
    return format_class


class BaseInputFormat(ABC):
    name = None

    BEGIN_ENTITY_TOKEN = '['
    END_ENTITY_TOKEN = ']'
    SEPARATOR_TOKEN = '|'
    RELATION_SEPARATOR_TOKEN = '='
    QUERY_SEPARATOR_TOKEN = ':'

    def format_input(self, example: InputExample, multitask=False, task_descriptor=None):
        res = self._format_input(example=example)
        if multitask:
            name = task_descriptor or example.dataset.task_descriptor or example.dataset.name
            res = f'{name} {self.QUERY_SEPARATOR_TOKEN} ' + res
        return res

    @abstractmethod
    def _format_input(self, example: InputExample, name='') -> str:
        raise NotImplementedError


@register_input_format
class PlainInputFormat(BaseInputFormat):
    """
    This format uses the plain sentence as input.
    """
    name = 'plain'

    def _format_input(self, example: InputExample) -> str:
        return ' '.join(example.tokens)

class JointERInputFormat(BaseInputFormat):
    name = ''
    def _format_input(self, example: InputExample) -> str:
        return "{} sentence: {}".format(self.name, " ".join(example.tokens))

@register_input_format
class Conll04InputFormat(JointERInputFormat):
    name = 'conll04'

@register_input_format
class ADEInputFormat(JointERInputFormat):
    name = 'ade'

@register_input_format
class NYTInputFormat(JointERInputFormat):
    name = 'nyt'

@register_input_format
class ACE2005REInputFormat(JointERInputFormat):
    name = 'ace2005_joint_er'

@register_input_format
class RelationClassificationInputFormat(BaseInputFormat):
    """
    Input format for relation classification.
    """
    name = 'rc_input'

    ENTITY_SQUARE_BRACKET_LEFT = ''
    ENTITY_SQUARE_BRACKET_RIGHT = ''

    def _format_input(self, example: InputExample) -> str:
        return "{} Sentence : {}".format(self.name, " ".join(example.tokens))

    def rc_format_input(self, example: InputExample, name) -> str:
        en1_span = [example.entities[0].start, example.entities[0].end]
        en2_span = [example.entities[1].start, example.entities[1].end]
        words = example.tokens
        first, latter, head_first = (en1_span, en2_span, True) if en1_span[0] < en2_span[0] \
            else (en2_span, en1_span, False)

        s = "rc fewrel sentence : " + " ".join(example.tokens)
        s += f" The relationship between {get_span(words, en1_span)} and {get_span(words, en2_span)} is"

        return s.strip()


@register_input_format
class EventInputFormat(BaseInputFormat):
    """
    Input format for event extraction, where an input example contains exactly one trigger.
    """
    name = 'ace2005_event_with_trigger'

    def _format_input(self, example: InputExample) -> str:
        triggers = example.triggers
        assert len(triggers) <= 1
        augmentations = [([(entity.type.natural,)], entity.start, entity.end) for entity in triggers]

        return augment_sentence(example.tokens, augmentations, self.BEGIN_ENTITY_TOKEN, self.SEPARATOR_TOKEN,
                                self.RELATION_SEPARATOR_TOKEN, self.END_ENTITY_TOKEN)


@register_input_format
class SRLInput(BaseInputFormat):
    """
    Input format for SRL, where the predicate is marked.
    """
    name = 'srl_input'

    def _format_input(self, example) -> str:
        try:
            assert len(example.sentence_level_entities) == 1
            start, end = example.sentence_level_entities[0].start, example.sentence_level_entities[0].end
            words = copy.copy(example.tokens)
            words.insert(end, self.END_ENTITY_TOKEN)
            words.insert(start, self.BEGIN_ENTITY_TOKEN)
            return ' '.join(words)
        except:
            return ""
