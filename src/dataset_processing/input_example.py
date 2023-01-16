# Adapted from https://github.com/amazon-science/tanl
from dataclasses import dataclass
from typing import List, Optional, Any, Dict, Union
from torch.utils.data.dataset import Dataset


@dataclass
class EntityType:
    """
    An entity type in a dataset.
    """
    short: str = None
    natural: str = None

    def __hash__(self):
        return hash(self.short)


@dataclass
class RelationType:
    """
    A relation type in a dataset.
    """
    short: str = None
    natural: str = None

    def __hash__(self):
        return hash(self.short)


@dataclass
class Entity:
    """
    An entity in a training/test example.
    """
    start: int
    end: int
    type: Optional[EntityType] = None
    id: Optional[int] = None

    def to_tuple(self):
        return self.type.natural, self.start, self.end

    def __hash__(self):
        return hash((self.id, self.start, self.end))


@dataclass
class Relation:
    """
    An (asymmetric) relation in a training/test example.
    """
    type: RelationType
    head: Entity
    tail: Entity

    def to_tuple(self):
        return self.type.natural, self.head.to_tuple(), self.tail.to_tuple()


@dataclass
class Intent:
    """
    The intent of an utterance.
    """
    short: str = None
    natural: str = None

    def __hash__(self):
        return hash(self.short)


@dataclass
class InputExample:
    """
    A single training/test example.
    """
    id: str
    tokens: List[str]
    dataset: Optional[Dataset] = None


    entities: List[Entity] = None
    relations: List[Relation] = None
    intent: Optional[Intent] = None


    triggers: List[Entity] = None


    sentence_level_entities: List[Entity] = None


    document_id: str = None
    chunk_id: int = None
    offset: int = None
    groups: List[List[Entity]] = None


    belief_state: Union[Dict[str, Any], str] = None
    utterance_tokens: str = None


@dataclass
class CorefDocument:
    """
    A document for the coreference resolution task.
    It has several input examples corresponding to chunks of the document.
    """
    id: str
    tokens: List[str]
    chunks: List[InputExample]
    chunk_centers: List[int]
    groups: List[List[Entity]]


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    input_ids: List[int]
    attention_mask: List[int]
    label_ids: Optional[List[int]] = None
