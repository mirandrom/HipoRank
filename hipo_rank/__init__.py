from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from numpy import ndarray


@dataclass
class Section:
    # dataclass wrapper for section in a document and associated text (split into sentences)
    id: str # section name
    sentences: List[str]
    meta: Optional[Dict] = None


@dataclass
class Document:
    # dataclass wrapper for documents yielded by a dataset iterator
    sections: List[Section]
    reference: List[str]
    meta: Optional[Dict] = None


@dataclass
class SentenceEmbeddings:
    # dataclass wrapper for section in a document and associated sentence embeddings
    id: str # section name
    embeddings: ndarray # first dim = number of sentences
    meta: Optional[Dict] = None


@dataclass
class SectionEmbedding:
    # dataclass wrapper for section in a document and associated embedding
    id: str # section name
    embedding: ndarray
    meta: Optional[Dict] = None


@dataclass
class Embeddings:
    # dataclass wrapper for section in a document and associated sentence embeddings
    sentence: List[SentenceEmbeddings]
    section: List[SectionEmbedding]
    meta: Optional[Dict] = None


PairIndices = List[Tuple[int, int]]


@dataclass
class SentenceSimilarities:
    # dataclass wrapper for intrasection similarities (sentence to sentence or sentence to section)
    id: str # section name
    similarities: ndarray
    pair_indices: PairIndices
    directions: List[str]
    meta: Optional[Dict] = None


@dataclass
class SectionSimilarities:
    # dataclass wrapper for inter-section similarities (section to section)
    similarities: ndarray
    pair_indices: PairIndices
    directions: List[str]
    meta: Optional[Dict] = None


@dataclass
class Similarities:
    # dataclass wrapper for similarities in a document
    sent_to_sent: List[SentenceSimilarities]
    sect_to_sect: SectionSimilarities
    sent_to_sect: List[SentenceSimilarities]
    meta: Optional[Dict] = None


score = float
section_idx = int
local_idx = int
global_idx = int
sentence = str
Scores = List[Tuple[score,section_idx,local_idx,global_idx]]
Summary = List[Tuple[sentence, score, section_idx, local_idx, global_idx]]
Reference = List[sentence]





