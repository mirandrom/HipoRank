import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from numpy import ndarray
from hipo_rank import Document, Embeddings, SectionEmbedding, SentenceEmbeddings
from typing import List


class SentTransformersEmbedder:
    def __init__(self, model: str):
        self.model = SentenceTransformer(model)

    def _get_sentences_embedding(self, sentences: List[str]) -> ndarray:
        return np.stack(self.model.encode(sentences, show_progress_bar=False))

    def get_embeddings(self, doc: Document) -> Embeddings:
        sentence_embeddings = []
        for section in doc.sections:
            id = section.id
            sentences = section.sentences
            se = self._get_sentences_embedding(sentences)
            sentence_embeddings += [SentenceEmbeddings(id=id, embeddings=se)]
        section_embeddings = [SectionEmbedding(id=se.id, embedding=np.mean(se.embeddings, axis=0))
                              for se in sentence_embeddings]

        return Embeddings(sentence=sentence_embeddings, section=section_embeddings)




