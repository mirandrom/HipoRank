import numpy as np

from numpy import ndarray
from hipo_rank import Document, Embeddings, SectionEmbedding, SentenceEmbeddings


class RandEmbedder:
    def __init__(self, dim: int = 100):
        self.dim = dim

    def _get_word_vector(self, word: str) -> ndarray:
        return np.random.rand(self.dim)

    def _get_sentence_embedding(self, sentence: str) -> ndarray:
        # no preprocessing necessary for pubmed w2v
        tokens = sentence.split()
        embeddings = np.stack([self._get_word_vector(w) for w in tokens])
        return np.mean(embeddings, axis=0)

    def get_embeddings(self, doc: Document) -> Embeddings:
        sentence_embeddings = []
        for section in doc.sections:
            id = section.id
            sentences = section.sentences
            se = np.stack([self._get_sentence_embedding(s) for s in sentences])
            sentence_embeddings += [SentenceEmbeddings(id=id, embeddings=se)]

        section_embeddings = [SectionEmbedding(id=se.id, embedding=np.mean(se.embeddings, axis=0))
                              for se in sentence_embeddings]

        return Embeddings(sentence=sentence_embeddings, section=section_embeddings)

