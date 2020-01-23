from gensim.models import KeyedVectors
import numpy as np
from collections import defaultdict

from numpy import ndarray
from hipo_rank import Document, Embeddings, SectionEmbedding, SentenceEmbeddings


class W2VEmbedder:
    def __init__(self, bin_path: str):
        self.word_vectors = self._load_vectors(bin_path)
        # float.32 for consistency with w2v type to avoid issues with matrix ops
        self.oov_vec = np.zeros(self.word_vectors.vector_size, dtype=np.float32)
        self.oov = defaultdict(int)

    def _load_vectors(self, bin_path):
        return KeyedVectors.load_word2vec_format(bin_path, binary=True)

    def _get_word_vector(self, word: str) -> ndarray:
        try:
            return self.word_vectors[word]
        except KeyError:
            self.oov[word] += 1
            return self.oov_vec

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




