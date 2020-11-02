from collections import Counter
from pathlib import Path
import json
import h5py

from typing import List, Iterator, Any
from dataclasses import dataclass
from hipo_rank import Document, Section


@dataclass
class CnndmDoc:
    # dataclass wrapper for original CNN_DM dataset format
    article_text: List[str]
    abstract_text: List[str]


class CnndmDataset(object):
    def __init__(self, file_path: str, split_into_n_sections: int = 1):
        self.docs = self._load_docs(file_path)
        self.n = split_into_n_sections

    @staticmethod
    def _load_docs(file_pattern):
        docs = []
        for file_name in Path("").glob(file_pattern):
            with h5py.File(file_name, 'r') as f:
                data = [json.loads(j_str) for j_str in f['dataset']]
                docs_batch = [CnndmDoc(x['article'], x['abstract']) for x in data]
                docs += docs_batch

        def filter_doc(doc: CnndmDoc):
            if len(doc.article_text) == 0:
                return False
            if len(doc.abstract_text) == 0:
                return False
            if all([s == '' for s in doc.article_text]):
                return False
            if all([s == '' for s in doc.abstract_text]):
                return False
            return True

        docs = list(filter(filter_doc, docs))
        return docs

    def _get_sections(self, doc: CnndmDoc) -> List[Section]:
        l = len(doc.article_text)
        s = l // self.n
        return [Section(id=str(i), sentences=doc.article_text[i:i+s])
                for i in range(0, l, s)]

    def _get_reference(self, doc: CnndmDoc) -> List[str]:
        return doc.abstract_text

    def __iter__(self) -> Iterator[Document]:
        for doc in self.docs:
            sections = self._get_sections(doc)
            reference = self._get_reference(doc)
            yield Document(sections=sections, reference=reference)

    def __getitem__(self, i):
        if isinstance(i, int):
            doc = self.docs[i]
            sections = self._get_sections(doc)
            reference = self._get_reference(doc)
            return Document(sections=sections, reference=reference)
        elif isinstance(i, slice):
            docs = self.docs[i]
            sections = [self._get_sections(doc) for doc in docs]
            references = [self._get_reference(doc) for doc in docs]
            return [Document(sections=s, reference=r) for s,r in zip(sections, references)]
