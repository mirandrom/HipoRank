from collections import Counter
from pathlib import Path
import json

from typing import List, Iterator, Any
from dataclasses import dataclass
from hipo_rank import Document, Section


@dataclass
class PubmedDoc:
    # dataclass wrapper for original pubmed dataset format
    article_id: str
    article_text: List[str]
    abstract_text: List[str]
    labels: Any
    section_names: List[str]
    sections: List[List[str]]


class PubmedDataset(object):
    def __init__(self, file_path, no_sections: bool = False):
        self._file_path = file_path
        self.no_sections = no_sections

    def _get_sections(self, doc: PubmedDoc) -> List[Section]:
        if self.no_sections:
            sentences = sum([s for s in doc.sections if s != ['']], [])
            return [Section(id="no_sections", sentences=sentences)]
        # handles edge case where sections have the same name
        section_names = []
        sn_counter = Counter(doc.section_names)
        for sn in reversed(doc.section_names):
            c = sn_counter[sn]
            if c > 1:
                section_names = [f"{sn}_{c}"] + section_names
                sn_counter[sn] -= 1
            else:
                section_names = [sn] + section_names
        sections = [Section(id=n, sentences=s) for n, s in zip(section_names, doc.sections) if s!=['']]
        return sections

    def _get_reference(self, doc: PubmedDoc) -> List[str]:
        # remove sentence tags in abstract which break rouge
        return [s.replace("<S>", "").replace("<S\>", "") for s in doc.abstract_text]

    def __iter__(self) -> Iterator[Document]:
        docs = [PubmedDoc(**json.loads(l)) for l in Path(self._file_path).read_text().split("\n") if l]
        for doc in docs:
            sections = self._get_sections(doc)
            reference = self._get_reference(doc)
            yield Document(sections=sections, reference=reference)



