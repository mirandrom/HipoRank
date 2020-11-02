from hipo_rank import Scores, Document, Summary
from summa.summarizer import summarize

class TextRankSummarizer:
    def __init__(self, num_words: int = 200, stay_under_num_words: bool = False):
        self.num_words = num_words
        self.stay_under_num_words = stay_under_num_words

    def get_summary(self, doc: Document, sorted_scores: Scores = None) -> Summary:
        sentences = []
        sect_idxs = []
        local_idxs = []
        # flatten data
        for sect_idx, section in enumerate(doc.sections):
            for local_idx, sentence in enumerate(section.sentences):
                sentences.append(sentence)
                sect_idxs.append(sect_idx)
                local_idxs.append(local_idxs)
        sentences = summarize(" ".join(sentences), scores=True, words=self.num_words)

        summary = [(s[0], s[1], 0, 0, 0) for s in sentences]
        return summary



