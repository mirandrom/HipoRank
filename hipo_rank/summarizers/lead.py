from hipo_rank import Scores, Document, Summary


class LeadSummarizer:
    def __init__(self, num_words: int = 200, stay_under_num_words: bool = False):
        self.num_words = num_words
        self.stay_under_num_words = stay_under_num_words

    def get_summary(self, doc: Document, sorted_scores: Scores = None) -> Summary:
        num_words = 0
        summary = []
        global_idx = 0
        score = 0
        for sect_idx, section in enumerate(doc.sections):
            for local_idx, sentence in enumerate(section.sentences):
                num_words += len(sentence.split())
                if self.stay_under_num_words and num_words > self.num_words:
                    break
                summary.append((sentence, score, sect_idx, local_idx, global_idx))
                if num_words >= self.num_words:
                    break
                global_idx += 1
        return summary



