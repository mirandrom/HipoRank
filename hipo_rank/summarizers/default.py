from hipo_rank import Scores, Document, Summary


class DefaultSummarizer:
    def __init__(self, num_words: int = 200, stay_under_num_words: bool = False):
        self.num_words = num_words
        self.stay_under_num_words = stay_under_num_words

    def get_summary(self, doc: Document, sorted_scores: Scores) -> Summary:
        num_words = 0
        summary = []
        i = 0
        while True:
            sect_idx = sorted_scores[i][1]
            local_idx = sorted_scores[i][2]
            sentence = doc.sections[sect_idx].sentences[local_idx]
            num_words += len(sentence.split())
            if self.stay_under_num_words and num_words > self.num_words:
                break
            summary.append((sentence, *sorted_scores[i]))
            i += 1
            if num_words >= self.num_words:
                break
            if i >= len(sorted_scores):
                break
        return summary



