from hipo_rank import Scores, Document, Summary
import rouge


class OracleSummarizer:
    def __init__(self,
                 metric: str = 'rouge-l',
                 prf: str = 'f',
                 num_words: int = 200,
                 stay_under_num_words: bool = False
                 ):
        self.evaluator = rouge.Rouge(metrics=[metric],
                                     alpha=0.5,  # Default F1_score
                                     stemming=True)
        self.metric = metric
        self.prf = prf
        self.num_words = num_words
        self.stay_under_num_words = stay_under_num_words

    def get_summary(self, doc: Document,
                    sorted_scores: Scores = None) -> Summary:
        # build dictionaries for easy book-keeping
        sentences = {}
        indices = {}
        global_idx = 0
        for sect_idx, section in enumerate(doc.sections):
            for local_idx, sentence in enumerate(section.sentences):
                sentences[global_idx] = sentence
                indices[global_idx] = (sect_idx, local_idx, global_idx)
                global_idx += 1

        ref = "\n".join(doc.reference) # reference summary
        c = ""  # candidate summary
        summary = []
        num_words = 0
        while True:
            scores = {
                i: self.evaluator.get_scores([f'{c}{s}\n'],[ref])[self.metric][self.prf]
                for i,s in sentences.items()
            }
            i = max(scores, key=scores.get)
            sentence = sentences.pop(i)
            c = f'{c}{sentence}\n'
            sentence_indices = indices.pop(i)
            num_words += len(sentence.split())
            if self.stay_under_num_words and num_words > self.num_words:
                break
            summary.append((sentence, scores[i], *sentence_indices))
            if num_words >= self.num_words:
                break
            if len(sentences) == 0:
                break
        return summary
