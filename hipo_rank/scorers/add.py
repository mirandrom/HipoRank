from hipo_rank import Similarities, Scores


class AddScorer:
    # adds sent_to_sect edges for each sentence
    def __init__(self,
                 forward_weight: float = 0.,
                 backward_weight: float = 1.,
                 section_weight: float = 1.,
                 sent_to_sent_weight: float = 0.5,
                 sent_to_sect_weight: float = 0.5,
                 ):
        # TODO: get rid of these god awful variable names
        self.forward_sent_to_sent_weight = forward_weight
        self.forward_sent_to_sect_weight = forward_weight * section_weight
        self.backward_sent_to_sent_weight = backward_weight
        self.backward_sent_to_sect_weight = backward_weight * section_weight

        self.sent_to_sent_weight = sent_to_sent_weight
        self.sent_to_sect_weight = sent_to_sect_weight

    def get_scores(self, similarities: Similarities) -> Scores:
        # build empty scores, indexed by scores[section_index][sentence_index]
        scores = []
        for sent_to_sent in similarities.sent_to_sent:
            if sent_to_sent.pair_indices:
                num_sents = max([x[1] for x in sent_to_sent.pair_indices]) + 1
            else:
                num_sents = 1
            scores.append([0 for _ in range(num_sents)])

        # add sent_to_sent scores
        for sect_index, sent_to_sent in enumerate(similarities.sent_to_sent):
            pids = sent_to_sent.pair_indices
            dirs = sent_to_sent.directions
            sims = sent_to_sent.similarities
            norm_factor = len(scores[sect_index]) - 1
            for ((i,j), dir, sim) in zip(pids, dirs, sims):
                if dir == "forward":
                    scores[sect_index][i] += self.forward_sent_to_sent_weight * sim / norm_factor * self.sent_to_sent_weight
                    scores[sect_index][j] += self.backward_sent_to_sent_weight * sim / norm_factor * self.sent_to_sent_weight
                elif dir == "backward":
                    scores[sect_index][j] += self.forward_sent_to_sent_weight * sim / norm_factor * self.sent_to_sent_weight
                    scores[sect_index][i] += self.backward_sent_to_sent_weight * sim / norm_factor * self.sent_to_sent_weight
                else:
                    scores[sect_index][j] += self.backward_sent_to_sent_weight * sim / norm_factor * self.sent_to_sent_weight
                    scores[sect_index][i] += self.backward_sent_to_sent_weight * sim / norm_factor * self.sent_to_sent_weight

        # add sent_to_sect scores
        for sect_index, sent_to_sect in enumerate(similarities.sent_to_sect):
            pids = sent_to_sect.pair_indices
            dirs = sent_to_sect.directions
            sims = sent_to_sect.similarities
            norm_factor = len(scores)
            for ((i,j), dir, sim) in zip(pids, dirs, sims):
                if dir == "forward":
                    scores[sect_index][i] += self.forward_sent_to_sect_weight * sim / norm_factor * self.sent_to_sect_weight
                elif dir == "backward" or dir == "undirected":
                    scores[sect_index][i] += self.backward_sent_to_sect_weight * sim / norm_factor * self.sent_to_sect_weight

        ranked_scores = []
        sect_global_idx = 0
        for sect_idx, sect_scores in enumerate(scores):
            for sent_idx, sent_score in enumerate(sect_scores):
                ranked_scores.append(
                    (sent_score,
                     sect_idx,
                     sent_idx,
                     sect_global_idx + sent_idx
                     )
                )
            sect_global_idx += len(sect_scores)

        ranked_scores.sort(key=lambda x: x[0], reverse=True)
        return ranked_scores
            
        

