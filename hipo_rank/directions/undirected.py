from hipo_rank import Similarities


class Undirected:
    def update_directions(self, similarities: Similarities) -> Similarities:
        similarities.sect_to_sect.directions = ["undirected" for _ in similarities.sect_to_sect.directions]

        for sect1_index, sent_sims in enumerate(similarities.sent_to_sect):
            similarities.sent_to_sect[sect1_index].directions = [
                "undirected" for _ in  similarities.sent_to_sect[sect1_index].directions
            ]

        for i, sent_sims in enumerate(similarities.sent_to_sent):
            similarities.sent_to_sent[i].directions = [
                "undirected" for _ in
                similarities.sent_to_sent[i].directions
            ]
        return similarities








