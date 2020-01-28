from hipo_rank import Similarities


class EdgeBased:
    def __init__(self, u: float = 1.0):
        self.u = u # how much to weigh distance from end relative to distance from start

    def _get_direction(self,  node1_index: int, node2_index: int, num_nodes: int) -> str:
        dist_i = min(node1_index, self.u * (num_nodes - node1_index - 1))
        dist_j = min(node2_index, self.u * (num_nodes - node2_index - 1))
        if dist_i > dist_j:
            return "forward"
        elif dist_i < dist_j:
            return "backward"
        else:
            return "undirected"

    def update_directions(self, similarities: Similarities) -> Similarities:
        # handle single section cases
        if len(similarities.sect_to_sect.pair_indices):
            num_nodes = max([x[1] for x in similarities.sect_to_sect.pair_indices]) + 1
        else:
            num_nodes = 1

        sect_directions = [self._get_direction(*x, num_nodes) for x in similarities.sect_to_sect.pair_indices]
        similarities.sect_to_sect.directions = sect_directions

        for sect1_index, sent_sims in enumerate(similarities.sent_to_sect):
            sect2_indices = [x[1] for x in sent_sims.pair_indices]
            similarities.sent_to_sect[sect1_index].directions = [self._get_direction(sect1_index, s2i, num_nodes) for s2i in sect2_indices]

        for i, sent_sims in enumerate(similarities.sent_to_sent):
            if len(sent_sims.pair_indices):
                num_nodes = max([x[1] for x in sent_sims.pair_indices]) + 1
                similarities.sent_to_sent[i].directions = [self._get_direction(*x, num_nodes) for x in sent_sims.pair_indices]

        return similarities
