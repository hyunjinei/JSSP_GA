import sys
import os
import random
from copy import deepcopy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.Crossover.base import Crossover
from GAS.Individual import Individual

class PSXCrossover(Crossover):
    def __init__(self, pc):
        self.pc = pc

    def cross(self, parent1, parent2):
        if random.random() > self.pc:
            return parent1, parent2

        size = len(parent1.seq)
        point1, point2 = sorted(random.sample(range(size), 2))

        # Step 1: Identify a partial schedule in one parent and identify the corresponding part in the other parent
        partial1 = parent1.seq[point1:point2]
        partial2 = []

        # Find the corresponding part in parent2 centered around the same jobs as in parent1
        start_index = None
        for i in range(size):
            if parent2.seq[i] in partial1:
                start_index = i
                break
        if start_index is not None:
            end_index = start_index + len(partial1)
            partial2 = parent2.seq[start_index:end_index]

        # Step 2: Exchange the partial schedules to create proto-offspring
        proto_offspring1 = deepcopy(parent1.seq)
        proto_offspring2 = deepcopy(parent2.seq)
        
        proto_offspring1[point1:point2] = partial2
        proto_offspring2[start_index:end_index] = partial1

        # Step 3: Legalize the proto-offspring by removing excess genes and adding missing genes
        def legalize(proto, original):
            original_set = set(original)
            proto_set = set(proto)
            
            missing_genes = list(original_set - proto_set)
            excess_genes = [gene for gene in proto if proto.count(gene) > 1]

            proto_gene_count = {gene: proto.count(gene) for gene in proto}

            for i in range(len(proto)):
                if proto_gene_count[proto[i]] > 1:
                    proto_gene_count[proto[i]] -= 1
                    if missing_genes:
                        proto[i] = missing_genes.pop(0)
                        proto_gene_count[proto[i]] = proto_gene_count.get(proto[i], 0) + 1

            return proto

        final_offspring1 = legalize(proto_offspring1, parent1.seq)
        final_offspring2 = legalize(proto_offspring2, parent2.seq)



        return Individual(config=parent1.config, seq=final_offspring1, op_data=parent1.op_data), Individual(config=parent2.config, seq=final_offspring2, op_data=parent2.op_data)

# Example usage
# parent1_seq = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# parent2_seq = [3, 1, 7, 8, 4, 6, 9, 5, 2]
# crossover = PSXCrossover(0.9)
# offspring1, offspring2 = crossover.cross(parent1, parent2)
