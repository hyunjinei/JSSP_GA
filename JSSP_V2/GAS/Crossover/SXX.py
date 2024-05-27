import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from GAS.Crossover.base import Crossover
from GAS.Individual import Individual

# Substring exchange crossover
class SXX(Crossover):
    def __init__(self, pc):
        self.pc = pc

    def cross(self, parent1, parent2):
        if random.random() < self.pc:
            point1, point2 = sorted(random.sample(range(len(parent1.seq)), 2))
            substring1 = parent1.seq[point1:point2]
            substring2 = parent2.seq[point1:point2]

            proto_child1 = parent1.seq[:point1] + substring2 + parent1.seq[point2:]
            proto_child2 = parent2.seq[:point1] + substring1 + parent2.seq[point2:]

            offspring1 = self.legalize(proto_child1, parent1.seq)
            offspring2 = self.legalize(proto_child2, parent2.seq)

            return Individual(config=parent1.config, seq=offspring1, op_data=parent1.op_data), Individual(config=parent1.config, seq=offspring2, op_data=parent1.op_data)

        return parent1, parent2

    def legalize(self, proto, original_seq):
        gene_count = {gene: original_seq.count(gene) for gene in original_seq}
        proto_count = {gene: proto.count(gene) for gene in proto}

        missing_genes = [gene for gene, count in gene_count.items() if proto_count.get(gene, 0) < count]
        excess_genes = [gene for gene, count in proto_count.items() if count > gene_count.get(gene, 0)]

        for i, gene in enumerate(proto):
            if proto_count[gene] > gene_count[gene]:
                proto_count[gene] -= 1
                if missing_genes:
                    proto[i] = missing_genes.pop(0)

        return proto
