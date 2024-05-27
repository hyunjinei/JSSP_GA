import geneticpython
from objects import *

# from Data.Dataset.Dataset import Dataset
from Data.Adams.abz6.abz6 import Dataset

from geneticpython.models import PermutationIndividual, IntIndividual
from geneticpython import Population
from geneticpython.core.operators import RouletteWheelSelection, OrderCrossover, PointCrossover, FlipBitMutation, RouletteWheelReplacement
from geneticpython import GAEngine
from geneticpython.tools.visualization import plot_single_objective_history
from Config.Run_Config import Run_Config
# from GA_pyGAD.Visualize import show_evolution
from GA_geneticpython.Visualize import show_evolution

from PMXCrossover import PMXCrossover
dataset = Dataset()
# dataset = Dataset('test_3030.txt')
op_data = dataset.op_data

config = Run_Config(dataset.n_job, dataset.n_machine, dataset.n_op,
                    False, False, False, False, False, False)


seed = 42

indv_temp = PermutationIndividual(length=dataset.n_op, start=0)
# indv_temp = IntIndividual(length=900, domains=(0, 900))
population = Population(indv_temp, size=100)
# population = Population(indv_temp, pop_size=100)
selection = RouletteWheelSelection()
# crossover = OrderCrossover(pc=0.8)
# crossover = PMXCrossover(pc=0.8)
crossover = PointCrossover(pc=0.9, n_points=1)
# crossover = PrimCrossover(pc=0.8)
# crossover = SBXCrossover(pc=0.8)
# crossover = UniformCrossover(pc=0.8)
mutation = FlipBitMutation(pm=0.5)
# this function decides which individuals will be survived
replacement = RouletteWheelReplacement()


engine = GAEngine(population, selection=selection,
                  selection_size=10,
                  crossover=crossover,
                  mutation=mutation,
                  replacement=replacement)

evol = []

popul = []
makespan = []
score = [[] for i in range(6)]
@engine.minimize_objective
def fitness(indv):
    raw = indv.chromosome.genes.tolist()
    # ind = Individual(config=config, seq=raw, op_data=op_data)
    index = sorted(range(len(raw)), key=lambda k: raw[k], reverse=False)
    result_sequence = [0 for i in range(len(raw))]

    for i, idx in enumerate(index):
        result_sequence[idx] = i
    ind = Individual(config=config, seq=result_sequence, op_data=op_data)
    evol.append(ind)

    makespan.append(ind.makespan)
    for k in range(6):
        score[k].append(ind.score[k])
    # return ind.makespan / ind.score[5]
    return ind.makespan

# engine.create_seed(seed)
history = engine.run(generations=100)
ans = engine.get_best_indv()
print(ans)
idx = makespan.index(min(makespan))

print(min(makespan))
print([score[k][idx] for k in range(6)])
# plot_single_objective_history({'geneticpython': history})
print('Total Collected Individuals : ',len(makespan))
show_evolution(makespan, score, title='Test3030 / without MIO / min:'+str(min(makespan)))
