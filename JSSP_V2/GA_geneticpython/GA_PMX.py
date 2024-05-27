from objects import *

from Data.Dataset.Dataset import Dataset
# from Data.Adams.abz5.abz5 import Dataset
# from Data.Adams.abz6.abz6 import Dataset
# from Data.Adams.abz7.abz7 import Dataset
# from Data.Adams.abz8.abz8 import Dataset
# from Data.Adams.abz9.abz9 import Dataset
from Config.Run_Config import Run_Config
from geneticpython.models import PermutationIndividual
from geneticpython import Population
from geneticpython.core.operators import RouletteWheelSelection, RouletteWheelReplacement, SwapMutation
from geneticpython import GAEngine

# from GA_pyGAD.Visualize import show_evolution
from GA_geneticpython.result.MIO.Visualize_spearman import show_evolution

from PMXCrossover import PMXCrossover

# dataset = Dataset()
dataset = Dataset('test_10015.txt')

op_data = dataset.op_data
config = Run_Config(dataset.n_job, dataset.n_machine, dataset.n_op,
                    False, False, False,
                    False, False, False)

seed = 42
n_generation = 20
n_population = 100
n_selection = 40
step_size = 1/ (n_population + n_selection*n_generation)
swap_pairs = 1
p_mutation = 0.8
p_crossover = 0.8

indv_temp = PermutationIndividual(length=dataset.n_op, start=0)
population = Population(indv_temp, size=n_population)
selection = RouletteWheelSelection()
crossover = PMXCrossover(pc=p_crossover)
mutation = SwapMutation(pm=p_mutation, n_points=swap_pairs)
# this function decides which individuals will be survived
replacement = RouletteWheelReplacement()

engine = GAEngine(population,
                  selection=selection,
                  selection_size=n_selection,
                  crossover=crossover,
                  mutation=mutation,
                  replacement=replacement)

evol = []
popul = []
makespan = []
mio_value = []
score = [[] for i in range(6)]

MIO = False
replace_MIO = True
MIO_adaptive = True
MIO_type = 0

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
    mio_value.append(1/ind.score[0])
    # mio_value.append(ind.score[2])

    for k in range(6):
        score[k].append(ind.score[k])
    if MIO:
        if MIO_adaptive:
            upper_bound_1 = makespan[0]
            upper_bound_2 = mio_value[0]
            ratio_1 = 60. + 40. * step_size * len(evol)
            ratio_2 = 100. - ratio_1

            makespan_score = ratio_1 * (ind.makespan / upper_bound_1)
            input_score = ratio_2 * ((1/ind.score[0]) / upper_bound_2)
            # input_score = ratio_2 * ((ind.score[2]) / upper_bound_2)
            return makespan_score + input_score

        else:

            if MIO_type == 2:
                return ind.makespan * 0.01 * ind.score[2]
            elif MIO_type == 5:
                return ind.makespan / ind.score[5]
            else:
                return ind.makespan / ind.score[1]


    else:
        return ind.makespan


history = engine.run(generations=n_generation)
ans = engine.get_best_indv()
# print(ans)
idx = makespan.index(min(makespan))

print(min(makespan))
print('\n\n')
print([score[k][idx] for k in range(6)])
print('Total Collected Individuals : ', len(makespan))
print('Number of replacement:',crossover.num_called)
print('P_mio:',crossover.p_mio)
# # print('upper_bound_1 =', max(makespan))
# # print('upper_bound_2 =',  max(spearman_value))
# print('upper_bound_1 =', makespan[0])
# print('upper_bound_2 =',  spearman_value[0])
# print('ratio_1 =',  0.8 + 0.2 * step_size * len(evol))
# print('ratio_2 =',  1. - 0.8 - 0.2 * step_size * len(evol))
# print()

if MIO:
    if MIO_adaptive:
        show_evolution(makespan, score,
                       title=dataset.name + '_' + str(n_generation) + 'gen' + '_MIO(adaptive)' + str(MIO_type) + '_' + str(
                           min(makespan)),
                       text="* Swap" + str(swap_pairs) + "pair\nP(mutation)=" + str(
                           p_mutation) + "\n" + "P(crossover)=" + str(p_crossover)
                       + "\n" + "Population=" + str(n_population)
                       + "\n" + "# of Selection=" + str(n_selection))

    else:
        show_evolution(makespan, score,
                       title=dataset.name + '_' + str(n_generation) + 'gen' + '_MIO'+str(MIO_type)+'_' + str(min(makespan)),
                       text="* Swap"+str(swap_pairs)+"pair\nP(mutation)="+str(p_mutation)+"\n"+"P(crossover)="+str(p_crossover)
                       + "\n" + "Population=" + str(n_population)
                       + "\n" + "# of Selection=" + str(n_selection))
else:
    if replace_MIO:
        show_evolution(makespan, score,
                       title=dataset.name + '_' + str(n_generation) + 'gen' + '_replace_' + str(min(makespan)),
                       text="* Swap " + str(swap_pairs) + " pair\nP(mutation)=" + str(
                           p_mutation) + "\n" + "P(crossover)=" + str(p_crossover)
                            + "\n" + "Population=" + str(n_population)
                            + "\n" + "# of Selection=" + str(n_selection))
    else:

        show_evolution(makespan, score,
                       title=dataset.name + '_' + str(n_generation) + 'gen' + '_plain_' + str(min(makespan)),
                       text="* Swap " + str(swap_pairs) + " pair\nP(mutation)="+str(p_mutation)+"\n"+"P(crossover)="+str(p_crossover)
                           + "\n" + "Population=" + str(n_population)
                           + "\n" + "# of Selection=" + str(n_selection))
