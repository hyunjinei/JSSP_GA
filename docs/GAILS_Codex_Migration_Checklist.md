# GAILS-style JSSP Migration Checklist for Codex

## 0. Purpose

This document is intended to be the first file a Codex agent reads before modifying the repository.

The current repository implements a Job Shop Scheduling Problem (JSSP) genetic algorithm framework with optional local search. The goal is to refactor the current codebase toward a GAILS-style implementation inspired by the Scientific Reports paper:

> GAILS: an effective multi-object job shop scheduler based on genetic algorithm and iterative local search

The requested target is not only to change GA parameters, but also to align the execution structure, operators, and local search behavior with a GAILS-like hybrid GA + ILS design.

---

## 1. Current Repository and Main Code Paths

Repository:

```text
hyunjinei/JSSP_GA
```

Main implementation directory:

```text
JSSP_V2/GAS/
```

Important files:

```text
JSSP_V2/GAS/run.py
JSSP_V2/GAS/GA.py
JSSP_V2/GAS/Population.py
JSSP_V2/GAS/Individual.py
JSSP_V2/Config/Run_Config.py
JSSP_V2/Data/Dataset/Dataset.py
JSSP_V2/GAS/Selection/TournamentSelection.py
JSSP_V2/GAS/Selection/RouletteSelection.py
JSSP_V2/GAS/Crossover/PMX.py
JSSP_V2/GAS/Crossover/JBX_수정필요_없애도될듯.py
JSSP_V2/GAS/Mutation/DisplacementMutation.py
JSSP_V2/GAS/Mutation/InversionMutation.py
JSSP_V2/GAS/Local_Search/TabuSearch.py
```

Auxiliary or experimental implementations also exist under:

```text
JSSP_V2/GA_geneticpython/
JSSP_V2/GA_pyGAD/
JSSP_V2/GAS/Another/
```

For this task, focus first on `JSSP_V2/GAS/`.

---

## 2. Current Code Analysis

### 2.1 `run.py`

Current role:

- Main execution script.
- Loads a dataset.
- Defines `TARGET_MAKESPAN`.
- Creates `Run_Config`.
- Defines GA operator combinations through `custom_settings`.
- Instantiates `GAEngine`.
- Optionally supports island-parallel/migration modes.
- Saves logs, machine logs, generation CSVs, and Gantt charts.

Current effective default configuration:

```python
TARGET_MAKESPAN = 83
MIGRATION_FREQUENCY = 4

dataset = Dataset('test_33.txt')
config = Run_Config(
    n_job=3,
    n_machine=3,
    n_op=9,
    population_size=50,
    generations=10,
    print_console=False,
    save_log=True,
    save_machinelog=True,
    show_gantt=False,
    save_gantt=True,
    show_gui=False,
    trace_object='Process4',
    title='Gantt Chart for JSSP'
)
```

Current default operator setting:

```python
{
    'crossover': PMXCrossover,
    'pc': 0.6,
    'mutation': DisplacementMutation,
    'pm': 0.8,
    'selection': RouletteSelection(),
    'local_search': TabuSearch()
}
```

Current `GAEngine` construction uses:

```python
elite_ratio=0.1
```

Key observations:

- This is a hybrid GA because it combines GA operators with `TabuSearch`.
- It is not currently GAILS-style.
- It uses `PMX + DisplacementMutation + RouletteSelection + TabuSearch`, not `POX/JBX + TournamentSelection + ILS`.
- The script has an outer generation loop and `GAEngine.evolve()` also has its own generation loop. This can cause conceptual confusion because `config.generations` may be applied more than once.
- The island/migration code increases complexity. For GAILS reproduction, first implement and validate a single-population baseline before restoring migration experiments.

Required direction:

- Replace the default setting with a GAILS-like setting.
- Simplify the initial run path so that `config.generations = 400` clearly means exactly 400 GA generations.

---

### 2.2 `Run_Config.py`

Current role:

- Stores problem dimensions: `n_job`, `n_machine`, `n_op`.
- Stores simulation/logging options.
- Stores GA parameters: `population_size`, `generations`.
- Creates output directories and file paths.

Current important attributes:

```python
self.population_size = population_size
self.generations = generations
self.simul_time = 10000
self.dispatch_mode = 'Manual'
```

Key observations:

- `Run_Config` does not currently store all GA parameters such as crossover rate, mutation rate, elite ratio, local search iteration count, or seed.
- Those parameters are scattered across `run.py`, crossover classes, mutation classes, and local search constructors.

Required direction:

- It is acceptable to keep operator parameters in `run.py` for now.
- For reproducible experiments, consider adding fields such as:

```python
self.random_seed
self.crossover_rate
self.mutation_rate
self.elite_ratio
self.local_search_max_iter
self.experiment_name
```

This is optional but recommended for paper-grade experiments.

---

### 2.3 `GA.py`

Current role:

- Defines `GAEngine`.
- Owns the population.
- Runs selection, crossover, mutation, optional local search, elitism, logging, and early stopping.

Current high-level loop:

```python
for generation in range(self.config.generations):
    self.population.evaluate(self.config.target_makespan)

    num_elites = int(self.elite_ratio * len(self.population.individuals))
    elites = sorted(self.population.individuals, key=lambda ind: ind.fitness, reverse=True)[:num_elites]

    self.population.select(self.selection)
    self.population.crossover(self.crossover)
    self.population.mutate(self.mutation)

    if self.local_search:
        for i in range(len(self.population.individuals)):
            optimized_ind = self.local_search.optimize(self.population.individuals[i], self.config)
            self.population.individuals[i] = optimized_ind

    self.population.individuals[:num_elites] = elites
```

Key observations:

- Elitism preserves the top `elite_ratio` by highest fitness.
- Local search is applied to every individual if enabled.
- For `population_size=400` and `ILS max_iter=10000`, applying local search to every individual every generation may be computationally prohibitive.
- The variable `best_fitness` is used to store the best makespan, so the name is misleading.
- Early stopping depends on `config.target_makespan`.

Required direction:

- Rename `best_fitness` to `best_makespan` for clarity.
- Decide whether ILS is applied to:
  - every offspring, or
  - only elite/top-k individuals, or
  - only the current best individual per generation.
- If exact GAILS reproduction is required, document the chosen interpretation.
- If practical runtime is important, apply ILS only to the top fraction of the population.

---

### 2.4 `Population.py`

Current role:

- Initializes population as random permutations of operation IDs.
- Evaluates all individuals.
- Delegates selection, crossover, and mutation.

Current initialization:

```python
self.individuals = [
    Individual(config, seq=random.sample(range(config.n_op), config.n_op), op_data=op_data)
    for _ in range(config.population_size)
]
```

Current mutation method:

```python
def mutate(self, mutation):
    for individual in self.individuals:
        mutation.mutate(individual)
```

Critical issue:

- Current mutation operators return a new `Individual`, but `Population.mutate()` ignores the returned object.
- Therefore, mutation may not actually update the population.

Required fix:

```python
def mutate(self, mutation):
    self.individuals = [
        mutation.mutate(individual)
        for individual in self.individuals
    ]
```

Additional observations:

- `select()` repeatedly calls `selection.select(self.individuals)`.
- If selection returns references to existing individuals, later crossover/mutation behavior should be checked carefully to avoid unintended aliasing.
- `crossover()` assumes an even number of individuals. `population_size=400` is even, so this is fine for GAILS.

---

### 2.5 `Individual.py`

Current role:

- Represents one JSSP solution.
- Converts raw operation permutation into job sequence and machine order.
- Evaluates makespan using a SimPy simulation environment.
- Computes fitness.

Current chromosome representation:

```text
seq = permutation of operation IDs from 0 to n_op - 1
```

Important derived structures:

```python
self.job_seq = self.get_repeatable()
self.feasible_seq = self.get_feasible()
self.machine_order = self.get_machine_order()
self.makespan, self.mio_score = self.evaluate(self.machine_order)
self.calculate_fitness(config.target_makespan)
```

Current fitness:

```python
self.fitness = 1 / (self.makespan / target_makespan)
```

Equivalent:

```text
fitness = target_makespan / makespan
```

Key observations:

- Lower makespan gives higher fitness.
- The objective is still makespan minimization.
- The current representation is operation-permutation based.
- The implementation does not currently use the full GAILS two-part encoding of OS/MS in a strict FJSSP sense.
- For pure JSSP, machine assignment is fixed by the problem data, so OS-focused adaptation is reasonable.

Potential issue:

- `evaluate()` appends to `self.MIO` and `self.MIO_sorted` without clearing them first. If `evaluate()` is called repeatedly on the same object, these lists may accumulate stale values.
- Many operators create new `Individual` objects, which reduces the impact of this issue, but `Population.evaluate()` re-evaluates existing individuals and may trigger accumulation.

Recommended fix before experiments:

```python
def evaluate(self, machine_order):
    self.MIO = []
    self.MIO_sorted = []
    ...
```

or ensure these lists are reset before each evaluation.

---

### 2.6 Selection Operators

#### `RouletteSelection.py`

Current role:

- Selects individuals proportionally to fitness.

Current selection method:

```python
max_fitness = sum(ind.fitness for ind in population)
pick = random.uniform(0, max_fitness)
```

Current default run uses `RouletteSelection()`.

GAILS direction:

- Replace with tournament selection.

#### `TournamentSelection.py`

Current role:

- Selects the best individual among a random subset.

Current constructor:

```python
class TournamentSelection:
    def __init__(self, tournament_size=2):
        self.tournament_size = tournament_size
```

This matches the target GAILS-style tournament size:

```text
k = 2
```

Required change in `run.py`:

```python
selection = TournamentSelection(tournament_size=2)
```

---

### 2.7 Crossover Operators

#### `PMX.py`

Current role:

- Partial-Mapped Crossover for permutation chromosomes.

Current default:

```python
PMXCrossover(pc=0.6)
```

Key observations:

- PMX is not the GAILS target crossover.
- GAILS target for operation sequence should use POX/JBX style crossover, with 50:50 selection between the two if following the previously identified setting.

#### `JBX_수정필요_없애도될듯.py`

Current role:

- Appears to be an experimental Job-Based Crossover implementation.

Key observations:

- Filename indicates it needs modification and may be disposable.
- Do not rely on this file as-is for GAILS implementation.
- Create a clean new crossover wrapper instead.

Required direction:

Create:

```text
JSSP_V2/GAS/Crossover/GAILSCrossover.py
```

Expected behavior:

- Accept `pc=0.8`.
- If crossover is not applied, return parents.
- If crossover is applied, randomly choose POX or JBX with 50:50 probability.
- Return new `Individual` objects.
- Preserve valid operation permutations.
- Ensure no duplicated or missing operation IDs.

Validation rule:

```python
sorted(child.seq) == list(range(config.n_op))
```

---

### 2.8 Mutation Operators

#### `DisplacementMutation.py`

Current role:

- Selects a subsequence and reinserts it elsewhere.
- Current default mutation in `run.py`.

Current default:

```python
DisplacementMutation(pm=0.8)
```

GAILS direction:

- Mutation rate should be changed to `pm=0.1`.
- The exact mutation operator should be selected deliberately.
- A simple neighbor/swap/inversion mutation is acceptable for OS-focused GAILS-like JSSP adaptation.

#### `InversionMutation.py`

Current role:

- Reverses a randomly selected segment.

Potential use:

```python
InversionMutation(pm=0.1)
```

Critical dependency:

- Mutation will only work correctly after fixing `Population.mutate()`.

---

### 2.9 `TabuSearch.py`

Current role:

- Optional local search operator used by current default run.

Current constructor:

```python
class TabuSearch:
    def __init__(self, tabu_size=10, max_iter=100, no_improve_limit=20):
```

Key issues:

1. It is Tabu Search, not Iterative Local Search.
2. `get_neighbors()` swaps sequence positions but only calls `calculate_fitness()` afterward.
3. `calculate_fitness()` uses the existing `makespan`; it does not recompute schedule feasibility or makespan after the swap.
4. Therefore, neighbor evaluation is likely incorrect.
5. `get_random_solution()` references `Individual` and `config` in a way that appears incomplete or undefined in the local scope.

Required direction:

- Do not use `TabuSearch` as the GAILS local search.
- Create a new `IterativeLocalSearch` implementation.
- Every perturbed or neighbor solution must be reconstructed as a new `Individual(config=config, seq=new_seq, op_data=...)` so that makespan is recalculated.

Create:

```text
JSSP_V2/GAS/Local_Search/IterativeLocalSearch.py
```

Expected ILS structure:

```text
1. Start from current individual.
2. Apply local improvement.
3. Perturb the incumbent solution.
4. Apply local improvement again.
5. Accept if improved, or occasionally accept non-improving candidate for diversification.
6. Repeat until max_iter.
```

Target setting:

```text
max_iter = 10000
```

Runtime caution:

- Because `Individual.evaluate()` uses simulation, `max_iter=10000` can be very expensive.
- Start with `max_iter=100` for debugging, then scale to `10000` for final experiments.

---

### 2.10 `Dataset.py`

Current role:

- Loads tab-separated dataset files.
- Reads number of jobs and machines from the first line.
- Builds `op_data` as `(machine_id, processing_time)` tuples.

Current format assumptions:

```python
self.n_job, self.n_machine = map(int, first_line.strip().split('\t'))
self.n_op = self.n_job * self.n_machine
```

Key observations:

- Current `run.py` hardcodes `n_job=3`, `n_machine=3`, `n_op=9` instead of using `dataset.n_job`, `dataset.n_machine`, `dataset.n_op`.
- For benchmark experiments, this should be changed.

Required change in `run.py`:

```python
config = Run_Config(
    n_job=dataset.n_job,
    n_machine=dataset.n_machine,
    n_op=dataset.n_op,
    ...
)
```

---

## 3. Target GAILS-style Configuration

Use this as the target default configuration after refactoring:

```text
population_size = 400
generations = 400
crossover_rate = 0.8
mutation_rate = 0.1
elite_ratio = 0.05
selection = TournamentSelection(tournament_size=2)
crossover = GAILSCrossover(POX/JBX 50:50)
mutation = InversionMutation or swap/neighborhood mutation
local_search = IterativeLocalSearch(max_iter=10000)
objective = minimize makespan
```

Important modeling note:

- The current code is pure JSSP-oriented and operation-permutation based.
- Strict GAILS for FJSSP may use both OS and MS components.
- For this repository, implement a JSSP-compatible GAILS-like version first using the existing operation permutation representation.
- Do not claim strict full GAILS reproduction unless OS/MS encoding and all paper-specific operators are implemented exactly.

---

## 4. Required Implementation Checklist

### Phase 1: Stabilize Current GA Behavior

- [ ] Fix `Population.mutate()` so returned mutated individuals are stored.
- [ ] Confirm that crossover always returns valid `Individual` objects.
- [ ] Confirm that mutation always returns valid `Individual` objects.
- [ ] Add validation utility for permutations:

```python
def is_valid_permutation(seq, n_op):
    return sorted(seq) == list(range(n_op))
```

- [ ] Use the validation utility after crossover and mutation during debugging.
- [ ] Rename misleading variable `best_fitness` in `GA.py` to `best_makespan`.
- [ ] Ensure `Individual.evaluate()` resets `MIO` and `MIO_sorted` before appending.
- [ ] Check whether `Population.evaluate()` unnecessarily re-evaluates newly constructed individuals that were already evaluated in `Individual.__init__()`.

---

### Phase 2: Simplify the Main Run Path

- [ ] Create a clean GAILS-specific runner, recommended path:

```text
JSSP_V2/GAS/run_gails.py
```

- [ ] Keep the existing `run.py` unchanged if backward compatibility is desired.
- [ ] In `run_gails.py`, remove island/migration logic initially.
- [ ] Use exactly one call to `GAEngine.evolve()`.
- [ ] Ensure `generations=400` means exactly 400 generations inside `GAEngine.evolve()`.
- [ ] Use dataset dimensions dynamically:

```python
n_job=dataset.n_job
n_machine=dataset.n_machine
n_op=dataset.n_op
```

- [ ] Add a debug mode with reduced parameters:

```text
population_size = 20
generations = 5
local_search_max_iter = 100
```

- [ ] Add a full experiment mode:

```text
population_size = 400
generations = 400
local_search_max_iter = 10000
```

---

### Phase 3: Implement GAILS-style Crossover

- [ ] Create:

```text
JSSP_V2/GAS/Crossover/GAILSCrossover.py
```

- [ ] Implement wrapper class:

```python
class GAILSCrossover:
    def __init__(self, pc=0.8):
        self.pc = pc

    def cross(self, parent1, parent2):
        ...
```

- [ ] If `random.random() > pc`, return parents unchanged.
- [ ] If crossover is applied, choose POX or JBX with 50:50 probability.
- [ ] Implement POX for operation permutation.
- [ ] Implement JBX for operation permutation.
- [ ] Ensure child sequences have no duplicated operation IDs.
- [ ] Ensure child sequences have no missing operation IDs.
- [ ] Return new `Individual` objects:

```python
return Individual(config=parent1.config, seq=child1_seq, op_data=parent1.op_data), \
       Individual(config=parent2.config, seq=child2_seq, op_data=parent2.op_data)
```

- [ ] Add lightweight tests for POX/JBX validity.

---

### Phase 4: Implement GAILS-style ILS

- [ ] Create:

```text
JSSP_V2/GAS/Local_Search/IterativeLocalSearch.py
```

- [ ] Implement class:

```python
class IterativeLocalSearch:
    def __init__(self, max_iter=10000, perturb_strength=2):
        ...

    def optimize(self, individual, config):
        ...
```

- [ ] Implement perturbation using swap, insertion, or inversion.
- [ ] Implement local improvement using neighborhood search.
- [ ] Every candidate must be reconstructed as a new `Individual` so that makespan is recomputed.
- [ ] Acceptance criterion should prefer lower makespan.
- [ ] Add optional non-improving acceptance probability for diversification.
- [ ] Make `max_iter` configurable.
- [ ] Add debug setting `max_iter=100`.
- [ ] Add final setting `max_iter=10000`.

Runtime safeguard:

- [ ] Consider applying ILS only to the best individual or top 5% of individuals per generation if full-population ILS is too slow.
- [ ] Document the chosen ILS application strategy in experiment logs.

---

### Phase 5: Update GAEngine for Practical GAILS Execution

- [ ] Keep elitism but change ratio to `0.05`.
- [ ] Replace roulette selection with tournament selection.
- [ ] Replace PMX with `GAILSCrossover`.
- [ ] Replace TabuSearch with `IterativeLocalSearch`.
- [ ] Replace current default mutation rate with `pm=0.1`.
- [ ] Decide whether mutation operator should be:
  - [ ] `InversionMutation(pm=0.1)`, or
  - [ ] new `SwapMutation(pm=0.1)`, or
  - [ ] GAILS-specific neighbor mutation.
- [ ] Ensure local search is applied after mutation and before final elite preservation, or document a different order if changed.
- [ ] Log best makespan per generation.
- [ ] Log average makespan per generation.
- [ ] Log operator setting in output filenames or metadata.

---

### Phase 6: Experiment Reproducibility

- [ ] Add random seed support.
- [ ] Set seeds for `random` and `numpy`.
- [ ] Store experiment metadata in a JSON or CSV file.
- [ ] Include at least:

```text
dataset name
n_job
n_machine
n_op
population size
generation count
crossover type
crossover rate
mutation type
mutation rate
selection type
elite ratio
local search type
local search max_iter
random seed
best makespan
execution time
```

- [ ] Ensure benchmark dataset path is not hardcoded to only `test_33.txt`.
- [ ] Add command-line arguments if feasible:

```bash
python run_gails.py --dataset test_33.txt --mode debug
python run_gails.py --dataset ft10.py --mode full
```

---

## 5. Suggested New File Structure

Recommended additions:

```text
JSSP_V2/GAS/run_gails.py
JSSP_V2/GAS/Crossover/GAILSCrossover.py
JSSP_V2/GAS/Local_Search/IterativeLocalSearch.py
JSSP_V2/GAS/Mutation/SwapMutation.py                 # optional
JSSP_V2/GAS/utils/validation.py                       # optional
JSSP_V2/GAS/utils/experiment_logger.py                # optional
```

Recommended not to modify initially:

```text
JSSP_V2/GA_geneticpython/
JSSP_V2/GA_pyGAD/
```

These appear to be alternative or experimental GA implementations and should not be mixed into the first GAILS migration unless necessary.

---

## 6. Minimal GAILS-style Runner Skeleton

Codex may create `JSSP_V2/GAS/run_gails.py` with this basic structure:

```python
import os
import sys
import random
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.GA import GAEngine
from GAS.Crossover.GAILSCrossover import GAILSCrossover
from GAS.Mutation.InversionMutation import InversionMutation
from GAS.Selection.TournamentSelection import TournamentSelection
from GAS.Local_Search.IterativeLocalSearch import IterativeLocalSearch
from Config.Run_Config import Run_Config
from Data.Dataset.Dataset import Dataset


def main():
    random_seed = 42
    random.seed(random_seed)
    np.random.seed(random_seed)

    dataset = Dataset('test_33.txt')

    debug_mode = True

    if debug_mode:
        population_size = 20
        generations = 5
        ils_max_iter = 100
    else:
        population_size = 400
        generations = 400
        ils_max_iter = 10000

    config = Run_Config(
        n_job=dataset.n_job,
        n_machine=dataset.n_machine,
        n_op=dataset.n_op,
        population_size=population_size,
        generations=generations,
        print_console=False,
        save_log=True,
        save_machinelog=True,
        show_gantt=False,
        save_gantt=True,
        show_gui=False,
        trace_object='Process4',
        title='GAILS-style JSSP'
    )

    config.target_makespan = 83
    config.random_seed = random_seed

    crossover = GAILSCrossover(pc=0.8)
    mutation = InversionMutation(pm=0.1)
    selection = TournamentSelection(tournament_size=2)
    local_search = IterativeLocalSearch(max_iter=ils_max_iter)

    ga_engine = GAEngine(
        config=config,
        op_data=dataset.op_data,
        crossover=crossover,
        mutation=mutation,
        selection=selection,
        local_search=local_search,
        elite_ratio=0.05
    )

    best, best_crossover, best_mutation, all_generations, execution_time, best_time = ga_engine.evolve()

    print('Best individual:', best)
    if best is not None:
        print('Best makespan:', best.makespan)
    print('Execution time:', execution_time)
    print('First best time:', best_time)


if __name__ == '__main__':
    main()
```

---

## 7. Acceptance Criteria

The migration can be considered successful when all of the following are true:

- [ ] `run_gails.py` runs without import errors.
- [ ] Debug mode completes on `test_33.txt`.
- [ ] Full mode can start with:

```text
population_size = 400
generations = 400
pc = 0.8
pm = 0.1
elite_ratio = 0.05
TournamentSelection(k=2)
GAILSCrossover(POX/JBX 50:50)
IterativeLocalSearch(max_iter=10000)
```

- [ ] Mutation actually changes individuals when applied.
- [ ] Crossover children are valid permutations.
- [ ] Local search candidates recompute makespan after sequence changes.
- [ ] Best makespan per generation is logged.
- [ ] Final output includes best makespan and runtime.
- [ ] Code clearly distinguishes:
  - current/original hybrid GA, and
  - new GAILS-style GA.

---

## 8. Important Cautions

- Do not simply change `population_size`, `generations`, `pc`, and `pm` and call it GAILS.
- The current local search is Tabu Search and has neighbor evaluation issues.
- The current mutation method in `Population.py` likely ignores returned mutated individuals.
- The current execution path has nested generation logic in `run.py` and `GAEngine.evolve()`.
- Full `population_size=400`, `generations=400`, `ILS max_iter=10000` may be very slow due to simulation-based evaluation.
- Implement and validate with debug settings first.
- Only claim strict paper reproduction if exact encoding, operators, objective functions, and benchmark settings are verified against the paper.

---

## 9. Recommended Work Order for Codex

1. Read this document completely.
2. Inspect `JSSP_V2/GAS/run.py`, `GA.py`, `Population.py`, and `Individual.py`.
3. Fix `Population.mutate()`.
4. Add/reset safeguards in `Individual.evaluate()`.
5. Create `GAILSCrossover.py`.
6. Create `IterativeLocalSearch.py`.
7. Create `run_gails.py`.
8. Run debug mode.
9. Validate permutation correctness after crossover/mutation.
10. Validate that local search recomputes makespan.
11. Increase parameters gradually.
12. Only after stable debug execution, run full GAILS-style setting.
