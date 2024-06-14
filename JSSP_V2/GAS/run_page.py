import os
import sys
import random
import copy
import datetime
from multiprocessing import Pool, Manager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GAS.GA import GAEngine
from GAS.Crossover.POXCrossover import POXCrossover
from GAS.Mutation.IntelligentMutation import IntelligentMutation
from GAS.Crossover.UniformCrossover import UniformCrossover
from GAS.Crossover.SequenceSwapMutation import SequenceSwapMutation
from GAS.Selection.TruncationSelection import TruncationSelection
from GAS.Mutation.SelectiveMutation import SelectiveMutation
from Config.Run_Config import Run_Config
from Data.Dataset.Dataset import Dataset
from visualization.Gantt import Gantt
from postprocessing.PostProcessing import generate_machine_log

TARGET_MAKESPAN = 1642
MIGRATION_FREQUENCY = 300
random_seed = 42

def run_ga_engine(args):
    ga_engine, index, result_txt_path, result_gantt_path, ga_generations_path, sync_generation, sync_lock, events = args
    try:
        best, best_crossover, best_mutation, all_generations, execution_time, best_time = ga_engine.evolve(index, sync_generation, sync_lock, events)

        print(f"GA{index+1} 상태:")
        print(f"Population: {ga_engine.population is not None}")
        print(f"Best Individual: {best is not None}")
        print(f"Current Generation: {sync_generation[index]}")

        if best is None:
            return None

        crossover_name = best_crossover.__class__.__name__
        mutation_name = best_mutation.__class__.__name__
        selection_name = ga_engine.selection.__class__.__name__
        local_search_names = [ls.__class__.__name__ for ls in ga_engine.local_search]
        local_search_name = "_".join(local_search_names)
        pso_name = ga_engines[index].pso.__class__.__name__ if ga_engines[index].pso else 'None'
        pc = best_crossover.pc
        pm = best_mutation.pm

        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        log_path = os.path.join(result_txt_path, f'log_GA{index+1}_{now}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_{pso_name}_pc{pc}_pm{pm}.csv')
        machine_log_path = os.path.join(result_txt_path, f'machine_log_GA{index+1}_{now}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_{pso_name}_pc{pc}_pm{pm}.csv')
        generations_path = os.path.join(ga_generations_path, f'ga_generations_GA{index+1}_{now}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_{pso_name}_pc{pc}_pm{pm}.csv')

        if best is not None and hasattr(best, 'monitor'):
            best.monitor.save_event_tracer(log_path)
            ga_engine.config.filename['log'] = log_path
            generated_log_df = generate_machine_log(ga_engine.config)
            generated_log_df.to_csv(machine_log_path, index=False)
            ga_engine.save_csv(all_generations, execution_time, generations_path)
        else:
            print("No valid best individual or monitor to save the event tracer.")

        return best, best_crossover, best_mutation, all_generations, execution_time, best_time, index
    except Exception as e:
        print(f"Exception in GA {index+1}: {e}")
        return None

def main():
    print("Starting main function...")

    island_mode = int(input("Select Island-Parallel GA mode (1: Independent, 2: Sequential Migration, 3: Random Migration): "))
    print(f"Selected Island-Parallel GA mode: {island_mode}")

    file = 'ta21.txt'
    print(f"Loading dataset from {file}...")
    dataset = Dataset(file)

    base_config = Run_Config(
        n_job=20, n_machine=20, n_op=400, population_size=300, generations=200, 
        print_console=False, save_log=True, save_machinelog=True, 
        show_gantt=False, save_gantt=True, show_gui=False,
        trace_object='Process4', title='Gantt Chart for JSSP',
        tabu_search_iterations=10, hill_climbing_iterations=30, simulated_annealing_iterations=50
    )
    
    print("Base config created...")

    base_config.dataset_filename = file
    base_config.target_makespan = TARGET_MAKESPAN
    base_config.island_mode = island_mode

    result_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'result')
    result_txt_path = os.path.join(result_path, 'result_txt')
    result_gantt_path = os.path.join(result_path, 'result_Gantt')
    ga_generations_path = os.path.join(result_path, 'ga_generations')

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(result_txt_path):
        os.makedirs(result_txt_path)
    if not os.path.exists(result_gantt_path):
        os.makedirs(result_gantt_path)
    if not os.path.exists(ga_generations_path):
        os.makedirs(ga_generations_path)

    print("Result directories checked/created...")

    # custom_settings 에서 crossover_assign 추가
    custom_settings = [
        {'crossover_seq': POXCrossover, 'crossover_assign': UniformCrossover, 'pc': 0.7, 'mutation_seq': SequenceSwapMutation, 'mutation_assign': MachineAssignmentMutation, 'pm': 0.05, 'selection': TruncationSelection(),'elite_TS': 0.2, 'local_search': [], 'pso':  None, 'selective_mutation': SelectiveMutation(pm_high=0.5, pm_low=0.01, rank_divide=0.4)},
    ]

    ga_engines = []
    for i, setting in enumerate(custom_settings):
        # crossover_seq와 crossover_assign 인스턴스 생성
        crossover_seq = crossover_seq_class(pc=pc)
        crossover_assign = crossover_assign_class(pc=pc)

        # mutation_seq와 mutation_assign 인스턴스 생성
        mutation_seq = mutation_seq_class(pm=pm)
        mutation_assign = mutation_assign_class(pm=pm) 

        mutation_class = setting['mutation']
        selection_instance = setting['selection']
        selection_instance.elite_TS = setting['elite_TS']
        local_search_methods = setting['local_search']
        pso_class = setting.get('pso')
        selective_mutation_instance = setting['selective_mutation']
        pc = setting['pc']
        pm = setting['pm']

        initialization_mode = input(f"Select Initialization GA mode for GA{i+1} (1: basic, 2: MIO, 3: GifflerThompson, 4: LoadBalanceJSSP, 5: EnergyBasedJSSP): ")
        print(f"Selected Initialization GA mode for GA{i+1}: {initialization_mode}")

        config = copy.deepcopy(base_config)
        config.filename['log'] = os.path.join(result_txt_path, f'GA{i+1}_{config.now}.csv')
        config.filename['machine'] = os.path.join(result_txt_path, f'GA{i+1}_{config.now}_machine.csv')
        config.filename['gantt'] = os.path.join(result_gantt_path, f'GA{i+1}_{config.now}.png')
        config.filename['csv'] = os.path.join(ga_generations_path, f'GA{i+1}_{config.now}.csv')

        config.ga_index = i + 1

        crossover = crossover_class(pc=pc)
        mutation = mutation_class(pm=pm)
        selection = selection_instance
        pso = pso_class if pso_class else None
        local_search = local_search_methods
        local_search_frequency = 300
        selective_mutation_frequency = 300
        selective_mutation = selective_mutation_instance

        # GAEngine 초기화 시 mutation_seq와 mutation_assign 인자 추가
        if initialization_mode == '1':
            ga_engine = GAEngine(config, dataset.op_data, crossover_seq, crossover_assign, mutation_seq, mutation_assign, selection, local_search, pso, selective_mutation, elite_ratio=0, ga_engines=ga_engines, island_mode=island_mode, migration_frequency=MIGRATION_FREQUENCY, local_search_frequency=local_search_frequency, selective_mutation_frequency=selective_mutation_frequency, random_seed=random_seed)
        elif initialization_mode == '2':
            ga_engine = GAEngine(config, dataset.op_data, crossover_seq, crossover_assign, mutation_seq, mutation_assign, selection, local_search, pso, selective_mutation, elite_ratio=0, ga_engines=ga_engines, island_mode=island_mode, migration_frequency=MIGRATION_FREQUENCY, initialization_mode='2', dataset_filename=config.dataset_filename, local_search_frequency=local_search_frequency, selective_mutation_frequency=selective_mutation_frequency, random_seed=random_seed)
        elif initialization_mode == '3':
            ga_engine = GAEngine(config, dataset.op_data, crossover_seq, crossover_assign, mutation_seq, mutation_assign, selection, local_search, pso, selective_mutation, elite_ratio=0, ga_engines=ga_engines, island_mode=island_mode, migration_frequency=MIGRATION_FREQUENCY, initialization_mode='3', dataset_filename=config.dataset_filename, local_search_frequency=local_search_frequency, selective_mutation_frequency=selective_mutation_frequency, random_seed=random_seed)
        
        ga_engines.append(ga_engine)

        print(f"Initialized GAEngine {i+1}")

    best_individuals = [None] * len(ga_engines)
    stop_evolution = Manager().Value('i', 0)
    elite_population = Manager().list([None] * len(ga_engines))

    manager = Manager()
    sync_generation = manager.list([0] * len(ga_engines))
    sync_lock = manager.Lock()

    if island_mode in ['2', '3']:
        events = [manager.Event() for _ in range(len(ga_engines))]

    with Pool() as pool:
        while True:
            if island_mode in ['2', '3']:
                args = [(ga_engines[i], i, result_txt_path, result_gantt_path, ga_generations_path, sync_generation, sync_lock, events) for i in range(len(ga_engines))]
            else:
                args = [(ga_engines[i], i, result_txt_path, result_gantt_path, ga_generations_path, sync_generation, sync_lock, None) for i in range(len(ga_engines))]
            
            results = pool.map(run_ga_engine, args)

            all_completed = True
            for result in results:
                if result is not None:
                    best, best_crossover, best_mutation, all_generations, execution_time, best_time, index = result
                    best_individuals[index] = (best, best_crossover, best_mutation, execution_time, best_time, all_generations)
                    elite_population[index] = best
                    crossover_name = best_crossover.__class__.__name__
                    mutation_name = best_mutation.__class__.__name__
                    selection_name = ga_engines[index].selection.__class__.__name__
                    local_search_names = [ls.__class__.__name__ for ls in ga_engines[index].local_search]
                    local_search_name = "_".join(local_search_names)
                    pso_name = ga_engines[index].pso.__class__.__name__ if ga_engines[index].pso else 'None'
                    pc = best_crossover.pc
                    pm = best_mutation.pm
                    log_path = os.path.join(result_txt_path, f'log_GA{index+1}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_{pso_name}_pc{pc}_pm{pm}.csv')
                    machine_log_path = os.path.join(result_txt_path, f'machine_log_GA{index+1}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_{pso_name}_pc{pc}_pm{pm}.csv')
                    generations_path = os.path.join(ga_generations_path, f'ga_generations_GA{index+1}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_{pso_name}_pc{pc}_pm{pm}.csv')

                    if best is not None and hasattr(best, 'monitor'):
                        best.monitor.save_event_tracer(log_path)
                        ga_engines[index].config.filename['log'] = log_path
                        generated_log_df = generate_machine_log(ga_engines[index].config)
                        generated_log_df.to_csv(machine_log_path, index=False)
                        ga_engines[index].save_csv(all_generations, execution_time, generations_path)
                    else:
                        print("No valid best individual or monitor to save the event tracer.")

                    if best.makespan <= TARGET_MAKESPAN:
                        stop_evolution.value = 1
                        print(f"Stopping early as best makespan {best.makespan} is below target {TARGET_MAKESPAN}.")
                        break

                    if os.path.exists(log_path) and os.path.exists(machine_log_path) and os.path.exists(generations_path):
                        stop_evolution.value = 1
                        print(f"Stopping as all files for GA{index+1} are generated.")
                        break
                else:
                    all_completed = False

            if stop_evolution.value or all_completed:
                break

    for i, result in enumerate(best_individuals):
        if result is not None:
            best, best_crossover, best_mutation, execution_time, best_time, all_generations = result
            crossover_name = best_crossover.__class__.__name__
            mutation_name = best_mutation.__class__.__name__
            selection_name = ga_engines[i].selection.__class__.__name__
            local_search_names = [ls.__class__.__name__ for ls in ga_engines[i].local_search]
            local_search_name = "_".join(local_search_names)
            pso_name = ga_engines[i].pso.__class__.__name__ if ga_engines[i].pso else 'None'
            pc = best_crossover.pc
            pm = best_mutation.pm
            print(f"Best solution for GA{i+1}: {best} using {crossover_name} with pc={pc} and {mutation_name} with pm={pm} and selection: {selection_name} and Local Search: {local_search_name} and pso: {pso_name}, Time taken: {execution_time:.2f} seconds, First best time: {best_time:.2f} seconds")
            machine_log_path = os.path.join(result_txt_path, f'machine_log_GA{i+1}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_{pso_name}_pc{pc}_pm{pm}.csv')
            gantt_path = os.path.join(result_gantt_path, f'gantt_chart_GA{i+1}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_{pso_name}_pc{pc}_pm{pm}.png')
            if os.path.exists(machine_log_path):
                machine_log = pd.read_csv(machine_log_path)
                ga_engines[i].config.filename['gantt'] = gantt_path
                Gantt(machine_log, ga_engines[i].config, best.makespan)
            else:
                print(f"Warning: {machine_log_path} does not exist.")

if __name__ == "__main__":
    main()


            