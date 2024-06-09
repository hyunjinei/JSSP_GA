import os
import sys
import random
import time
import copy
import csv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from GAS.GA import GAEngine
# Crossover
from GAS.Crossover.PMX import PMXCrossover
from GAS.Crossover.CX import CXCrossover
from GAS.Crossover.LOX import LOXCrossover
from GAS.Crossover.OrderBasedCrossover import OBC
from GAS.Crossover.PositionBasedCrossover import PositionBasedCrossover
from GAS.Crossover.SXX import SXX
from GAS.Crossover.PSX import PSXCrossover
from GAS.Crossover.OrderCrossover import OrderCrossover

# Mutation
from GAS.Mutation.GeneralMutation import GeneralMutation
from GAS.Mutation.DisplacementMutation import DisplacementMutation
from GAS.Mutation.InsertionMutation import InsertionMutation
from GAS.Mutation.ReciprocalExchangeMutation import ReciprocalExchangeMutation
from GAS.Mutation.ShiftMutation import ShiftMutation
from GAS.Mutation.InversionMutation import InversionMutation
from GAS.Mutation.SwapMutation import SwapMutation

# Selection
from GAS.Selection.RouletteSelection import RouletteSelection
from GAS.Selection.SeedSelection import SeedSelection
from GAS.Selection.TournamentSelection import TournamentSelection

# Local Search
from Local_Search.HillClimbing import HillClimbing
from Local_Search.TabuSearch import TabuSearch
from Local_Search.SimulatedAnnealing import SimulatedAnnealing
from Local_Search.GifflerThompson_LS import GifflerThompson_LS

# Meta Heuristic
from Meta.PSO import PSO  # pso를 추가합니다

# 선택 mutation 
from GAS.Mutation.SelectiveMutation import SelectiveMutation

from Config.Run_Config import Run_Config
from Data.Dataset.Dataset import Dataset
from visualization.Gantt import Gantt
from postprocessing.PostProcessing import generate_machine_log  # 수정된 부분

# from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor, as_completed

'''
txt : TARGET_MAKESPAN, Jobs, Machines

la01: 666  10, 5/  la11: 1222  20, 5
la02: 655  10, 5/  la12: 1039  20, 5
la03: 597  10, 5/  la13: 1150  20, 5
la04: 590  10, 5/  la14: 1292  20, 5
la05: 593  10, 5/  la15: 1207  20, 5
la06: 926  15, 5/  la16: 945   10, 10
la07: 890  15, 5/  la17: 784   10, 10
la08: 863  15, 5/  la18: 848   10, 10
la09: 951  15, 5/  la19: 842   10, 10
la10: 958  15, 5/  la20: 902   10, 10

ta21: 1642 20 20/  ta51: 2760 50 15
ta22: 1561 1600 20 20/  ta52: 2756 50 15
ta31: 1764 30 15/  ta61: 2868 50 20
ta32: 1774 1784 30 15/  ta62: 2869 50 20
ta41: 1906 2005 30 20/  ta71: 5464 100 20
ta42: 1884 1937 30 20/  ta72: 5181 100 20

abz5 = 1234  10, 10
ft20 = 1165
'''
TARGET_MAKESPAN = 83  # 목표 Makespan
MIGRATION_FREQUENCY = 1  # Migration frequency 설정
random_seed = 42 # Population 초기화시 일정하게 만들기 위함. None을 넣으면 아예 랜덤 생성(GA들끼리 같지않음)

def run_ga_engine(config, dataset, crossover_class, mutation_class, selection_instance, local_search_methods, pso_class, selective_mutation_instance, index, initialization_mode, island_mode, elite=None):
    try:
        crossover = crossover_class(pc=0.8)
        mutation = mutation_class(pm=0.1)
        selection = selection_instance
        pso = pso_class if pso_class else None
        local_search = local_search_methods
        local_search_frequency = 1
        selective_mutation_frequency = 20
        selective_mutation = selective_mutation_instance

        if initialization_mode == '1':
            ga_engine = GAEngine(config, dataset.op_data, crossover, mutation, selection, local_search, pso, selective_mutation, elite_ratio=0.05, island_mode=island_mode, migration_frequency=MIGRATION_FREQUENCY, local_search_frequency=local_search_frequency, selective_mutation_frequency=selective_mutation_frequency, random_seed=random_seed)
        elif initialization_mode == '2':
            ga_engine = GAEngine(config, dataset.op_data, crossover, mutation, selection, local_search, pso, selective_mutation, elite_ratio=0.05, island_mode=island_mode, migration_frequency=MIGRATION_FREQUENCY, initialization_mode='2', dataset_filename=config.dataset_filename, local_search_frequency=local_search_frequency, selective_mutation_frequency=selective_mutation_frequency, random_seed=random_seed)
        elif initialization_mode == '3':
            ga_engine = GAEngine(config, dataset.op_data, crossover, mutation, selection, local_search, pso, selective_mutation, elite_ratio=0.05, island_mode=island_mode, migration_frequency=MIGRATION_FREQUENCY, initialization_mode='3', dataset_filename=config.dataset_filename, local_search_frequency=local_search_frequency, selective_mutation_frequency=selective_mutation_frequency, random_seed=random_seed)

        best, best_crossover, best_mutation, all_generations, execution_time, best_time = ga_engine.evolve()
        if best is None:
            return index, None, None, None, None, None, None

        return index, best, best_crossover, best_mutation, all_generations, execution_time, best_time
    except Exception as e:
        print(f"Exception in GA {index+1}: {e}")
        return index, None, None, None, None, None, None

def main():
    initialization_mode = input("Select Initialization GA mode (1: basic, 2: MIO, 3: GifflerThompson): ")
    print(f"Selected Initialization GA mode: {initialization_mode}")
    
    island_mode = input("Select Island-Parallel GA mode (1: Independent, 2: Sequential Migration, 3: Random Migration): ")
    print(f"Selected Island-Parallel GA mode: {island_mode}")

    file = 'la01.txt'
    dataset = Dataset(file)
    config = Run_Config(n_job=10, n_machine=5, n_op=50, population_size=50, generations=10, 
                        print_console=False, save_log=True, save_machinelog=True, 
                        show_gantt=False, save_gantt=True, show_gui=False,
                        trace_object='Process4', title='Gantt Chart for JSSP',
                        tabu_search_iterations=10, hill_climbing_iterations=10, simulated_annealing_iterations=10)

    config.dataset_filename = file  
    config.target_makespan = TARGET_MAKESPAN
    config.island_mode = island_mode  

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

    custom_settings = [
        {'crossover': OrderCrossover, 'pc': 0.8, 'mutation': InsertionMutation, 'pm': 0.1, 'selection': TournamentSelection(), 'local_search': [], 'pso': None, 'selective_mutation': SelectiveMutation(pm_high=0.5, pm_low=0.01, rank_divide=0.4)},
        {'crossover': PMXCrossover, 'pc': 0.8, 'mutation': InsertionMutation, 'pm': 0.1, 'selection': TournamentSelection(), 'local_search': [], 'pso': None, 'selective_mutation': SelectiveMutation(pm_high=0.5, pm_low=0.01, rank_divide=0.4)},
    ]

    ga_engines = []
    for i, setting in enumerate(custom_settings):
        ga_engines.append(setting)

    best_individuals = [None] * len(ga_engines)
    stop_evolution = False
    elite_population = [None] * len(ga_engines)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_ga_engine, config, dataset, ga_engines[i]['crossover'], ga_engines[i]['mutation'], ga_engines[i]['selection'], ga_engines[i]['local_search'], ga_engines[i]['pso'], ga_engines[i]['selective_mutation'], i, initialization_mode, island_mode, elite_population[i]) for i in range(len(ga_engines))]
        for future in as_completed(futures):
            try:
                index, best, best_crossover, best_mutation, all_generations, execution_time, best_time = future.result()
                if best is None:
                    print(f"GA {index+1} did not produce a valid result during evolution.")
                else:
                    best_individuals[index] = (index, best, best_crossover, best_mutation, execution_time, best_time, all_generations)
                    elite_population[index] = best
                    crossover_name = best_crossover.__class__.__name__
                    mutation_name = best_mutation.__class__.__name__
                    selection_name = ga_engines[index]['selection'].__class__.__name__
                    local_search_names = [ls.__class__.__name__ for ls in ga_engines[index]['local_search']]
                    local_search_name = "_".join(local_search_names)
                    pso_name = ga_engines[index]['pso'].__class__.__name__ if ga_engines[index]['pso'] else 'None'
                    pc = best_crossover.pc
                    pm = best_mutation.pm

                    timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
                    log_path = os.path.join(result_txt_path, f'{timestamp}_GA{index+1}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_{pso_name}_pc{pc}_pm{pm}.csv')
                    machine_log_path = os.path.join(result_txt_path, f'{timestamp}_machine_GA{index+1}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_{pso_name}_pc{pc}_pm{pm}.csv')
                    generations_path = os.path.join(ga_generations_path, f'{timestamp}_ga_generations_GA{index+1}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_{pso_name}_pc{pc}_pm{pm}.csv')
                    
                    if best is not None and hasattr(best, 'monitor'):
                        best.monitor.save_event_tracer(log_path)
                        config.filename['log'] = log_path
                        generated_log_df = generate_machine_log(config)
                        generated_log_df.to_csv(machine_log_path, index=False)
                        ga_engines[index]['ga_engine'].save_csv(all_generations, execution_time, generations_path)
                    else:
                        print("No valid best individual or monitor to save the event tracer.")

                    if best.makespan <= TARGET_MAKESPAN:
                        stop_evolution = True
                        print(f"Stopping early as best makespan {best.makespan} is below target {TARGET_MAKESPAN}.")
                        break

                    if os.path.exists(log_path) and os.path.exists(machine_log_path) and os.path.exists(generations_path):
                        stop_evolution = True
                        print(f"Stopping as all files for GA{index+1} are generated.")
                        break

            except Exception as exc:
                print(f'Exception during evolution: {exc}')

    for i, result in enumerate(best_individuals):
        if result is not None:
            index, best, best_crossover, best_mutation, execution_time, best_time, all_generations = result
            crossover_name = best_crossover.__class__.__name__
            mutation_name = best_mutation.__class__.__name__
            selection_name = ga_engines[index]['selection'].__class__.__name__
            local_search_names = [ls.__class__.__name__ for ls in ga_engines[index]['local_search']]
            local_search_name = "_".join(local_search_names)
            pso_name = ga_engines[index]['pso'].__class__.__name__ if ga_engines[index]['pso'] else 'None'
            pc = best_crossover.pc
            pm = best_mutation.pm
            print(f"Best solution for GA{index+1}: {best} using {crossover_name} with pc={pc} and {mutation_name} with pm={pm} and selection: {selection_name} and Local Search: {local_search_name} and pso: {pso_name}, Time taken: {execution_time:.2f} seconds, First best time: {best_time:.2f} seconds")
            timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
            machine_log_path = os.path.join(result_txt_path, f'{timestamp}_machine_GA{index+1}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_{pso_name}_pc{pc}_pm{pm}.csv')
            gantt_path = os.path.join(result_gantt_path, f'{timestamp}_gantt_chart_GA{index+1}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_{pso_name}_pc{pc}_pm{pm}.png')
            if os.path.exists(machine_log_path):
                machine_log = pd.read_csv(machine_log_path)
                config.filename['gantt'] = gantt_path
                Gantt(machine_log, config, best.makespan)
            else:
                print(f"Warning: {machine_log_path} does not exist.")

if __name__ == "__main__":
    main()