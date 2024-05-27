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

# Selection
from GAS.Selection.RouletteSelection import RouletteSelection
from GAS.Selection.SeedSelection import SeedSelection
from GAS.Selection.TournamentSelection import TournamentSelection

# Local Search
from Local_Search.HillClimbing import HillClimbing
from Local_Search.TabuSearch import TabuSearch
from Local_Search.SimulatedAnnealing import SimulatedAnnealing
from Local_Search.GifflerThompson import GifflerThompson

from Config.Run_Config import Run_Config
from Data.Dataset.Dataset import Dataset
from visualization.Gantt import Gantt
from postprocessing.PostProcessing import generate_machine_log  # 수정된 부분

from concurrent.futures import ThreadPoolExecutor, as_completed

TARGET_MAKESPAN = 84  # 목표 Makespan
MIGRATION_FREQUENCY = 4  # Migration frequency 설정

# GA 엔진 실행 함수
def run_ga_engine(ga_engine, index, elite=None):
    try:
        best, best_crossover, best_mutation, all_generations, execution_time, best_time = ga_engine.evolve()
        if best is None:
            return None
        return best, best_crossover, best_mutation, all_generations, execution_time, best_time
    except Exception as e:
        print(f"Exception in GA {index+1}: {e}")
        return None

def main():
    dataset = Dataset('test_33.txt')
    config = Run_Config(n_job=3, n_machine=3, n_op=9, population_size=50, generations=10, 
                        print_console=False, save_log=True, save_machinelog=True, 
                        show_gantt=False, save_gantt=True, show_gui=False,
                        trace_object='Process4', title='Gantt Chart for JSSP')

    config.target_makespan = TARGET_MAKESPAN

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

    '''
    crossovers
    [OrderCrossover, PMXCrossover, LOXCrossover, OBC, 
    PositionBasedCrossover, SXX,PSXCrossover]  # Crossover 리스트
    '''

    '''
    mutations 
    [GeneralMutation, DisplacementMutation, InsertionMutation, 
    ReciprocalExchangeMutation,ShiftMutation, InversionMutation]
    '''

    '''
    selection 
    [TournamentSelection(), SeedSelection(), RouletteSelection()]
    '''

    '''
    Local Search
    [HillClimbing(), TabuSearch(), SimulatedAnnealing(), GifflerThompson()] # Local Search 리스트
    GifflerThompson(priority_rule='SPT') -> SPT, LPT, MWR, LWR, MOR, LOR, EDD, FCFS, RANDOM
    '''


    custom_settings = [
        {'crossover': PMXCrossover, 'pc': 0.6, 'mutation': DisplacementMutation, 'pm': 0.8, 'selection': RouletteSelection(), 'local_search': TabuSearch()},
        # {'crossover': LOXCrossover, 'pc': 0.6, 'mutation': InsertionMutation, 'pm': 0.8, 'selection': SeedSelection(), 'local_search': None},
        # {'crossover': OBC, 'pc': 0.6, 'mutation': ReciprocalExchangeMutation, 'pm': 0.8, 'selection': TournamentSelection(), 'local_search': None},
        # {'crossover': OBC, 'pc': 0.6, 'mutation': ReciprocalExchangeMutation, 'pm': 0.8, 'selection': TournamentSelection(), 'local_search': HillClimbing()},
        # {'crossover': OBC, 'pc': 0.6, 'mutation': ReciprocalExchangeMutation, 'pm': 0.8, 'selection': TournamentSelection(), 'local_search': TabuSearch()},
        # {'crossover': OBC, 'pc': 0.6, 'mutation': ReciprocalExchangeMutation, 'pm': 0.8, 'selection': TournamentSelection(), 'local_search': GifflerThompson(priority_rule='LWR')}
        # {'crossover': PositionBasedCrossover, 'pc': 0.6, 'mutation': ShiftMutation, 'pm': 0.8, 'selection': SeedSelection(), 'local_search': None},
        # {'crossover': PSXCrossover, 'pc': 0.6, 'mutation': InversionMutation, 'pm': 0.8, 'selection': TournamentSelection(), 'local_search': None},
        # {'crossover': OrderCrossover, 'pc': 0.6, 'mutation': InversionMutation, 'pm': 0.8, 'selection': TournamentSelection(), 'local_search': None},
        # {'crossover': PMXCrossover, 'pc': 0.6, 'mutation': DisplacementMutation, 'pm': 0.8, 'selection': SeedSelection(), 'local_search': None},
        # {'crossover': PositionBasedCrossover, 'pc': 0.6, 'mutation': ReciprocalExchangeMutation, 'pm': 0.8, 'selection': RouletteSelection(), 'local_search': None},
        # {'crossover': CXCrossover, 'pc': 0.6, 'mutation': InsertionMutation, 'pm': 0.8, 'selection': RouletteSelection(), 'local_search': None}
    ]

    island_mode = input("Select Island-Parallel GA mode (1: Independent, 2: Sequential Migration, 3: Random Migration): ")

    ga_engines = []
    for i, setting in enumerate(custom_settings):
        crossover_class = setting['crossover']
        mutation_class = setting['mutation']
        selection_instance = setting['selection']
        local_search_class = setting['local_search']
        pc = setting['pc']
        pm = setting['pm']

        crossover = crossover_class(pc=pc)
        mutation = mutation_class(pm=pm)
        selection = selection_instance
        local_search = local_search_class if local_search_class else None

        ga_engines.append(GAEngine(config, dataset.op_data, crossover, mutation, selection, local_search, elite_ratio=0.1))

    best_individuals = [None] * len(ga_engines)  # Indexing issue fix
    stop_evolution = False
    elite_population = [None] * len(ga_engines)  # 엘리트 개체를 저장할 리스트 초기화

    for generation in range(config.generations):
        with ThreadPoolExecutor() as executor:  # 병렬 처리
            future_to_index = {executor.submit(run_ga_engine, ga, i, elite_population[i] if elite_population else None): i for i, ga in enumerate(ga_engines)}
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                while best_individuals[index] is None:
                    try:
                        result = future.result()
                        if result is None:
                            print(f"GA {index+1} did not produce a valid result during evolution.")
                            # Re-submit the GA engine task until it produces a valid result
                            future = executor.submit(run_ga_engine, ga_engines[index], index, elite_population[i] if elite_population else None)
                        else:
                            best, best_crossover, best_mutation, all_generations, execution_time, best_time = result
                            best_individuals[index] = (best, best_crossover, best_mutation, execution_time, best_time, all_generations)
                            elite_population[index] = best  # 엘리트 개체 저장
                            crossover_name = best_crossover.__class__.__name__
                            mutation_name = best_mutation.__class__.__name__
                            selection_name = ga_engines[index].selection.__class__.__name__
                            local_search_name = ga_engines[index].local_search.__class__.__name__ if ga_engines[index].local_search else 'None'
                            pc = best_crossover.pc
                            pm = best_mutation.pm
                            log_path = os.path.join(result_txt_path, f'log_GA{index+1}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_pc{pc}_pm{pm}.csv')
                            machine_log_path = os.path.join(result_txt_path, f'machine_log_GA{index+1}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_pc{pc}_pm{pm}.csv')
                            generations_path = os.path.join(ga_generations_path, f'ga_generations_GA{index+1}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_pc{pc}_pm{pm}.csv')
                            
                            if best is not None and hasattr(best, 'monitor'):
                                best.monitor.save_event_tracer(log_path)
                                config.filename['log'] = log_path  # Add this line to update config with the correct log path
                                generated_log_df = generate_machine_log(config)  # 수정된 부분
                                generated_log_df.to_csv(machine_log_path, index=False)
                                ga_engines[index].save_csv(all_generations, execution_time, generations_path)  # CSV 저장
                            else:
                                print("No valid best individual or monitor to save the event tracer.")

                            # 목표 Makespan에 도달하면 멈춤
                            if best.makespan <= TARGET_MAKESPAN:
                                stop_evolution = True
                                print(f"Stopping early as best makespan {best.makespan} is below target {TARGET_MAKESPAN}.")
                                break
                    except Exception as exc:
                        print(f'Exception during evolution: {exc}')

            if stop_evolution:
                break


    if island_mode != '1' and not stop_evolution:  # If not independent mode and not stopped
        if island_mode == '2':  # Sequential Migration
            migration_order = list(range(len(ga_engines)))
        elif island_mode == '3':  # Random Migration
            migration_order = random.sample(range(len(ga_engines)), len(ga_engines))
                        
        for generation in range(config.generations):
            with ThreadPoolExecutor() as executor:
                future_to_index = {executor.submit(run_ga_engine, ga, i): i for i, ga in enumerate(ga_engines)}
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    while best_individuals[index] is None:
                        try:
                            result = future.result()
                            if result is None:
                                print(f"GA {index+1} did not produce a valid result during migration.")
                                future = executor.submit(run_ga_engine, ga_engines[index], index)
                            else:
                                best, best_crossover, best_mutation, all_generations, execution_time, best_time = result
                                best_individuals[index] = (best, best_crossover, best_mutation, execution_time, best_time, all_generations)
                                generations_path = os.path.join(ga_generations_path, f'ga_generations_GA{index+1}_{best_crossover.__class__.__name__}_{best_mutation.__class__.__name__}_{selection_name}_{local_search_name}_pc{pc}_pm{pm}.csv')

                                if (generation + 1) % MIGRATION_FREQUENCY == 0:
                                    next_index = (index + 1) % len(ga_engines) if island_mode == '2' else migration_order[index]
                                    best_individuals[next_index] = copy.deepcopy(best_individuals[index])

                                # 목표 Makespan에 도달하면 멈춤
                                if best.makespan <= TARGET_MAKESPAN:
                                    print(f"Stopping early as best makespan {best.makespan} is below target {TARGET_MAKESPAN}.")
                                    stop_evolution = True
                                    break
                        except Exception as exc:
                            print(f'Exception during migration: {exc}')

                if stop_evolution:
                    break

    # Load machine log to create Gantt chart
    for i, result in enumerate(best_individuals):
        if result is not None:
            best, best_crossover, best_mutation, execution_time, best_time, all_generations = result
            crossover_name = best_crossover.__class__.__name__
            mutation_name = best_mutation.__class__.__name__
            selection_name = ga_engines[i].selection.__class__.__name__
            local_search_name = ga_engines[i].local_search.__class__.__name__ if ga_engines[i].local_search else 'None'
            pc = best_crossover.pc
            pm = best_mutation.pm
            print(f"Best solution for GA{i+1}: {best} using {crossover_name} with pc={pc} and {mutation_name} with pm={pm} and selection: {selection_name} and Local Search: {local_search_name}, Time taken: {execution_time:.2f} seconds, First best time: {best_time:.2f} seconds")
            machine_log_path = os.path.join(result_txt_path, f'machine_log_GA{i+1}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_pc{pc}_pm{pm}.csv')
            gantt_path = os.path.join(result_gantt_path, f'gantt_chart_GA{i+1}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_pc{pc}_pm{pm}.png')
            if os.path.exists(machine_log_path):
                machine_log = pd.read_csv(machine_log_path)
                config.filename['gantt'] = gantt_path
                Gantt(machine_log, config, best.makespan)
            else:
                print(f"Warning: {machine_log_path} does not exist.")

if __name__ == "__main__":
    main()

