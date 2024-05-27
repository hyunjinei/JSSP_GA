# run.py

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

TARGET_MAKESPAN = 1234  # 목표 Makespan
MIGRATION_FREQUENCY = 10  # Migration frequency 설정

def main():
    # Dataset and configuration
    dataset = Dataset('abz5.txt')
    config = Run_Config(n_job=10, n_machine=10, n_op=100, population_size=500, generations=200, 
                        print_console=False, save_log=True, save_machinelog=True, 
                        show_gantt=False, save_gantt=True, show_gui=False,
                        trace_object='Process4', title='Gantt Chart for JSSP')

    # Create necessary folders
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

    # 사용자 정의 crossover, mutation 및 selection 설정
    custom_settings = [
        {'crossover': OrderCrossover, 'pc': 0.9, 'mutation': GeneralMutation, 'pm': 0.7, 'selection': TournamentSelection(), 'local_search': None},
        {'crossover': PMXCrossover, 'pc': 0.9, 'mutation': DisplacementMutation, 'pm': 0.7, 'selection': SeedSelection(), 'local_search': None},
        {'crossover': PositionBasedCrossover, 'pc': 0.9, 'mutation': ReciprocalExchangeMutation, 'pm': 0.7, 'selection': RouletteSelection(), 'local_search': None},
        {'crossover': CXCrossover, 'pc': 0.9, 'mutation': InsertionMutation, 'pm': 0.7, 'selection': RouletteSelection(), 'local_search': None}
        # {'crossover': PMXCrossover, 'pc': 0.8, 'mutation': DisplacementMutation, 'pm': 0.5, 'selection': TournamentSelection(tournament_size=3), 'local_search': TabuSearch()},
        # {'crossover': CXCrossover, 'pc': 0.8, 'mutation': ReciprocalExchangeMutation, 'pm': 0.6, 'selection': SeedSelection(k=0.75), 'local_search': SimulatedAnnealing},
        # {'crossover': OrderCrossover, 'pc': 0.9, 'mutation': GeneralMutation, 'pm': 0.5, 'selection': RouletteSelection(), 'local_search': GifflerThompson(priority_rule='SPT')}
    ]

    # Island-Parallel GA mode selection
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

        ga_engines.append(GAEngine(config, dataset.op_data, crossover, mutation, selection, local_search))

    best_individuals = []
    stop_evolution = False  # 진화 멈춤 여부를 추적하는 플래그 추가

    for generation in range(config.generations):
        for i, ga in enumerate(ga_engines):
            best, best_crossover, best_mutation, all_generations, elapsed_time = ga.evolve()
            best_individuals.append((best, best_crossover, best_mutation, elapsed_time))
            crossover_name = best_crossover.__class__.__name__
            mutation_name = best_mutation.__class__.__name__
            selection_name = ga.selection.__class__.__name__
            local_search_name = ga.local_search.__class__.__name__ if ga.local_search else 'None'
            pc = best_crossover.pc
            pm = best_mutation.pm
            log_path = os.path.join(result_txt_path, f'log_GA{i+1}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_pc{pc}_pm{pm}.csv')
            machine_log_path = os.path.join(result_txt_path, f'machine_log_GA{i+1}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_pc{pc}_pm{pm}.csv')
            best.monitor.save_event_tracer(log_path)
            config.filename['log'] = log_path  # Add this line to update config with the correct log path
            generated_log_df = generate_machine_log(config)  # 수정된 부분
            generated_log_df.to_csv(machine_log_path, index=False)

            # 목표 Makespan에 도달하면 멈춤
            if best.makespan <= TARGET_MAKESPAN:
                stop_evolution = True
                print(f"Stopping early as best makespan {best.makespan} is below target {TARGET_MAKESPAN}.")
                break

        if stop_evolution:
            break

    if island_mode != '1' and not stop_evolution:  # If not independent mode and not stopped
        if island_mode == '2':  # Sequential Migration
            migration_order = list(range(len(ga_engines)))
        elif island_mode == '3':  # Random Migration
            migration_order = random.sample(range(len(ga_engines)), len(ga_engines))
        
        for generation in range(config.generations):
            for i, ga in enumerate(ga_engines):
                best, best_crossover, best_mutation, all_generations, elapsed_time = ga.evolve()
                best_individuals[i] = (best, best_crossover, best_mutation, elapsed_time)
                
                if (generation + 1) % MIGRATION_FREQUENCY == 0:
                    next_index = (i + 1) % len(ga_engines) if island_mode == '2' else migration_order[i]
                    best_individuals[next_index] = copy.deepcopy(best_individuals[i])

                # 목표 Makespan에 도달하면 멈춤
                if best.makespan <= TARGET_MAKESPAN:
                    print(f"Stopping early as best makespan {best.makespan} is below target {TARGET_MAKESPAN}.")
                    stop_evolution = True
                    break

            if stop_evolution:
                break
    
    for i, (best, best_crossover, best_mutation, execution_time, all_generations) in enumerate(best_individuals):
        crossover_name = best_crossover.__class__.__name__
        mutation_name = best_mutation.__class__.__name__
        selection_name = ga.selection.__class__.__name__
        local_search_name = ga.local_search.__class__.__name__ if ga.local_search else 'None'
        pc = best_crossover.pc
        pm = best_mutation.pm
        print(f"Best solution for GA{i+1}: {best} using {crossover_name} with pc={pc} and {mutation_name} with pm={pm} and selection: {selection_name}and Local Search: {local_search_name}, Time taken: {execution_time:.2f} seconds")

    # Load machine log to create Gantt chart
    for i, (best, best_crossover, best_mutation, execution_time, all_generations) in enumerate(best_individuals):
        crossover_name = best_crossover.__class__.__name__
        mutation_name = best_mutation.__class__.__name__
        selection_name = ga.selection.__class__.__name__
        local_search_name = ga.local_search.__class__.__name__ if ga.local_search else 'None'
        pc = best_crossover.pc
        pm = best_mutation.pm
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
