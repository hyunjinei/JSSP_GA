# run.py

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from GAS.GA import GAEngine
from Config.Run_Config import Run_Config
from Data.Dataset.Dataset import Dataset
from visualization.Gantt import Gantt
from postprocessing.PostProcessing import generate_machine_log

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

# Local Search
from Local_Search.HillClimbing import HillClimbing
from Local_Search.TabuSearch import TabuSearch
from Local_Search.SimulatedAnnealing import SimulatedAnnealing

import time

def main():
    # Dataset and configuration
    dataset = Dataset('test_55.txt')
    config = Run_Config(n_job=5, n_machine=5, n_op=25, population_size=10, generations=3, 
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

    # 사용자 정의 crossover, mutation, selection 및 local search 설정
    custom_settings = [
        {'crossover': OrderCrossover, 'pc': 0.9, 'mutation': GeneralMutation, 'pm': 0.1, 'selection': RouletteSelection, 'local_search': HillClimbing},
        {'crossover': PMXCrossover, 'pc': 0.8, 'mutation': DisplacementMutation, 'pm': 0.1, 'selection': RouletteSelection, 'local_search': TabuSearch},
        {'crossover': CXCrossover, 'pc': 0.8, 'mutation': ReciprocalExchangeMutation, 'pm': 0.2, 'selection': RouletteSelection, 'local_search': SimulatedAnnealing}
    ]

    # Island-Parallel GA mode selection
    island_mode = input("Select Island-Parallel GA mode (1: Independent, 2: Sequential Migration, 3: Random Migration): ")

    ga_engines = []
    for i, setting in enumerate(custom_settings):
        crossover_class = setting['crossover']
        mutation_class = setting['mutation']
        selection_class = setting['selection']
        local_search_class = setting['local_search']
        pc = setting['pc']
        pm = setting['pm']

        crossover = crossover_class(pc=pc)
        mutation = mutation_class(pm=pm)
        selection = selection_class()
        local_search = local_search_class() if local_search_class else None

        ga_engines.append(GAEngine(config, dataset.op_data, crossover, mutation, selection, local_search))
        
    best_individuals = []
    execution_times = []  # 시간을 기록할 리스트 추가
    for i, ga in enumerate(ga_engines):
        start_time = time.time()  # 시작 시간 기록
        best, best_crossover, best_mutation, all_generations = ga.evolve()  # 모든 세대 데이터를 가져옴
        end_time = time.time()  # 종료 시간 기록
        execution_time = end_time - start_time  # 걸린 시간 계산
        execution_times.append(execution_time)  # 걸린 시간 리스트에 추가

        # 각 GA별 고유한 CSV 파일 경로 설정
        csv_path = os.path.join(ga_generations_path, f'ga_generations_GA{i+1}.csv')
        ga.save_csv(all_generations, execution_time, csv_path)  # 고유한 CSV 파일 저장

        best_individuals.append((best, best_crossover, best_mutation, execution_time)) # 걸린 시간 추가
        crossover_name = best_crossover.__class__.__name__
        mutation_name = best_mutation.__class__.__name__
        selection_name = ga.selection.__class__.__name__
        local_search_name = ga.local_search.__class__.__name__ if ga.local_search else 'None'
        pc = best_crossover.pc
        pm = best_mutation.pm
        log_path = os.path.join(result_txt_path, f'log_GA{i+1}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_pc{pc}_pm{pm}.csv')
        machine_log_path = os.path.join(result_txt_path, f'machine_log_GA{i+1}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_pc{pc}_pm{pm}.csv')
        best.monitor.save_event_tracer(log_path)
        config.filename['log'] = log_path
        generated_log_df = generate_machine_log(config)
        generated_log_df.to_csv(machine_log_path, index=False)
        
    for i, (best, best_crossover, best_mutation, execution_time) in enumerate(best_individuals): # 걸린 시간 추가
        crossover_name = best_crossover.__class__.__name__
        mutation_name = best_mutation.__class__.__name__
        selection_name = ga.selection.__class__.__name__
        local_search_name = ga.local_search.__class__.__name__ if ga.local_search else 'None'
        pc = best_crossover.pc
        pm = best_mutation.pm
        print(f"Best solution for GA{i+1}: {best} using {crossover_name} with pc={pc} and {mutation_name} with pm={pm} and  selection: {selection_name}and Local Search: {local_search_name}, Time taken: {execution_time:.2f} seconds")

    # Load machine log to create Gantt chart
    for i, (best, best_crossover, best_mutation, execution_time) in enumerate(best_individuals): # 걸린 시간 추가
        crossover_name = best_crossover.__class__.__name__
        mutation_name = best_mutation.__class__.__name__
        selection_name = ga.selection.__class__.__name__
        local_search_name = ga.local_search.__class__.__name__ if ga.local_search else 'None'
        pc = best_crossover.pc
        pm = best_mutation.pm
        machine_log_path = os.path.join(result_txt_path, f'machine_log_GA{i+1}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_pc{pc}_pm{pm}.csv')
        if os.path.exists(machine_log_path):
            machine_log = pd.read_csv(machine_log_path)
            gantt_path = os.path.join(result_gantt_path, f'gantt_chart_GA{i+1}_{crossover_name}_{mutation_name}_{selection_name}_{local_search_name}_pc{pc}_pm{pm}.png')
            config.filename['gantt'] = gantt_path
            Gantt(machine_log, config, best.makespan)
        else:
            print(f"Warning: {machine_log_path} does not exist.")

    if __name__ == "__main__":
        main()
