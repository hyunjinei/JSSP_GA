# run.py

import os
import sys
import random
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

# Local Search
from Local_Search.HillClimbing import HillClimbing
from Local_Search.TabuSearch import TabuSearch
from Local_Search.SimulatedAnnealing import SimulatedAnnealing

from Config.Run_Config import Run_Config
from Data.Dataset.Dataset import Dataset
from visualization.Gantt import Gantt
from postprocessing.PostProcessing import generate_machine_log  # 수정된 부분

def main():
    # Dataset and configuration
    dataset = Dataset('test_33.txt')
    config = Run_Config(n_job=3, n_machine=3, n_op=9, population_size=50, generations=3, 
                        print_console=False, save_log=True, save_machinelog=True, 
                        show_gantt=False, save_gantt=True, show_gui=False,
                        trace_object='Process4', title='Gantt Chart for JSSP')

    # Create necessary folders
    result_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'result')
    result_txt_path = os.path.join(result_path, 'result_txt')
    result_gantt_path = os.path.join(result_path, 'result_Gantt')
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(result_txt_path):
        os.makedirs(result_txt_path)
    if not os.path.exists(result_gantt_path):
        os.makedirs(result_gantt_path)

    # 사용자 정의 crossover, mutation 및 selection 설정
    custom_settings = [
        {'crossover': OrderCrossover, 'pc': 0.9, 'mutation': GeneralMutation, 'pm': 0.1, 'selection': RouletteSelection},
        {'crossover': PMXCrossover, 'pc': 0.8, 'mutation': GeneralMutation, 'pm': 0.1, 'selection': RouletteSelection},
        {'crossover': OrderCrossover, 'pc': 0.8, 'mutation': GeneralMutation, 'pm': 0.2, 'selection': RouletteSelection},
        {'crossover': OrderCrossover, 'pc': 0.8, 'mutation': GeneralMutation, 'pm': 0.2, 'selection': RouletteSelection}
    ]

    ga_engines = []
    for setting in custom_settings:
        crossover_class = setting['crossover']
        mutation_class = setting['mutation']
        selection_class = setting['selection']
        pc = setting['pc']
        pm = setting['pm']

        crossover = crossover_class(pc=pc)
        mutation = mutation_class(pm=pm)
        selection = selection_class()

        ga_engines.append(GAEngine(config, dataset.op_data, crossover, mutation, selection))
    
    best_individuals = []
    for i, ga in enumerate(ga_engines):
        best, best_crossover, best_mutation = ga.evolve()
        best_individuals.append((best, best_crossover, best_mutation))
        crossover_name = best_crossover.__class__.__name__
        mutation_name = best_mutation.__class__.__name__
        pc = best_crossover.pc
        pm = best_mutation.pm
        log_path = os.path.join(result_txt_path, f'log_GA{i+1}_{crossover_name}_{mutation_name}_pc{pc}_pm{pm}.csv')
        machine_log_path = os.path.join(result_txt_path, f'machine_log_GA{i+1}_{crossover_name}_{mutation_name}_pc{pc}_pm{pm}.csv')
        best.monitor.save_event_tracer(log_path)
        config.filename['log'] = log_path  # Add this line to update config with the correct log path
        generated_log_df = generate_machine_log(config)  # 수정된 부분
        generated_log_df.to_csv(machine_log_path, index=False)
    
    for i, (best, best_crossover, best_mutation) in enumerate(best_individuals):
        crossover_name = best_crossover.__class__.__name__
        mutation_name = best_mutation.__class__.__name__
        pc = best_crossover.pc
        pm = best_mutation.pm
        print(f"Best solution for GA{i+1}: {best} using {crossover_name} with pc={pc} and {mutation_name} with pm={pm}")

    # Load machine log to create Gantt chart
    for i, (best, best_crossover, best_mutation) in enumerate(best_individuals):
        crossover_name = best_crossover.__class__.__name__
        mutation_name = best_mutation.__class__.__name__
        pc = best_crossover.pc
        pm = best_mutation.pm
        machine_log_path = os.path.join(result_txt_path, f'machine_log_GA{i+1}_{crossover_name}_{mutation_name}_pc{pc}_pm{pm}.csv')
        machine_log = pd.read_csv(machine_log_path)
        gantt_path = os.path.join(result_gantt_path, f'gantt_chart_GA{i+1}_{crossover_name}_{mutation_name}_pc{pc}_pm{pm}.png')
        config.filename['gantt'] = gantt_path
        Gantt(machine_log, config, best.makespan)

if __name__ == "__main__":
    main()
