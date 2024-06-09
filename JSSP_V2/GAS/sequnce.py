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

from concurrent.futures import ThreadPoolExecutor, as_completed

# Individual 클래스를 임포트합니다.
from GAS.Individual import Individual
import matplotlib.pyplot as plt

# Dataset을 로드합니다.
def load_dataset(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    n_jobs, n_machines = map(int, lines[0].strip().split())
    processing_times = [list(map(int, line.strip().split())) for line in lines[1:n_jobs+1]]
    machines = [list(map(int, line.strip().split())) for line in lines[n_jobs+1:]]
    
    return n_jobs, n_machines, processing_times, machines

# 주어진 시퀀스를 평가합니다.
def evaluate_sequence(n_jobs, n_machines, processing_times, machines, sequence):
    job_operation_counter = {i: 0 for i in range(1, n_jobs+1)}
    machine_end_times = {i: 0 for i in range(1, n_machines+1)}
    job_end_times = {i: 0 for i in range(1, n_jobs+1)}

    gantt_chart_data = []

    for job in sequence:
        job_id = job
        operation_idx = job_operation_counter[job_id]
        machine_id = machines[job_id-1][operation_idx]
        processing_time = processing_times[job_id-1][operation_idx]

        start_time = max(job_end_times[job_id], machine_end_times[machine_id])
        end_time = start_time + processing_time

        gantt_chart_data.append((job_id, machine_id, start_time, end_time))

        job_end_times[job_id] = end_time
        machine_end_times[machine_id] = end_time

        job_operation_counter[job_id] += 1

    makespan = max(job_end_times.values())
    return makespan, gantt_chart_data

# Gantt 차트를 생성합니다.
def plot_gantt_chart(gantt_chart_data):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for job_id, machine_id, start_time, end_time in gantt_chart_data:
        ax.barh(machine_id, end_time - start_time, left=start_time, edgecolor='black', height=0.5)
        ax.text(start_time + (end_time - start_time) / 2, machine_id, f'Job {job_id}', 
                va='center', ha='center', color='white', fontsize=8, fontweight='bold')

    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_title('Gantt Chart')
    plt.show()

# 메인 함수
def main():
    dataset_path = 'ta21.txt'
    sequence = [18, 8, 0, 19, 7, 15, 12, 4, 2, 9, 5, 16, 1, 4, 10, 11, 15, 3, 6, 17, 7, 19, 13, 9, 7, 4, 1, 19, 5, 10, 0, 10, 7, 14, 19, 9, 11, 15, 3, 13, 16, 8, 19, 10, 19, 13, 6, 15, 16, 11, 18, 14, 9, 0, 16, 13, 14, 6, 11, 19, 14, 9, 7, 18, 16, 3, 14, 2, 6, 9, 8, 14, 13, 0, 13, 8, 16, 18, 11, 7, 6, 16, 2, 1, 19, 4, 3, 0, 13, 10, 19, 2, 18, 7, 4, 15, 7, 9, 1, 3, 0, 19, 16, 5, 18, 2, 16, 19, 10, 9, 12, 11, 14, 15, 1, 3, 9, 18, 4, 16, 11, 5, 6, 15, 14, 3, 10, 18, 8, 2, 7, 6, 12, 4, 9, 11, 1, 4, 14, 5, 8, 15, 0, 17, 19, 10, 8, 18, 12, 11, 15, 7, 4, 3, 1, 5, 19, 13, 1, 8, 9, 17, 13, 19, 18, 5, 7, 4, 10, 0, 6, 14, 3, 2, 10, 17, 0, 15, 12, 5, 8, 13, 1, 11, 12, 3, 14, 17, 17, 13, 8, 7, 16, 2, 17, 15, 15, 6, 10, 0, 1, 12, 7, 14, 11, 13, 18, 8, 17, 12, 5, 0, 1, 14, 17, 16, 8, 11, 19, 2, 6, 5, 16, 6, 13, 16, 9, 18, 12, 17, 0, 2, 17, 16, 4, 5, 6, 13, 9, 18, 0, 19, 17, 12, 8, 0, 10, 5, 13, 12, 2, 19, 6, 3, 5, 13, 4, 17, 12, 18, 6, 8, 5, 4, 0, 7, 16, 2, 14, 4, 1, 11, 5, 17, 3, 3, 16, 9, 10, 4, 8, 17, 10, 11, 5, 7, 12, 9, 16, 14, 2, 8, 11, 1, 10, 15, 3, 5, 7, 11, 11, 17, 12, 6, 1, 0, 0, 14, 18, 10, 6, 4, 1, 9, 16, 11, 10, 3, 2, 4, 1, 15, 4, 8, 14, 18, 7, 12, 2, 6, 9, 6, 2, 13, 11, 17, 2, 1, 3, 3, 15, 12, 18, 1, 12, 9, 18, 8, 3, 15, 15, 5, 1, 2, 10, 8, 17, 14, 7, 19, 5, 12, 9, 7, 18, 0, 10, 12, 13, 6, 19, 14, 13, 0, 15, 15, 17, 2, 4, 3, 9, 12, 16, 11, 7, 6, 1, 19, 8, 18, 10, 13, 17, 5, 4, 0, 15, 14, 3, 2]
    sequence = [job + 1 for job in sequence]  # 시퀀스의 모든 요소에 1을 더합니다.
    
    n_jobs, n_machines, processing_times, machines = load_dataset(dataset_path)
    makespan, gantt_chart_data = evaluate_sequence(n_jobs, n_machines, processing_times, machines, sequence)
    print(f"Makespan: {makespan}")
    
    # Gantt 차트를 그립니다.
    plot_gantt_chart(gantt_chart_data)

if __name__ == "__main__":
    main()