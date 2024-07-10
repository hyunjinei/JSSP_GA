import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from RLDataset_FJSSP import RLDataset_FJSSP

def calculate_makespan(solution, dataset):
    job_start_times = {job: 0 for job in range(dataset.n_job)}
    machine_avail_times = {machine: 0 for machine in range(dataset.n_machine)}
    makespan = 0

    for job, op, machine, duration in solution:
        start_time = max(job_start_times[job], machine_avail_times[machine])
        end_time = start_time + duration
        job_start_times[job] = end_time
        machine_avail_times[machine] = end_time
        makespan = max(makespan, end_time)

    return makespan

def solve_with_heuristic(dataset, heuristic='SPT'):
    solutions = []

    job_completion = [0] * dataset.n_job
    machine_available_time = [0] * dataset.n_machine
    job_start_times = [0] * dataset.n_job
    machine_end_times = [0] * dataset.n_machine

    while any(job_completion[job] < len(dataset.op_data[job]) for job in range(dataset.n_job)):
        job_op_times = []
        for job in range(dataset.n_job):
            if job_completion[job] < len(dataset.op_data[job]):
                op_idx = job_completion[job]
                for machine, duration in dataset.op_data[job][op_idx]:
                    start_time = max(job_start_times[job], machine_end_times[machine])
                    job_op_times.append((duration, start_time, job, op_idx, machine))
        
        if heuristic == 'SPT':
            job_op_times.sort()
        elif heuristic == 'LPT':
            job_op_times.sort(reverse=True)
        elif heuristic == 'MINPT':
            job_op_times.sort(key=lambda x: x[0])
        elif heuristic == 'MAXPT':
            job_op_times.sort(key=lambda x: x[0], reverse=True)
        elif heuristic == 'MWKR':
            job_op_times.sort(key=lambda x: sum(t for _, t in dataset.op_data[x[2]][x[3]:]), reverse=True)
        
        next_op = job_op_times[0]
        duration, start_time, job, op_idx, machine = next_op
        
        end_time = start_time + duration
        job_start_times[job] = end_time
        machine_end_times[machine] = end_time
        job_completion[job] += 1

        solutions.append((job, op_idx, machine, duration))

    makespan = calculate_makespan(solutions, dataset)
    return solutions, makespan

def generate_colors(n):
    colors = plt.colormaps['tab20'](range(n))
    return [mcolors.rgb2hex(c) for c in colors]

def color(row, color_map):
    return color_map[row['Job']]

def draw_gantt_chart(predictions, dataset, title="Gantt Chart"):
    job_start_times = {job: 0 for job in range(dataset.n_job)}
    machine_avail_times = {machine: 0 for machine in range(dataset.n_machine)}
    makespan = 0

    gantt_chart = []
    job_info = []

    for job, op, machine, duration in predictions:
        start_time = max(job_start_times[job], machine_avail_times[machine])
        end_time = start_time + duration

        gantt_chart.append((machine, start_time, end_time, job))
        job_info.append((job, op, machine, start_time, end_time, duration))
        
        job_start_times[job] = end_time
        machine_avail_times[machine] = end_time
        makespan = max(makespan, end_time)

    gantt_df = pd.DataFrame(gantt_chart, columns=['Machine', 'Start', 'End', 'Job'])
    job_info_df = pd.DataFrame(job_info, columns=['Job', 'Operation', 'Machine', 'Start', 'End', 'Duration'])

    unique_jobs = gantt_df['Job'].unique()
    colors = plt.cm.get_cmap('tab20', len(unique_jobs)).colors
    color_map = {job: mcolors.rgb2hex(colors[i]) for i, job in enumerate(unique_jobs)}

    gantt_df['Color'] = gantt_df.apply(lambda row: color_map[row['Job']], axis=1)
    gantt_df['Delta'] = gantt_df['End'] - gantt_df['Start']

    fig, ax = plt.subplots(1, figsize=(16*0.8, 9*0.8))
    ax.barh(gantt_df['Machine'], gantt_df['Delta'], left=gantt_df['Start'], color=gantt_df['Color'], edgecolor='black')

    # 수정된 부분: 세로축을 1 간격으로 설정
    ax.set_yticks(np.arange(0, dataset.n_machine + 1, 1))
    ax.set_yticklabels(np.arange(0, dataset.n_machine + 1, 1))

    legend_elements = [Patch(facecolor=color_map[job], label=f'Job {job}') for job in unique_jobs]
    plt.legend(handles=legend_elements)
    plt.title(title, size=24)
    ax.set_xlim(0, makespan + 10)

    plt.text(makespan, -1, f'{makespan}', color='black', ha='center', va='center')
    plt.text(makespan, ax.get_ylim()[1], f'Max Makespan: {makespan}', color='red', ha='right', va='top')

    plt.xlabel('Time')
    plt.ylabel('Machine')

    plt.subplots_adjust(left=0.2, top=0.7)

    plt.show()

def main():
    datasets = ['fjsspdataset/BrandimarteMk6.fjs']

    heuristics = ['SPT', 'LPT', 'MINPT', 'MAXPT', 'MWKR']

    for dataset_path in datasets:
        dataset = RLDataset_FJSSP(dataset_path)
        for heuristic in heuristics:
            print(f"Solving {dataset_path} with {heuristic} heuristic:")
            solution, makespan = solve_with_heuristic(dataset, heuristic=heuristic)
            print(f"{heuristic} Makespan: {makespan}\n")
            draw_gantt_chart(solution, dataset, title=f"{heuristic} Heuristic Gantt Chart for {os.path.basename(dataset_path)}")

if __name__ == "__main__":
    main()
