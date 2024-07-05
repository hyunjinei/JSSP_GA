import os
import pandas as pd
from RLDataset import RLDataset
from JobShopEnv_multi import JobShopEnv
import matplotlib.pyplot as plt
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import gc
import time
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

# 로그 파일 설정
logging.basicConfig(filename='training_progress.log', level=logging.INFO, format='%(asctime)s - %(message)s')

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

def create_graph(env):
    nodes = []
    for i in range(env.n_jobs):
        nodes.append([1, 0])
    for j in range(env.n_machines):
        nodes.append([0, 1])
    
    edges = []
    edge_features = []
    for job in range(env.n_jobs):
        for op in range(env.n_machines):
            machine = env.machine_sequence[job][op]
            edges.append([job, env.n_jobs + machine])
            edge_features.append([env.process_times[job][op][1]])
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    x = torch.tensor(nodes, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data

class DQN(nn.Module):
    def __init__(self, max_state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(max_state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, max_state_size, action_size, update_target_frequency=32, replay_start_size=1000):
        self.state_size = max_state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = DQN(max_state_size, action_size)
        self.target_model = DQN(max_state_size, action_size)
        self.target_model2 = DQN(max_state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss() # self.criterion = F.smooth_l1_loss  # Huber Loss 사용
        self.update_target_frequency = update_target_frequency
        self.replay_start_size = replay_start_size
        self.update_counter = 0

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model2.load_state_dict(self.model.state_dict())
        print("Updated target network.")  # 타겟 네트워크 업데이트 메시지 추가

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, mask=None):
        state = torch.FloatTensor(state).unsqueeze(0)
        if mask is not None:
            mask = torch.FloatTensor(mask).unsqueeze(0)
        act_values = self.model(state, mask).detach().numpy()[0]
        
        max_action = np.argmax(act_values)
        probabilities = np.ones(self.action_size) * (self.epsilon / self.action_size)
        probabilities[max_action] += (1.0 - self.epsilon)
        
        return np.random.choice(self.action_size, p=probabilities)

    def replay(self, batch_size):
        if len(self.memory) < self.replay_start_size:
            return  # 메모리에 충분한 경험이 쌓일 때까지 리플레이를 시작하지 않음

        minibatch = random.sample(self.memory, batch_size)
        for idx, (state, action, reward, next_state, done) in enumerate(minibatch):
            state, mask = pad_state_action(state, [])
            next_state, next_mask = pad_state_action(next_state, [])
            state = torch.FloatTensor(state).unsqueeze(0)
            mask = torch.FloatTensor(mask).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).unsqueeze(0)
            next_mask = torch.FloatTensor(next_mask).unsqueeze(0)
            
            target = reward
            if not done:
                best_action = torch.argmax(self.model(next_state, next_mask)[0]).item()
                target += self.gamma * self.target_model2(next_state, next_mask)[0][best_action].item()
            
            target_f = self.model(state, mask)
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state, mask))
            loss.backward()
            self.optimizer.step()
            
            if idx % 1 == 0:  # 로그 빈도를 줄임
                logging.info(f"Replay: Step {idx+1}/{batch_size} processed. Loss: {loss.item()}")
                print(f"Replay: Step {idx+1}/{batch_size} processed. Loss: {loss.item()}")
                gc.collect()

            self.update_counter += 1  # 이 부분을 반복 내에 추가하여 매 스텝마다 업데이트 카운터를 증가시킴

        if self.update_counter % self.update_target_frequency == 0:
            self.update_target_model()
            print("Updated target network after replay.")  # 타겟 네트워크 업데이트 메시지 추가

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay




class TaskAgent(DQNAgent):
    def __init__(self, max_state_size, action_size, gnn, update_target_frequency=32, replay_start_size=1000):
        super(TaskAgent, self).__init__(max_state_size, action_size, update_target_frequency, replay_start_size)
        self.gnn = gnn

    def act(self, state, mask=None, env=None):
        graph_data = create_graph(env)
        gnn_output = self.gnn(graph_data)
        gnn_output_flat = gnn_output.flatten().detach().numpy()
        
        extended_state = np.concatenate((state, gnn_output_flat))
        padded_state, mask = pad_state_action(extended_state, [])
        
        return super(TaskAgent, self).act(padded_state, mask)

class MachineAgent(DQNAgent):
    def __init__(self, max_state_size, action_size, gnn, update_target_frequency=32, replay_start_size=1000):
        super(MachineAgent, self).__init__(max_state_size, action_size, update_target_frequency, replay_start_size)
        self.gnn = gnn

    def act(self, state, mask=None, env=None):
        graph_data = create_graph(env)
        gnn_output = self.gnn(graph_data)
        gnn_output_flat = gnn_output.flatten().detach().numpy()
        
        extended_state = np.concatenate((state, gnn_output_flat))
        padded_state, mask = pad_state_action(extended_state, [])
        
        return super(MachineAgent, self).act(padded_state, mask)

def load_solutions(filename, max_solutions=100000):
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    if len(lines) <= max_solutions:
        solutions = [list(map(int, line.strip().split())) for line in lines]
    else:
        lines = random.sample(lines, max_solutions)
        solutions = [list(map(int, line.strip().split())) for line in lines]
    
    return solutions

def pad_state_action(state, action, max_state_size=6000, pad_value=-2):
    if len(state) > max_state_size:
        state = state[:max_state_size]
    
    max_action_size = 100 * 20
    padded_state = np.full(max_state_size, pad_value)
    padded_state[:len(state)] = state
    mask = np.zeros(max_state_size)
    mask[:len(state)] = 1
    padded_action = np.full(max_action_size, pad_value)
    padded_action[:len(action)] = action
    return padded_state, mask

def pretrain_with_solutions(task_agent, machine_agent, solutions, env, repeat_count=100):
    total_solutions = len(solutions)
    for i in range(repeat_count):
        solution = solutions[random.randint(0, total_solutions - 1)]
        state = env.reset()
        state, mask = pad_state_action(np.concatenate((np.array([state[0]]), state[1], state[2])), [])
        done = False
        while not done:
            for j, action in enumerate(solution):
                job, op = divmod(action - 1, env.n_machines)
                if job >= env.n_jobs or op >= len(env.process_times[job]):
                    continue
                next_state, done = env.step(job, op, use_solution_actions=True)
                next_state, next_mask = pad_state_action(np.concatenate((np.array([next_state[0]]), next_state[1], next_state[2])), [])
                task_agent.remember(state, job * env.n_machines + op, 0, next_state, done)
                machine_agent.remember(state, op * env.n_jobs + job, 0, next_state, done)
                state, mask = next_state, next_mask

                if done:
                    break

            if done:
                reward_task, reward_machine = env.calculate_episode_rewards(is_pretrain=True)
                # print(f"Pretraining: Episode {i+1}/{repeat_count} processed. Reward Task: {reward_task}, Reward Machine: {reward_machine}")
                break

        if len(task_agent.memory) >= task_agent.replay_start_size:
            for _ in range(4):  # 각 에피소드에서 replay가 충분히 호출되도록 설정
                task_agent.replay(32)
                machine_agent.replay(32)

        if i % 10 == 0:
            logging.info(f"Pretraining: Solution {i+1}/{repeat_count} processed.")
            print(f"Pretraining: Solution {i+1}/{repeat_count} processed.")
            gc.collect()

    task_agent.update_target_model()
    machine_agent.update_target_model()




def combine_models(models, max_state_size, max_action_size):
    combined_model = models[0]
    combined_state_dict = combined_model.model.state_dict()

    for key in combined_state_dict.keys():
        combined_weight = combined_state_dict[key].clone()
        for model in models[1:]:
            model_weight = model.model.state_dict()[key]
            if combined_weight.size() == model_weight.size():
                combined_weight += model_weight
            else:
                logging.warning(f"Skipping combining weights for layer {key} due to size mismatch: {combined_weight.size()} vs {model_weight.size()}")
        combined_state_dict[key] = combined_weight / len(models)

    combined_model.model.load_state_dict(combined_state_dict)
    combined_model.state_size = max_state_size
    combined_model.action_size = max_action_size
    logging.info("Models combined into a single model.")
    print("Models combined into a single model.")
    return combined_model

def analyze_data(op_data):
    first_op_machines = [job[0][0] for job in op_data]
    last_op_machines = [job[-1][0] for job in op_data]
    
    machine_op_distribution = {m: [0] * len(op_data[0]) for m in range(len(op_data[0]))}
    op_durations = {op: 0 for op in range(len(op_data[0]))}
    max_duration = 0

    for job in op_data:
        for op_index, (machine, duration) in enumerate(job):
            machine_op_distribution[machine][op_index] += 1
            op_durations[op_index] += duration
            if duration > max_duration:
                max_duration = duration

    op_durations_total = sum(op_durations.values())
    
    machine_op_ratios = {m: [op / len(op_data) for op in ops] for m, ops in machine_op_distribution.items()}
    op_time_ratios_total = {op: duration / op_durations_total for op, duration in op_durations.items()}
    op_time_ratios_max = {op: duration / max_duration for op, duration in op_durations.items()}

    cumulative_time = 0
    
    for op in range(len(op_data[0])):
        cumulative_time += op_durations[op]

    first_op_machine_ratio = len(set(first_op_machines)) / len(op_data[0])
    last_op_machine_ratio = len(set(last_op_machines)) / len(op_data[0])

    return {
        "first_op_machine_ratio": first_op_machine_ratio,
        "last_op_machine_ratio": last_op_machine_ratio,
        "machine_op_ratios": machine_op_ratios,
        "op_time_ratios_total": op_time_ratios_total,
        "op_time_ratios_max": op_time_ratios_max,
    }

def train_individual_models(datasets):
    task_agents = []
    machine_agents = []
    max_state_size = 100 * 20 * 3
    max_action_size = 100 * 20

    gnn_input_dim = 2
    gnn_hidden_dim = 128
    gnn_output_dim = 64

    gnn = GNN(gnn_input_dim, gnn_hidden_dim, gnn_output_dim)

    for i, dataset in enumerate(datasets):
        logging.info(f"Starting training for dataset {dataset} ({i+1}/{len(datasets)})")
        print(f"Starting training for dataset {dataset} ({i+1}/{len(datasets)})")

        data = RLDataset(dataset)
        solutions = load_solutions(f"{data.name}_solutions.txt", max_solutions=100000)

        op_data_analysis = analyze_data(data.op_data)
        print(f"Data Analysis Results: {op_data_analysis}")

        analysis_values = []
        for key, value in op_data_analysis.items():
            if isinstance(value, dict):
                analysis_values.extend(value.values())
            else:
                analysis_values.append(value)

        env = JobShopEnv(data.op_data, [[m for m, _ in job] for job in data.op_data], solutions, analysis_values)

        state_size = env.n_jobs * env.n_machines * 3
        action_size = env.n_jobs * env.n_machines

        task_agent = TaskAgent(max_state_size, action_size, gnn, update_target_frequency=64, replay_start_size=2000)
        machine_agent = MachineAgent(max_state_size, action_size, gnn, update_target_frequency=64, replay_start_size=2000)
        repeat_count = 100 if len(solutions) < 100000 else len(solutions)

        pretrain_with_solutions(task_agent, machine_agent, solutions, env, repeat_count=repeat_count)

        # replay 호출 사이에 충분한 업데이트와 학습을 수행하도록 추가 반복 설정
        for _ in range(100):
            task_agent.replay(32)
            machine_agent.replay(32)

            if task_agent.update_counter % task_agent.update_target_frequency == 0:
                task_agent.update_target_model()
                print("Updated target network for task agent during training.")
            
            if machine_agent.update_counter % machine_agent.update_target_frequency == 0:
                machine_agent.update_target_model()
                print("Updated target network for machine agent during training.")

        task_agents.append(task_agent)
        machine_agents.append(machine_agent)

        logging.info(f"Completed training for dataset {dataset} ({i+1}/{len(datasets)})")
        print(f"Completed training for dataset {dataset} ({i+1}/{len(datasets)})")
        gc.collect()

    return task_agents, machine_agents, max_state_size, max_action_size

def predict(task_agent, machine_agent, env, test_dataset, num_predictions=1, max_steps=100000000):
    valid_solution_count = 0
    all_predictions = []
    step_count = 0
    
    op_data_analysis = analyze_data(test_dataset.op_data)

    analysis_values = []
    for key, value in op_data_analysis.items():
        if isinstance(value, dict):
            for v in value.values():
                if isinstance(v, list):
                    analysis_values.extend(v)
                else:
                    analysis_values.append(v)
        else:
            analysis_values.append(value)

    analysis_values = np.array(analysis_values)

    while valid_solution_count < num_predictions and step_count < max_steps:
        state = env.reset()
        state = np.concatenate((np.array([state[0]]), state[1], state[2], analysis_values))
        state, mask = pad_state_action(state, [], max_state_size=6000)

        solution = []
        job_op_counter = {job: 0 for job in range(env.n_jobs)}
        total_operations = env.n_jobs * env.n_machines
        done = False

        while not done and step_count < max_steps:
            start_time = time.time()
            
            if np.random.rand() < 0.5:
                action = task_agent.act(state, mask, env)
                job, op = divmod(action, env.n_machines)
            else:
                action = machine_agent.act(state, mask, env)
                op, job = divmod(action, env.n_jobs)

            if job >= env.n_jobs or op >= env.n_machines or job_op_counter[job] != op:
                continue

            solution.append((job, op))
            job_op_counter[job] += 1
            step_count += 1

            if step_count % 100 == 0:
                print(f"Predicting: Step {step_count}, Current Solution: {solution[-5:]}")
                gc.collect()

            next_state, done = env.step(job, op, use_solution_actions=False)
            state = np.concatenate((np.array([next_state[0]]), next_state[1], next_state[2], analysis_values))
            state, mask = pad_state_action(state, [], max_state_size=6000)
            end_time = time.time()
            print(f"Step {step_count}")

        if done:
            reward_task, reward_machine = env.calculate_episode_rewards()
            print(f"Prediction: Episode {valid_solution_count+1}/{num_predictions} processed. Reward Task: {reward_task}, Reward Machine: {reward_machine}")
            valid_solution_count += 1
            all_predictions.append(solution)
            gc.collect()

    if valid_solution_count != num_predictions:
        print(f"Warning: Only {valid_solution_count} valid solutions were collected.")

    return all_predictions

def main():
    datasets = ['la03.txt']
    task_agents, machine_agents, max_state_size, max_action_size = train_individual_models(datasets)
    test_dataset = RLDataset('la03.txt')

    op_data_analysis = analyze_data(test_dataset.op_data)

    analysis_values = []
    for key, value in op_data_analysis.items():
        if isinstance(value, dict):
            analysis_values.extend(value.values())
        else:
            analysis_values.append(value)

    combined_task_agent = combine_models(task_agents, max_state_size, max_action_size)
    combined_machine_agent = combine_models(machine_agents, max_state_size, max_action_size)
    logging.info("Training completed for individual models and combined model created.")
    print("Training completed for individual models and combined model created.")

    logging.info("Creating environment object.")
    print("Creating environment object.")
    env = JobShopEnv(test_dataset.op_data, [[m for m, _ in job] for job in test_dataset.op_data], solutions=None, analysis_values=analysis_values)

    logging.info("Starting prediction process.")
    print("Starting prediction process.")
    all_predictions = predict(combined_task_agent, combined_machine_agent, env, test_dataset, num_predictions=2, max_steps=100000000)

    logging.info(f"Predicted Solutions: {all_predictions}")
    print("Predicted Solutions:", all_predictions)

    best_solution = None
    best_makespan = float('inf')
    for solution in all_predictions:
        makespan = calculate_makespan(solution, test_dataset)
        if makespan < best_makespan:
            best_makespan = makespan
            best_solution = solution

    logging.info(f"Best Solution: {best_solution}")
    print("Best Solution:", best_solution)

    ga_initial_population = [job * test_dataset.n_machine + op for job, op in best_solution]
    logging.info(f"GA Initial Population: {ga_initial_population}")
    print("GA Initial Population:", ga_initial_population)

    draw_gantt_chart(best_solution, test_dataset)

def calculate_makespan(solution, dataset):
    job_start_times = {job: 0 for job in range(dataset.n_job)}
    machine_avail_times = {machine: 0 for machine in range(dataset.n_machine)}
    makespan = 0

    for job, op in solution:
        machine, duration = dataset.op_data[job][op]
        start_time = max(job_start_times[job], machine_avail_times[machine])
        end_time = start_time + duration
        job_start_times[job] = end_time
        machine_avail_times[machine] = end_time
        makespan = max(makespan, end_time)

    return makespan

def generate_colors(n):
    colors = plt.cm.get_cmap('tab20', n).colors
    return [mcolors.rgb2hex(c) for c in colors]

def color(row, color_map):
    return color_map[row['Job']]

def draw_gantt_chart(predictions, dataset):
    job_start_times = {job: 0 for job in range(dataset.n_job)}
    machine_avail_times = {machine: 0 for machine in range(dataset.n_machine)}
    makespan = 0

    gantt_chart = []
    job_info = []

    for job, op in predictions:
        machine, duration = dataset.op_data[job][op]
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
    colors = generate_colors(len(unique_jobs))
    color_map = {job: colors[i] for i, job in enumerate(unique_jobs)}

    gantt_df['Color'] = gantt_df.apply(lambda row: color(row, color_map), axis=1)
    gantt_df['Delta'] = gantt_df['End'] - gantt_df['Start']

    fig, ax = plt.subplots(1, figsize=(16*0.8, 9*0.8))
    ax.barh(gantt_df.Machine, gantt_df.Delta, left=gantt_df.Start, color=gantt_df.Color, edgecolor='black')

    legend_elements = [Patch(facecolor=color_map[job], label=f'Job {job}') for job in unique_jobs]
    plt.legend(handles=legend_elements)
    plt.title('Gantt Chart', size=24)
    ax.set_xlim(0, makespan + 10)

    plt.text(makespan, -1, f'{makespan}', color='black', ha='center', va='center')
    plt.text(makespan, ax.get_ylim()[1], f'Max Makespan: {makespan}', color='red', ha='right', va='top')

    plt.xlabel('Time')
    plt.ylabel('Machine')

    table_data = job_info_df[['Job', 'Operation', 'Machine', 'Start', 'End', 'Duration']].values
    table = plt.table(cellText=table_data, colLabels=job_info_df.columns, cellLoc='center', loc='top', bbox=[0, -0.3, 1, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    plt.subplots_adjust(left=0.2, top=0.7)

    plt.show()

if __name__ == "__main__":
    main()
