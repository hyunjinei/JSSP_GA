# Ptr_Net_test.py
# Ptr_Net_test.py
import os
import random
import torch
import logging
import torch.nn as nn
import torch.optim as optim
from collections import deque
from RLDataset import RLDataset
from JobShopEnv import JobShopEnv
from SharedPtrNet import SharedPtrNet
from critic_copy import PtrNet2
import torch.nn.functional as F

# 로그 파일 설정
logging.basicConfig(filename='training_progress.log', level=logging.INFO, format='%(asctime:s - %(message)s')

class PtrNetAgent:
    def __init__(self, shared_params, dynamic_params, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = SharedPtrNet(shared_params, dynamic_params)
        combined_params = {**shared_params, **dynamic_params}
        self.critic = PtrNet2(combined_params) if shared_params["use_critic"] else None
        self.optimizer = optim.Adam(self.model.parameters(), lr=shared_params["learning_rate"])
        if self.critic:
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=shared_params["learning_rate"])
            self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=shared_params["memory_size"])
        self.shared_params = shared_params

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, device):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.model.eval()
        with torch.no_grad():
            action, _, _ = self.model(state, device)
        return action[0].cpu().numpy()

    def learn(self, device):
        if len(self.memory) < self.shared_params["batch_size"]:
            return

        batch = random.sample(self.memory, self.shared_params["batch_size"])
        states, actions, rewards, next_states, dones = zip(*batch)

        states = [torch.FloatTensor([state[0]] + list(state[1]) + list(state[2])).to(device) for state in states]
        next_states = [torch.FloatTensor([next_state[0]] + list(next_state[1]) + list(next_state[2])).to(device) for next_state in next_states]
        rewards = torch.FloatTensor(rewards).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)

        self.model.train()
        log_probs = []
        for state, action in zip(states, actions):
            logging.info(f"State shape before model: {state.size()}")
            _, log_prob, _ = self.model(state.unsqueeze(0), device, action)
            log_probs.append(log_prob)

        log_probs = torch.cat(log_probs).squeeze()
        loss = -(log_probs * rewards).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        logging.info(f"Actor loss: {loss.item()}")

        if self.critic:
            self.critic.train()
            pred_rewards = []
            for state in states:
                pred_reward = self.critic(state.unsqueeze(0), device).squeeze()
                pred_rewards.append(pred_reward.unsqueeze(0))  # 0차원 텐서를 1차원으로 변환

            pred_rewards = torch.cat(pred_rewards)
            critic_loss = self.criterion(pred_rewards, rewards)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            logging.info(f"Critic loss: {critic_loss.item()}")

    # pretrain 함수 수정
    def pretrain(self, solutions, env, device, repeat_count=100):
        total_solutions = len(solutions)
        for i in range(repeat_count):
            solution = solutions[random.randint(0, total_solutions - 1)]
            state = env.reset()
            total_reward = 0
            for step in solution:
                job, op = divmod(step - 1, env.n_machines)
                next_state, reward, done = env.step(job, op)
                reward += self.shared_params["solution_match_reward"] if (job, op) in env.solution_actions else 0
                self.remember(state, (job, op), reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break
            logging.info("Total reward for pretraining solution {}/{}: {}".format(i + 1, repeat_count, total_reward))
        self.learn(device)


def load_solutions(filename, max_solutions=100000):
    with open(filename, 'r') as file:
        lines = file.readlines()

    if len(lines) <= max_solutions:
        solutions = [list(map(int, line.strip().split())) for line in lines]
    else:
        lines = random.sample(lines, max_solutions)
        solutions = [list(map(int, line.strip().split())) for line in lines]

    return solutions

def load_datasets(file_list):
    datasets = {}
    for file in file_list:
        dataset = RLDataset(file)
        solution_file = file.replace('.txt', '_solutions.txt')
        solutions = load_solutions(solution_file, max_solutions=100000)
        datasets[file] = (dataset, solutions)
    return datasets

# pretrain_with_solutions 함수의 일부
def pretrain_with_solutions(agent, solutions, env, device, repeat_count=100):
    total_solutions = len(solutions)
    for i in range(repeat_count):
        solution = solutions[random.randint(0, total_solutions - 1)]
        state = env.reset()
        total_reward = 0
        for step in solution:
            job, op = divmod(step - 1, env.n_machines)
            next_state, reward, done = env.step(job, op)
            reward += agent.shared_params["solution_match_reward"] if (job, op) in env.solution_actions else 0

            # 디버깅 출력 추가
            logging.info("State: {}, Next State: {}".format(state, next_state))

            state_vector = [state[0]] + list(state[1]) + list(state[2])
            next_state_vector = [next_state[0]] + list(next_state[1]) + list(next_state[2])

            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0).to(device)
            next_state_tensor = torch.FloatTensor(next_state_vector).unsqueeze(0).to(device)

            agent.remember(state_tensor, (job, op), reward, next_state_tensor, done)
            state = next_state
            total_reward += reward
            if done:
                break
        logging.info("Total reward for pretraining solution {}/{}: {}".format(i + 1, repeat_count, total_reward))
        agent.learn(device)


def train_individual_models(datasets, shared_params, device):
    models = []
    for filename, (dataset, solutions) in datasets.items():
        print(f"Pretraining with dataset {filename}")
        env = JobShopEnv(dataset.op_data, [[m for m, _ in job] for job in dataset.op_data], solutions)

        # 각 데이터셋의 입력 크기를 기반으로 input_dim 설정
        example_state = env.reset()
        print(f"Example state: {example_state}")
        input_dim = 1 + len(example_state[1]) + len(example_state[2])
        print(f"Calculated input_dim: {input_dim}")
        dynamic_params = {
            "embedding_input_dim": input_dim,
            "n_embedding": 128,
            "n_hidden": 128
        }

        state_size = env.n_jobs * env.n_machines * 3
        action_size = env.n_jobs * env.n_machines
        shared_params["num_of_process"] = env.n_machines  # Ensure the number of processes is set correctly
        shared_params["state_size"] = state_size  # Ensure the state size is set correctly
        shared_params["action_size"] = action_size  # Ensure the action size is set correctly
        shared_params["n_process"] = env.n_machines  # Ensure n_process is set correctly

        agent = PtrNetAgent(shared_params, dynamic_params, state_size, action_size)
        agent.model.to(device)
        if agent.critic:
            agent.critic.to(device)
        pretrain_with_solutions(agent, solutions, env, device, repeat_count=shared_params["repeat_count"])
        models.append(agent)
        torch.save(agent.model.state_dict(), f"ptrnet_model_{filename}.pth")
        if agent.critic:
            torch.save(agent.critic.state_dict(), f"ptrnet_critic_{filename}.pth")
    return models

def ensemble_models(models):
    def ensemble_prediction(state, device):
        predictions = []
        for model in models:
            model.model.eval()
            with torch.no_grad():
                action, _, _ = model.model(state, device)
                predictions.append(action)

        # 각 텐서를 동일한 크기로 패딩합니다.
        max_len = max(pred.shape[0] for pred in predictions)
        padded_predictions = []
        for pred in predictions:
            if pred.shape[0] < max_len:
                padding = (0, max_len - pred.shape[0])
                pred = F.pad(pred, padding, "constant", -1)
            padded_predictions.append(pred)

        # 다수결 앙상블 예측을 수행합니다.
        predictions = torch.stack(padded_predictions)
        ensemble_action = torch.mode(predictions, dim=0).values
        return ensemble_action
    return ensemble_prediction


def evaluate_model(ensemble_predict, benchmark_dataset, env, device, num_predictions=100):
    env.reset()
    predictions = []
    for _ in range(num_predictions):
        state = env.reset()
        done = False
        prediction = []
        while not done:
            state_tensor = torch.FloatTensor([state[0]] + list(state[1]) + list(state[2])).unsqueeze(0).to(device)
            action = ensemble_predict(state_tensor, device).cpu().numpy()
            next_state, reward, done = env.step(*action)
            prediction.append(action)
            state = next_state
        predictions.append(prediction)
    return predictions

def main():
    train_file_list = ['abz5.txt', 'abz6.txt', 'dmu32.txt']
    benchmark_file = 'ta21.txt'

    shared_params = {
        "learning_rate": 1e-4,
        "num_episodes": 10000,
        "log_interval": 10,
        "save_interval": 100,
        "use_critic": True,
        "n_embedding": 128,
        "n_hidden": 128,
        "init_min": -0.08,
        "init_max": 0.08,
        "decode_type": "sampling",
        "n_glimpse": 1,
        "T": 1.0,
        "C": 10,
        "batch_size": 64,
        "memory_size": 200000,
        "solution_match_reward": 1000, # 솔루션과 일치하는 경우 보상
        "repeat_count": 1, # 사전 훈련 반복 횟수
        "use_logit_clipping": True,  # 추가된 키
        "n_process": None, # 초기화
        "embedding_input_dim": 128
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = []
    
    for file in train_file_list:
        dataset = RLDataset(file)
        solution_file = file.replace('.txt', '_solutions.txt')
        solutions = load_solutions(solution_file, max_solutions=100000)

        env = JobShopEnv(dataset.op_data, [[m for m, _ in job] for job in dataset.op_data], solutions)
        example_state = env.reset()
        input_dim = 1 + len(example_state[1]) + len(example_state[2])
        dynamic_params = {
            "embedding_input_dim": input_dim,
            "n_embedding": 128,
            "n_hidden": 128
        }

        state_size = env.n_jobs * env.n_machines * 3
        action_size = env.n_jobs * env.n_machines
        shared_params["num_of_process"] = env.n_machines
        shared_params["state_size"] = state_size
        shared_params["action_size"] = action_size
        shared_params["n_process"] = env.n_machines

        agent = PtrNetAgent(shared_params, dynamic_params, state_size, action_size)
        agent.model.to(device)
        if agent.critic:
            agent.critic.to(device)
        agent.pretrain(solutions, env, device, repeat_count=shared_params["repeat_count"])
        models.append(agent)
        torch.save(agent.model.state_dict(), f"ptrnet_model_{file}.pth")
        if agent.critic:
            torch.save(agent.critic.state_dict(), f"ptrnet_critic_{file}.pth")

    ensemble_predict = ensemble_models(models)
    torch.save(models, "ptrnet_models.pth")

    benchmark_dataset = RLDataset(benchmark_file)
    env = JobShopEnv(benchmark_dataset.op_data, [[m for m, _ in job] for job in benchmark_dataset.op_data])
    state_size = env.n_jobs * env.n_machines * 3
    action_size = env.n_jobs * env.n_machines
    predictions = evaluate_model(ensemble_predict, benchmark_dataset, env, device)

    print(f"Predictions for benchmark dataset {benchmark_file}: {predictions}")

if __name__ == "__main__":
    main()
