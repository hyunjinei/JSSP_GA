import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from collections import deque
import random
from JobShopEnv_FJSSP_clude import JobShopEnv_FJSSP
from RLDataset_FJSSP import RLDataset_FJSSP

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.conv3(x, edge_index)

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

class DeepMAG:
    def __init__(self, state_size, action_size, n_jobs, n_machines):
        self.state_size = state_size
        self.action_size = action_size
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.update_target_frequency = 1000
        self.batch_size = 64

        self.task_model = DQN(state_size, action_size)
        self.machine_model = DQN(state_size, action_size)
        self.target_task_model = DQN(state_size, action_size)
        self.target_machine_model = DQN(state_size, action_size)
        
        self.task_optimizer = optim.Adam(self.task_model.parameters(), lr=self.learning_rate)
        self.machine_optimizer = optim.Adam(self.machine_model.parameters(), lr=self.learning_rate)

        self.gnn = GNN(state_size, 64, 32)
        self.gnn_optimizer = optim.Adam(self.gnn.parameters(), lr=self.learning_rate)

        self.update_target_models()

    def update_target_models(self):
        self.target_task_model.load_state_dict(self.task_model.state_dict())
        self.target_machine_model.load_state_dict(self.machine_model.state_dict())

    def remember(self, state, task_action, machine_action, reward, next_state, done):
        self.memory.append((state, task_action, machine_action, reward, next_state, done))

    def act(self, state, agent_type, edge_index, valid_actions):
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)
        
        state = torch.FloatTensor(state).unsqueeze(0)
        gnn_output = self.gnn(state, edge_index)
        state_with_gnn = torch.cat((state, gnn_output), dim=-1)
        
        if agent_type == 'task':
            act_values = self.task_model(state_with_gnn).squeeze()
        else:
            act_values = self.machine_model(state_with_gnn).squeeze()
        
        # 유효한 액션 중에서 최대값을 가진 액션 선택
        valid_act_values = act_values[valid_actions]
        return valid_actions[torch.argmax(valid_act_values).item()]

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, task_action, machine_action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            
            # GNN processing
            edge_index = torch.tensor(state['graph'], dtype=torch.long)
            gnn_output = self.gnn(state, edge_index)
            state_with_gnn = torch.cat((state, gnn_output), dim=-1)
            
            next_edge_index = torch.tensor(next_state['graph'], dtype=torch.long)
            next_gnn_output = self.gnn(next_state, next_edge_index)
            next_state_with_gnn = torch.cat((next_state, next_gnn_output), dim=-1)

            # Task agent update
            task_target = reward
            if not done:
                task_target = reward + self.gamma * torch.max(self.target_task_model(next_state_with_gnn))
            task_target_f = self.task_model(state_with_gnn)
            task_target_f[0][task_action] = task_target
            task_loss = F.mse_loss(self.task_model(state_with_gnn), task_target_f)
            self.task_optimizer.zero_grad()
            task_loss.backward()
            self.task_optimizer.step()

            # Machine agent update
            machine_target = reward
            if not done:
                machine_target = reward + self.gamma * torch.max(self.target_machine_model(next_state_with_gnn))
            machine_target_f = self.machine_model(state_with_gnn)
            machine_target_f[0][machine_action] = machine_target
            machine_loss = F.mse_loss(self.machine_model(state_with_gnn), machine_target_f)
            self.machine_optimizer.zero_grad()
            machine_loss.backward()
            self.machine_optimizer.step()

            # GNN update
            gnn_loss = task_loss + machine_loss
            self.gnn_optimizer.zero_grad()
            gnn_loss.backward()
            self.gnn_optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def create_graph(self, state):
        # This is a placeholder. You need to implement the actual graph creation logic
        # based on the operation relationships among machines and jobs
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        return edge_index

    def train(self, env, episodes):
        for e in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                edge_index = torch.tensor(state['graph'], dtype=torch.long)
                valid_actions = env.get_valid_actions()
                task_action = self.act(state, 'task', edge_index, valid_actions)
                machine_action = self.act(state, 'machine', edge_index, valid_actions)
                
                job, op = divmod(task_action, self.n_machines)
                machine = machine_action % self.n_machines
                
                next_state, reward, done = env.step(job, op, machine)
                
                self.remember(state, task_action, machine_action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1
                
                self.replay()
                
                if steps % self.update_target_frequency == 0:
                    self.update_target_models()
            
            print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.2f}")


    def load(self, name):
        self.task_model.load_state_dict(torch.load(f"{name}_task.pth"))
        self.machine_model.load_state_dict(torch.load(f"{name}_machine.pth"))
        self.gnn.load_state_dict(torch.load(f"{name}_gnn.pth"))

    def save(self, name):
        torch.save(self.task_model.state_dict(), f"{name}_task.pth")
        torch.save(self.machine_model.state_dict(), f"{name}_machine.pth")
        torch.save(self.gnn.state_dict(), f"{name}_gnn.pth")

def main():
    # Load dataset
    dataset = RLDataset_FJSSP('fjsspdataset/HurinkRdata7.fjs')

    # Create environment
    env = JobShopEnv_FJSSP(dataset.op_data)

    # Initialize DeepMAG
    state = env.reset()
    state_size = (1 +  # current_time
                  len(state['job_completion']) +
                  len(state['machine_available_time']) +
                  len(state['machine_utilization']) +
                  len(state['job_queue_length']) +
                  len(state['job_progress']) +
                  len(state['remaining_job_time']) +
                  sum(len(status) for status in state['job_op_status']) +
                  len(state['machine_status']))
    action_size = env.n_jobs * env.n_machines
    agent = DeepMAG(state_size, action_size, env.n_jobs, env.n_machines)

    # Train the agent
    agent.train(env, episodes=1000)

    # Save the trained model
    agent.save("deepmag_model")

if __name__ == "__main__":
    main()