import random

class QLearningAgent:
    def __init__(self, n_jobs, n_machines, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def get_q_value(self, state, action):
        state_tuple = (state[0], tuple(state[1]), tuple(state[2]))
        action_tuple = tuple(action)
        return self.q_table.get((state_tuple, action_tuple), 0.0)
    
    def update_q_value(self, state, action, reward, next_state, valid_actions):
        state_tuple = (state[0], tuple(state[1]), tuple(state[2]))
        action_tuple = tuple(action)
        next_state_tuple = (next_state[0], tuple(next_state[1]), tuple(next_state[2]))
        best_next_action = max(self.get_q_value(next_state_tuple, tuple(next_action)) for next_action in valid_actions)
        self.q_table[(state_tuple, action_tuple)] = self.get_q_value(state_tuple, action_tuple) + self.alpha * (reward + self.gamma * best_next_action - self.get_q_value(state_tuple, action_tuple))
    
    def get_action(self, state, valid_actions):
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            state_tuple = (state[0], tuple(state[1]), tuple(state[2]))
            return max(valid_actions, key=lambda action: self.get_q_value(state_tuple, tuple(action)))
