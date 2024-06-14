import numpy as np

class JobShopEnv:
    def __init__(self, process_times, machine_sequence):
        self.n_jobs = len(process_times)
        self.n_machines = len(process_times[0])
        self.process_times = process_times
        self.machine_sequence = machine_sequence
        self.reset()
        
    def reset(self):
        self.current_time = 0
        self.job_completion = [0] * self.n_jobs
        self.machine_available_time = [0] * self.n_machines
        self.state = (self.current_time, self.job_completion, self.machine_available_time)
        return self.state
    
    def step(self, job, machine):
        processing_time = self.process_times[job][self.job_completion[job]]
        self.machine_available_time[machine] += processing_time
        self.job_completion[job] += 1
        self.current_time = max(self.machine_available_time)
        reward = -self.current_time
        done = all(c == self.n_machines for c in self.job_completion)
        self.state = (self.current_time, self.job_completion, self.machine_available_time)
        return self.state, reward, done
    
    def get_valid_actions(self):
        valid_actions = []
        for job in range(self.n_jobs):
            if self.job_completion[job] < self.n_machines:
                machine = self.machine_sequence[job][self.job_completion[job]]
                valid_actions.append((job, machine))
        return valid_actions
