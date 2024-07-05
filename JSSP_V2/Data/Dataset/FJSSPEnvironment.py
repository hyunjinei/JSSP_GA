import numpy as np

class FJSSPEnvironment:
    def __init__(self, process_times, machine_sequence):
        self.process_times = process_times
        self.machine_sequence = machine_sequence
        self.n_jobs = len(machine_sequence)
        self.n_machines = max(max(m for m, _ in op) for job in machine_sequence for op in job) + 1
        self.reset()

    def reset(self):
        self.current_time = 0
        self.job_completion = [0] * self.n_jobs
        self.machine_available_time = [0] * self.n_machines
        self.mask = self.create_initial_mask()
        return self.get_state(), self.flatten_mask(self.mask)

    def create_initial_mask(self):
        mask = [[True if i == 0 else False for i in range(len(self.machine_sequence[job]))] for job in range(self.n_jobs)]
        return mask

    def flatten_mask(self, mask):
        flattened_mask = []
        for job_mask in mask:
            flattened_mask.extend(job_mask)
        return flattened_mask

    def get_state(self):
        state = []
        for job in range(self.n_jobs):
            for op in range(len(self.machine_sequence[job])):
                for machine, time in self.machine_sequence[job][op]:
                    state.append([job, op, machine, time, 1 if self.mask[job][op] else 0])
        return np.array(state)

    def step(self, job, op, machine):
        processing_time = next(time for m, time in self.job_data[job][op] if m == machine)
        start_time = max(self.current_time, self.machine_available_time[machine])
        end_time = start_time + processing_time

        self.current_time = max(self.current_time, end_time)
        self.job_completion[job] = op + 1
        self.machine_available_time[machine] = end_time

        if self.job_completion[job] < len(self.job_data[job]):
            self.mask[job][self.job_completion[job]] = True

        done = all(completion == len(job) for completion, job in zip(self.job_completion, self.job_data))
        reward = -processing_time  # Negative processing time as reward

        return self.get_state(), self.mask, reward, done
