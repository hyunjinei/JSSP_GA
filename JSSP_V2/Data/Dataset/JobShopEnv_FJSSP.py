import random
import numpy as np

class JobShopEnv_FJSSP:
    def __init__(self, process_times, machine_sequence, solutions=None, analysis_values=None):
        print("Initializing JobShopEnv_FJSSP")
        self.n_jobs = len(machine_sequence)
        self.n_machines = max(max(m for m, _ in job) for job_seq in machine_sequence for job in job_seq) + 1
        print(f"Number of jobs: {self.n_jobs}, Number of machines: {self.n_machines}")
        self.process_times = process_times
        self.machine_sequence = machine_sequence
        self.solutions = solutions
        self.analysis_values = analysis_values
        self.previous_makespan = None
        self.previous_idle_time = None
        self.previous_waiting_time = None
        self.reset()

    def reset(self):
        print("Resetting environment")
        self.current_time = 0
        self.job_completion = [0] * self.n_jobs
        self.machine_available_time = [0] * self.n_machines
        self.machine_task_completion = [0] * self.n_machines
        self.machine_start_times = [-1] * self.n_machines  # 초기화 시 각 머신의 시작 시간을 -1로 설정
        self.agent_actions = []
        self.rewards = []
        self.previous_idle_time = calculate_idle_time(self.agent_actions, self)
        self.state = self.get_state()
        print(f"Initial state: {self.state}")
        return self.state

    def get_state(self):
        um_t = self.calculate_machine_utilization()
        qj_t = self.calculate_job_queue_length()
        pr_t = self.calculate_job_progress()
        rj_t = self.calculate_remaining_job_time()
        
        # Lowerbound 계산
        lowerbound = self.calculate_lowerbound()

        machine_available_time_dict = {i: self.machine_available_time[i] for i in range(len(self.machine_available_time))}

        state = (self.current_time, 
                 tuple(self.job_completion), 
                 tuple(machine_available_time_dict.values()),
                 tuple(um_t),
                 tuple(qj_t),
                 tuple(pr_t),
                 )  # Lowerbound 추가
        return state

    def calculate_lowerbound(self):
        # 모든 남은 작업의 최소 처리 시간의 합을 계산
        lowerbound = 0
        for job in range(self.n_jobs):
            remaining_operations = self.machine_sequence[job][self.job_completion[job]:]
            for operations in remaining_operations:
                # 각 작업의 남은 각 작업에 대해 가능한 최소 처리 시간을 선택
                lowerbound += min(duration for _, duration in operations)
        
        # 현재 시간에 lowerbound를 더해 전체 lowerbound 계산
        lowerbound += self.current_time
        
        # 모든 기계의 현재 가용 시간 중 최대값을 고려
        max_machine_available_time = max(self.machine_available_time)
        lowerbound = max(lowerbound, max_machine_available_time)
        
        return lowerbound

    def calculate_machine_utilization(self):
        # 기계 평균 이용률 (Um(t)) 계산
        utilization = []
        for i in range(self.n_machines):
            usage_time = sum(
                [self.process_times[job][op][1] for job, op, machine in self.agent_actions if machine == i and job < len(self.process_times) and op < len(self.process_times[job])]
            )
            utilization.append(usage_time / self.current_time if self.current_time > 0 else 0)
        return utilization

    def calculate_job_queue_length(self):
        # 작업 큐 길이 (Qj(t)) 계산
        job_queue_length = [0] * self.n_machines
        for job, op, machine in self.agent_actions:
            job_queue_length[machine] += 1
        return job_queue_length

    def calculate_job_progress(self):
        # 작업 진행률 (Pr(t)) 계산
        job_progress = [completion / len(self.machine_sequence[job]) for job, completion in enumerate(self.job_completion)]
        return job_progress

    def calculate_remaining_job_time(self):
        # 작업 남은 시간 (Rj(t)) 계산
        remaining_time = []
        for job in range(self.n_jobs):
            remaining_operations = self.machine_sequence[job][self.job_completion[job]:]
            remaining_duration = 0
            for operations in remaining_operations:
                remaining_duration += sum([duration for machine, duration in operations])
            remaining_time.append(remaining_duration)
        return remaining_time

    def step(self, job, op, machine=None):
        # print(f"Executing step: job={job}, op={op}")

        if job >= len(self.machine_sequence) or op >= len(self.machine_sequence[job]):
            raise IndexError(f"Invalid operation index: job={job}, op={op}")

        machine_options = self.machine_sequence[job][op]
        print(f"Machine options: {machine_options}")

        if machine is None:
            machine, processing_time = random.choice(machine_options)
        else:
            processing_time = next(t for m, t in machine_options if m == machine)

        start_time = max(self.current_time, self.machine_available_time[machine])
        end_time = start_time + processing_time

        # Update machine_available_time for the specific machine
        self.machine_available_time[machine] = end_time
        # Calculate the current time using the custom function
        self.current_time = calculate_current_time(self.agent_actions, self)
        # Calculate machine available times using the custom function
        self.machine_available_time = calculate_machine_available_time(self.agent_actions, self)

        self.job_completion[job] += 1        

        # Print updated state
        # print(f"Updated state: current_time={self.current_time}, job_completion={self.job_completion}, machine_available_time={self.machine_available_time}")

        # 처음으로 작업 시작 시간 기록
        if self.machine_start_times[machine] == -1:
            self.machine_start_times[machine] = start_time

        self.agent_actions.append((job, op, machine))

        done = all(c == len(self.machine_sequence[job]) for c in self.job_completion)

        # Calculate step reward based on idle time
        current_idle_time = calculate_idle_time(self.agent_actions, self)
        if self.previous_idle_time is not None and current_idle_time == self.previous_idle_time:
            step_reward = 0.1  # 보상 값을 필요에 따라 조정
        else:
            step_reward = -0.5
        self.previous_idle_time = current_idle_time
        # print(f"Step idle time reward: {step_reward}")

        

        self.state = self.get_state()
        print(f"Updated state after get_state: {self.state}")  # 추가된 상태 출력
        return self.state, done, step_reward


    def calculate_episode_rewards(self, is_pretrain=False):
        print("Calculating episode rewards")
        total_reward_task = 0
        total_reward_machine = 0

        makespan = calculate_makespan_reward(self.agent_actions, self)
        # idle_time = calculate_idle_time(self.agent_actions, self)
        # utilization_reward = calculate_idle_machine_utilization(self.agent_actions, self)

        # print(f'Makespan: {makespan}, Idle Time: {idle_time}, Utilization Reward: {utilization_reward}')
        print(f'Makespan: {makespan}')

        reward_task = -makespan * 0  # waiting_time * 0.1
        reward_machine = -makespan * 0 #- idle_time * 0 + utilization_reward

        if self.previous_makespan is not None:
            makespan_diff = self.previous_makespan - makespan
            reward_task += makespan_diff * 2
            reward_machine += makespan_diff * 2
            # print(f'Makespan difference: {makespan_diff}, Adjusted Reward Task: {reward_task}')

        # if self.previous_idle_time is not None:
        #     idle_time_diff = self.previous_idle_time - idle_time
        #     reward_machine += idle_time_diff
        #     print(f'Idle Time difference: {idle_time_diff}, Adjusted Reward Machine: {reward_machine}')

        self.previous_makespan = makespan
        # self.previous_idle_time = idle_time

        # 에피소드 종료 후 초기 보상 및 패널티 계산
        early_start_bonus, penalty = calculate_early_start_bonus(self.agent_actions, self)

        # print(f'Penalty for unused machines: {penalty}')
        # print(f'early_start_bonus: {early_start_bonus}')

        reward_machine += early_start_bonus * 2  # 초기 보상 크기는 필요에 따라 조정
        reward_task += early_start_bonus * 2  # 초기 보상 크기는 필요에 따라 조정

        total_reward_task += reward_task
        total_reward_machine += reward_machine

        # print(f'Total rewards: reward_task: {total_reward_task}, reward_machine: {total_reward_machine}')

        return total_reward_task, total_reward_machine


def calculate_early_start_bonus(agent_actions, env):
    job_start_times = {job: 0 for job in range(env.n_jobs)}
    machine_avail_times = {machine: 0 for machine in range(env.n_machines)}
    machine_start_times = {machine: -1 for machine in range(env.n_machines)}

    # 각 머신의 시작 시간을 초기화
    for job, op, machine in agent_actions:
        processing_time = next((t for m, t in env.machine_sequence[job][op] if m == machine), None)
        if processing_time is None:
            continue  # 기계가 작업을 처리할 수 없는 경우 건너뜁니다.

        start_time = max(job_start_times[job], machine_avail_times[machine])
        end_time = start_time + processing_time
        job_start_times[job] = end_time
        machine_avail_times[machine] = end_time

        if machine_start_times[machine] == -1:
            machine_start_times[machine] = start_time

    # print(f"Machine start times: {machine_start_times}")
    # print(f"Agent actions: {agent_actions}")

    # 초기 보상 및 패널티 계산
    early_start_bonus = 0
    penalty = 0
    for machine, start_time in machine_start_times.items():
        if start_time == 0:
            early_start_bonus += 0 # start_time이 0인 갯수 * 숫자, 원래는 1로함.
        else:
            penalty -= 1 # -1에서 -5으로 변경

    # 모든 머신의 시작 시간이 0인 경우 추가 보너스를 줌
    if all(start_time == 0 for start_time in machine_start_times.values()):
        early_start_bonus += 5

    # print(f"Penalty for unused machines: {penalty}")
    # print(f"early_start_bonus: {early_start_bonus}")

    return early_start_bonus + penalty, penalty

    

def calculate_makespan_reward(agent_actions, env):
    job_start_times = {job: 0 for job in range(env.n_jobs)}
    machine_avail_times = {machine: 0 for machine in range(env.n_machines)}
    makespan = 0

    for job, op, machine in agent_actions:
        processing_time = next((t for m, t in env.machine_sequence[job][op] if m == machine), None)
        if processing_time is None:
            continue  # 기계가 작업을 처리할 수 없는 경우 건너뜁니다.

        start_time = max(job_start_times[job], machine_avail_times[machine])
        end_time = start_time + processing_time
        job_start_times[job] = end_time
        machine_avail_times[machine] = end_time
        makespan = max(makespan, end_time)

    return makespan


def calculate_idle_time(agent_actions, env):
    machine_end_times = {m: 0 for m in range(env.n_machines)}
    job_start_times = {job: 0 for job in range(env.n_jobs)}
    machine_idle_times = {m: 0 for m in range(env.n_machines)}

    for job, op, machine in agent_actions:
        # 선택된 작업과 기계의 처리 시간을 가져옵니다.
        processing_time = next((t for m, t in env.machine_sequence[job][op] if m == machine), None)
        if processing_time is None:
            continue  # 처리 시간이 없는 경우 건너뜁니다.

        start_time = max(job_start_times[job], machine_end_times[machine])
        end_time = start_time + processing_time

        # 기계의 유휴 시간을 계산합니다.
        if start_time > machine_end_times[machine]:
            idle_time = start_time - machine_end_times[machine]
            machine_idle_times[machine] += idle_time

        machine_end_times[machine] = end_time
        job_start_times[job] = end_time

    total_idle_time = sum(machine_idle_times.values())
    return total_idle_time


def calculate_waiting_time(agent_actions, env):
    waiting_time = 0
    job_start_times = {job: 0 for job in range(env.n_jobs)}
    machine_end_times = {machine: 0 for machine in range(env.n_machines)}
    job_operations = {job: [] for job in range(env.n_jobs)}

    for job, op, machine in agent_actions:
        # 선택된 작업과 기계의 처리 시간을 가져옵니다.
        processing_time = next((t for m, t in env.machine_sequence[job][op] if m == machine), None)
        if processing_time is None:
            continue  # 처리 시간이 없는 경우 건너뜁니다.

        start_time = max(job_start_times[job], machine_end_times[machine])

        # 작업의 대기 시간을 계산합니다.
        job_waiting_time = start_time - job_start_times[job]
        waiting_time += job_waiting_time

        job_operations[job].append((op, job_waiting_time))

        end_time = start_time + processing_time
        machine_end_times[machine] = end_time
        job_start_times[job] = end_time

    return waiting_time


def calculate_idle_machine_utilization(agent_actions, env):
    machine_usage = {machine: False for machine in range(env.n_machines)}

    for job, op, machine in agent_actions:
        processing_time = next((t for m, t in env.machine_sequence[job][op] if m == machine), None)
        if processing_time is None:
            continue  # 기계가 작업을 처리할 수 없는 경우 건너뜁니다.
        machine_usage[machine] = True

    all_machines_used = all(machine_usage.values())
    utilization_reward = 1 if all_machines_used else -1

    # 기계 사용 상태 출력
    print(f"Machine Usage: {machine_usage}")
    print(f"All Machines Used: {all_machines_used}")
    print(f"Utilization Reward: {utilization_reward}")

    return utilization_reward

def calculate_current_time(agent_actions, env):
    job_start_times = {job: 0 for job in range(env.n_jobs)}
    machine_avail_times = {machine: 0 for machine in range(env.n_machines)}
    current_time = 0

    for job, op, machine in agent_actions:
        processing_time = next((t for m, t in env.machine_sequence[job][op] if m == machine), None)
        if processing_time is None:
            continue

        start_time = max(job_start_times[job], machine_avail_times[machine])
        end_time = start_time + processing_time
        job_start_times[job] = end_time
        machine_avail_times[machine] = end_time
        current_time = max(current_time, end_time)

    return current_time

def calculate_machine_available_time(agent_actions, env):
    job_start_times = {job: 0 for job in range(env.n_jobs)}
    machine_avail_times = {machine: 0 for machine in range(env.n_machines)}

    for job, op, machine in agent_actions:
        processing_time = next((t for m, t in env.machine_sequence[job][op] if m == machine), None)
        if processing_time is None:
            continue

        start_time = max(job_start_times[job], machine_avail_times[machine])
        end_time = start_time + processing_time
        job_start_times[job] = end_time
        machine_avail_times[machine] = end_time

    return machine_avail_times