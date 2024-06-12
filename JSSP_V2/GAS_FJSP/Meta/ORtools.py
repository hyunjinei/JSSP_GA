from ortools.sat.python import cp_model
import copy

class ORToolsOptimizer:
    def __init__(self, num_iterations=100):
        self.num_iterations = num_iterations

    def optimize(self, individual, config):
        print("OR-Tools 시작")
        best_individual = self.create_new_individual(individual, individual.seq, config)

        model = cp_model.CpModel()
        horizon = sum(task[1] for job in config.op_data for task in job)

        all_tasks = {}
        machine_to_intervals = {}

        for job_id, job in enumerate(config.op_data):
            for task_id, task in enumerate(job):
                machine, duration = task

                suffix = f'_{job_id}_{task_id}'
                start_var = model.NewIntVar(0, horizon, 'start' + suffix)
                end_var = model.NewIntVar(0, horizon, 'end' + suffix)
                interval_var = model.NewIntervalVar(start_var, duration, end_var, 'interval' + suffix)

                all_tasks[job_id, task_id] = (start_var, end_var, interval_var)

                if machine not in machine_to_intervals:
                    machine_to_intervals[machine] = []
                machine_to_intervals[machine].append(interval_var)

        for machine in machine_to_intervals:
            model.AddNoOverlap(machine_to_intervals[machine])

        for job_id, job in enumerate(config.op_data):
            for task_id in range(len(job) - 1):
                model.Add(all_tasks[job_id, task_id + 1][0] >= all_tasks[job_id, task_id][1])

        obj_var = model.NewIntVar(0, horizon, 'makespan')
        model.AddMaxEquality(obj_var, [all_tasks[job_id, len(job) - 1][1] for job_id, job in enumerate(config.op_data)])

        model.Minimize(obj_var)

        solver = cp_model.CpSolver()
        status = solver.Solve(model)

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            new_seq = []
            for job_id, job in enumerate(config.op_data):
                job_schedule = sorted((solver.Value(all_tasks[job_id, task_id][0]), task_id) for task_id in range(len(job)))
                new_seq.extend(job_id * config.n_machine + task_id for _, task_id in job_schedule)

            optimized_individual = self.create_new_individual(individual, new_seq, config)
            return optimized_individual
        else:
            print("No feasible solution found.")
            return best_individual

    def create_new_individual(self, individual, new_seq, config):
        new_individual = copy.deepcopy(individual)
        new_individual.seq = new_seq
        new_individual.job_seq = new_individual.get_repeatable()
        new_individual.feasible_seq = new_individual.get_feasible()
        new_individual.machine_order = new_individual.get_machine_order()
        new_individual.makespan, new_individual.mio_score = new_individual.evaluate(new_individual.machine_order)
        new_individual.calculate_fitness(config.target_makespan)
        return new_individual
 
