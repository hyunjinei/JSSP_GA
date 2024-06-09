import os
import datetime

class Run_Config:
    def __init__(self, n_job, n_machine, n_op, population_size, generations,
                 print_console=False,
                 save_log=False,
                 save_machinelog=False,
                 show_gantt=False,
                 save_gantt=False,
                 show_gui=False,
                 trace_object='Process4', title=None,
                 tabu_search_iterations=100, hill_climbing_iterations=100, simulated_annealing_iterations=100,
                 ga_index=0):

        self.n_job = n_job
        self.n_machine = n_machine
        self.n_op = n_op

        self.trace_object = trace_object
        self.trace_type = 'Single Part'

        self.print_console = print_console
        self.save_log = save_log
        self.save_machinelog = save_machinelog
        self.show_gantt = show_gantt
        self.save_gantt = save_gantt
        self.show_gui = show_gui

        self.num_parts = 1
        self.IAT = float("inf")

        self.simul_time = 10000
        self.dispatch_mode = 'Manual'
        self.gantt_title = title

        self.population_size = population_size
        self.generations = generations
        self.dataset_filename = None  # 새로운 속성 추가

        self.tabu_search_iterations = tabu_search_iterations
        self.hill_climbing_iterations = hill_climbing_iterations
        self.simulated_annealing_iterations = simulated_annealing_iterations

        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Get the directory of the current script
        folder_name = 'result'  # Define the folder name
        self.save_path = os.path.join(script_dir, folder_name)  # Construct the full path to the folder
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        ga_generations_path = os.path.join(self.save_path, 'ga_generations')
        if not os.path.exists(ga_generations_path):
            os.makedirs(ga_generations_path)
                        
        now = datetime.datetime.now()
        self.now = now.strftime('%Y-%m-%d-%H-%M-%S')
        self.filename = {
            'log': os.path.join(self.save_path, f'GA{ga_index}_{self.now}.csv'),
            'machine': os.path.join(self.save_path, f'GA{ga_index}_{self.now}_machine.csv'),
            'gantt': os.path.join(self.save_path, f'GA{ga_index}_{self.now}.png'),
            'csv': os.path.join(ga_generations_path, f'GA{ga_index}_{self.now}.csv')  # 추가된 부분
        }
