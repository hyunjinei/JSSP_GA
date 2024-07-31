import os
import datetime

class Run_Config_Makespan:
    def __init__(self, n_job, n_machine, n_op,
                 print_console=True,
                 show_gantt=False,
                 save_gantt=True,
                 title=None):

        self.n_job = n_job
        self.n_machine = n_machine
        self.n_op = n_op

        self.print_console = print_console
        self.show_gantt = show_gantt
        self.save_gantt = save_gantt

        self.gantt_title = title
        self.dataset_filename = None  # 데이터셋 파일 이름을 저장할 속성

        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        folder_name = 'result_makespan'
        self.save_path = os.path.join(script_dir, folder_name)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        now = datetime.datetime.now()
        self.now = now.strftime('%Y-%m-%d-%H-%M-%S')
        self.filename = {
            'gantt': os.path.join(self.save_path, f'makespan_{self.now}.png')
        }

    def set_dataset_filename(self, filename):
        self.dataset_filename = filename
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        self.filename['gantt'] = os.path.join(self.save_path, f'makespan_{base_filename}_{self.now}.png')

    def update_gantt_filename(self, job_order):
        order_str = '-'.join(job_order)
        self.filename['gantt'] = os.path.join(self.save_path, f'makespan_{order_str}_{self.now}.png')