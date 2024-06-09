import pandas as pd

# region Monitor
class Monitor(object):
    def __init__(self, config):
        self.config = config  # Event tracer 저장 경로
        self.time = []
        self.event = []
        self.part = []
        self.process_name = []
        self.machine_name = []

    def record(self, time, process, machine, part_name=None, event=None):
        if time is not None and process is not None and machine is not None:
            self.time.append(time)
            self.event.append(event)
            self.part.append(part_name)  # string
            self.process_name.append(process)
            self.machine_name.append(machine)

    def save_event_tracer(self, file_path=None):
        event_tracer = pd.DataFrame({
            'Time': self.time,
            'Event': self.event,
            'Part': self.part,
            'Process': self.process_name,
            'Machine': self.machine_name
        })
        if self.config.save_log:
            if file_path:
                event_tracer.to_csv(file_path, index=False)
            else:
                event_tracer.to_csv(self.config.filename['log'], index=False)

        return event_tracer
# endregion

def monitor_by_console(console_mode, env, part, object='Single Part', command=''):
    if console_mode:
        operation = part.op[part.step]
        command = f" {command} "
        if object == 'Single Part' and operation.process_type == 0:
            pass
        elif object == 'Single Job' and operation.part_name == 'Part0_0':
            pass
        elif object == 'Entire Process':
            pass
        elif object == 'Machine':
            print_by_machine(env, part)

def print_by_machine(env, part):
    machine_idx = part.op[part.step].machine
    machine_positions = ["\t\t\t\t", "\t\t\t\t\t\t\t", "\t\t\t\t\t\t\t\t\t\t", "\t\t\t\t\t\t\t\t\t\t\t\t\t", "\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t"]
    if 0 <= machine_idx < len(machine_positions):
        print(f"{env.now}{machine_positions[machine_idx]}{part.op[part.step].name}")
    else:
        print()
