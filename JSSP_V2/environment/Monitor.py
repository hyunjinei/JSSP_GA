import pandas as pd

# region Monitor
class Monitor(object):
    def __init__(self, config):
        self.config = config  ## Event tracer 저장 경로
        self.time = list()
        self.event = list()
        self.part = list()
        self.process_name = list()
        self.machine_name = list()

    def record(self, time, process, machine, part_name=None, event=None):
        self.time.append(time)
        self.event.append(event)
        self.part.append(part_name)  # string
        self.process_name.append(process)
        self.machine_name.append(machine)

    def save_event_tracer(self, file_path=None):
        event_tracer = pd.DataFrame(columns=['Time', 'Event', 'Part', 'Process', 'Machine'])
        event_tracer['Time'] = self.time
        event_tracer['Event'] = self.event
        event_tracer['Part'] = self.part
        event_tracer['Process'] = self.process_name
        event_tracer['Machine'] = self.machine_name
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
        command = " " + command + " "
        if object == 'Single Part':
            if operation.process_type == 0:
                pass
                # print(str(env.now) + '\t' + str(operation.name) + command + 'M' + str(operation.machine))
        elif object == 'Single Job':
            if operation.part_name == 'Part0_0':
                pass
                # print(str(env.now) + '\t' + str(operation.name) + command + 'M' + str(operation.machine))
        elif object == 'Entire Process':
            pass
            # print(str(env.now) + '\t' + str(operation.name) + command + 'M' + str(operation.machine))
        elif object == 'Machine':
            pass
            # print_by_machine(env, part)


def print_by_machine(env, part):
    if part.op[part.step].machine == 0:
        print(str(env.now) + '\t\t\t\t' + str(part.op[part.step].name))
    elif part.op[part.step].machine == 1:
        print(str(env.now) + '\t\t\t\t\t\t\t' + str(part.op[part.step].name))
    elif part.op[part.step].machine == 2:
        print(str(env.now) + '\t\t\t\t\t\t\t\t\t\t' + str(part.op[part.step].name))
    elif part.op[part.step].machine == 3:
        print(str(env.now) + '\t\t\t\t\t\t\t\t\t\t\t\t\t' + str(part.op[part.step].name))
    elif part.op[part.step].machine == 4:
        print(str(env.now) + '\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t' + str(part.op[part.step].name))
    else:
        print()
