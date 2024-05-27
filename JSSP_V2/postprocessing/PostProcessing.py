import pandas as pd
import os
from collections import OrderedDict, defaultdict

def generate_machine_log(config):
    # 첫 번째 열을 'Time' 열로 인식하도록 설정
    df = pd.read_csv(config.filename['log'], names=['Time', 'Event', 'Part', 'Process', 'Machine'], skiprows=1)

    # Filter 'Started' and 'Finished' events
    df_started = df[df['Event'] == 'Started'].drop(['Event', 'Process'], axis=1).reset_index(drop=True)
    df_finished = df[df['Event'] == 'Finished'].drop(['Event', 'Process'], axis=1).reset_index(drop=True)

    machine_start = []
    machine_finish = []
    for i in range(config.n_machine):
        machine_start.append(df_started[(df_started['Machine'] == 'M' + str(i))])
        machine_finish.append(df_finished[(df_finished['Machine'] == 'M' + str(i))])

        machine_start[i].reset_index(drop=True, inplace=True)
        machine_finish[i].reset_index(drop=True, inplace=True)
    data = []

    for i in range(config.n_machine):
        for j in range(len(machine_finish[i])):
            temp = {'Machine': i,
                    'Job': machine_start[i].loc[j, 'Part'],
                    'Start': int(float(machine_start[i].loc[j, 'Time'])),  # 수정된 부분
                    'Finish': int(float(machine_finish[i].loc[j, 'Time'])),  # 수정된 부분
                    'Delta': int(float(machine_finish[i].loc[j, 'Time']) - float(machine_start[i].loc[j, 'Time']))}  # 수정된 부분
            data.append(temp)

    data = pd.DataFrame(data)
    data = data.sort_values(by=['Start'])
    data.reset_index(drop=True, inplace=True)
    if config.save_machinelog:
        data.to_csv(config.filename['machine'], index=False)
    return data

