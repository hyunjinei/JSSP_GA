import pandas as pd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO

simmode = ''

def color(row):
    c_dict = {'Part0': '#0000ff', 'Part1': '#ffa500', 'Part2': '#006400',
              'Part3': '#ff0000', 'Part4': '#cdc0b0', 'Part5': '#66cdaa',
              'Part6': '#1abc9c', 'Part7': '#a52a2a', 'Part8': '#5bc0de',
              'Part9': '#fc8c84'}
    return c_dict[row['Job'][0:5]]

def Gantt(machine_log, config, makespan):
    machine_log['color'] = machine_log.apply(color, axis=1)
    machine_log['Delta'] = machine_log['Finish'] - machine_log['Start']

    fig, ax = plt.subplots(1, figsize=(16*0.8, 9*0.8))
    ax.barh(machine_log.Machine, machine_log.Delta, left=machine_log.Start, color=machine_log.color, edgecolor='black')

    ##### LEGENDS #####
    c_dict = {'Part0': '#0000ff', 'Part1': '#ffa500', 'Part2': '#006400',
              'Part3': '#ff0000', 'Part4': '#cdc0b0', 'Part5': '#66cdaa',
              'Part6': '#1abc9c', 'Part7': '#a52a2a', 'Part8': '#5bc0de',
              'Part9': '#fc8c84'}
    legend_elements = [Patch(facecolor=c_dict[i], label=i) for i in c_dict]
    plt.legend(handles=legend_elements)
    plt.title(config.gantt_title, size=24)

    # Set x-axis limit to makespan + 10
    ax.set_xlim(0, makespan + 10)

    # Mark the makespan value on the x-axis
    plt.text(makespan, -1, f'{makespan}', color='black', ha='center', va='center')

    plt.xlabel('Time')
    plt.ylabel('Machine')

    ##### TICKS #####
    if config.show_gantt:
        plt.show()

    # Save the figure as an image file
    if config.save_gantt:
        fig.savefig(config.filename['gantt'], format='png')

    # Create a BytesIO object
    image_bytes_io = BytesIO()
    fig.savefig(image_bytes_io, format='png')  # This is different from saving file as .png

    # Get the image bytes
    image_bytes = image_bytes_io.getvalue()

    return image_bytes
