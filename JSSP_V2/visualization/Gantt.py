import pandas as pd
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
import matplotlib.colors as mcolors  # 추가된 부분

simmode = ''

def generate_colors(n):
    """Generate n distinct colors."""
    colors = plt.cm.get_cmap('tab10', n).colors
    return [mcolors.rgb2hex(c) for c in colors]  # 수정된 부분

def color(row, color_map):
    return color_map[row['Job'][0:5]]

def Gantt(machine_log, config, makespan):
    unique_parts = machine_log['Job'].apply(lambda x: x[:5]).unique()
    num_parts = len(unique_parts)
    
    # Generate a color map for the parts
    colors = generate_colors(num_parts)
    color_map = {f'Part{i}': colors[i] for i in range(num_parts)}

    machine_log['color'] = machine_log.apply(lambda row: color(row, color_map), axis=1)
    machine_log['Delta'] = machine_log['Finish'] - machine_log['Start']

    fig, ax = plt.subplots(1, figsize=(16*0.8, 9*0.8))
    ax.barh(machine_log.Machine, machine_log.Delta, left=machine_log.Start, color=machine_log.color, edgecolor='black')

    ##### LEGENDS #####
    legend_elements = [Patch(facecolor=color_map[f'Part{i}'], label=f'Part{i}') for i in range(num_parts)]
    plt.legend(handles=legend_elements)
    plt.title(config.gantt_title, size=24)

    # Set x-axis limit to makespan + 10
    ax.set_xlim(0, makespan + 10)

    # Mark the makespan value on the x-axis
    plt.text(makespan, -1, f'{makespan}', color='black', ha='center', va='center')

    # Mark the makespan value on the x-axis at the top
    plt.text(makespan, ax.get_ylim()[1], f'Max Makespan: {makespan}', color='red', ha='right', va='top')

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
