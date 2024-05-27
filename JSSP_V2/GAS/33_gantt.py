import matplotlib.pyplot as plt
import pandas as pd

# Define the job data
jobs_data = {
    "Job 0": [(2, 13), (3, 31), (1, 17)],
    "Job 1": [(2, 31), (3, 11), (1, 19)],
    "Job 2": [(1, 23), (2, 11), (3, 29)]
}

# Initialize the start times for each machine and job
machine_start_times = {machine: 0 for machine in range(1, 4)}
job_end_times = {job: 0 for job in jobs_data.keys()}

# Create a dataframe to store the Gantt chart data
gantt_data = []

# Fill the Gantt chart data
for job, tasks in jobs_data.items():
    for machine, duration in tasks:
        start_time = max(machine_start_times[machine], job_end_times[job])
        end_time = start_time + duration
        gantt_data.append([job, machine, start_time, end_time])
        machine_start_times[machine] = end_time
        job_end_times[job] = end_time

# Convert to DataFrame
gantt_df = pd.DataFrame(gantt_data, columns=["Job", "Machine", "Start", "End"])

# Plot the Gantt chart
fig, ax = plt.subplots(figsize=(10, 6))
colors = {"Job 0": 'blue', "Job 1": 'orange', "Job 2": 'green'}

for job, machine, start, end in gantt_data:
    ax.barh(machine, end - start, left=start, color=colors[job], edgecolor='black', label=job if job not in [i[0] for i in gantt_data] else "")

# Add labels
ax.set_xlabel("Time")
ax.set_ylabel("Machine")
ax.set_yticks([1, 2, 3])
ax.set_yticklabels(["Machine 1", "Machine 2", "Machine 3"])
ax.set_title("Gantt Chart for JSSP")
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

plt.show()
