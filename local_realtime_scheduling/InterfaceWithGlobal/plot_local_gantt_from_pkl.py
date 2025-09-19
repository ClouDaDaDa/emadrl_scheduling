import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.cm as cm
from local_realtime_scheduling.InterfaceWithGlobal.divide_global_schedule_to_local_from_pkl import LocalSchedule, Local_Job_schedule

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
# plt.rcParams['font.size'] = 13  # Default font size
# plt.rcParams['axes.titlesize'] = 14    # Title font size
# # plt.rcParams['axes.labelsize'] = 13    # Axis label font size
# plt.rcParams['xtick.labelsize'] = 13   # x-axis tick label font size
# plt.rcParams['ytick.labelsize'] = 14   # y-axis tick label font size
# plt.rcParams['legend.fontsize'] = 14   # Legend font size


def plot_local_gantt(local_schedule, save_fig_dir=None):
    """
    Plot a Gantt chart with operations colored by their assigned resources (e.g., machines or transbots).

    Args:
        local_schedule (LocalSchedule): The local schedule containing jobs and operations.
    """
    # Generate unique colors for resources using a colormap
    machine_resource_ids = set(
        f"Machine {operation.assigned_machine}"
        for job in local_schedule.jobs.values()
        for operation in job.operations.values() if operation.is_current_window
    )
    transbot_resource_ids = set(
        f"Transbot {operation.assigned_transbot}"
        for job in local_schedule.jobs.values()
        for operation in job.operations.values()
        if operation.assigned_transbot is not None and operation.is_current_window
    )
    resource_ids = machine_resource_ids | transbot_resource_ids
    # resource_ids = set(
    #     f"Machine {operation.machine_assigned}" if operation.type == "Processing"
    #     else f"Transbot {operation.transbot_assigned}"
    #     for job in local_schedule.jobs.values() for operation in job.operations.values()
    # )
    num_resources = len(resource_ids)
    colormap = plt.colormaps["tab20"] if num_resources <= 20 else cm.get_cmap("nipy_spectral", num_resources)
    resource_color_map = {resource: colormap(i / max(1, num_resources - 1)) for i, resource in enumerate(sorted(resource_ids))}

    time_window = [local_schedule.time_window_start, local_schedule.time_window_end]

    # Collect all end times to determine x-axis range
    max_end_time = time_window[1]
    for job in local_schedule.jobs.values():
        for operation in job.operations.values():
            if operation.is_current_window:
                max_end_time = max(max_end_time, operation.scheduled_finish_processing_time)

    # Create the Gantt chart
    fig, ax = plt.subplots(figsize=(6, 5))

    yticks = []
    yticklabels = []

    for idx, job in local_schedule.jobs.items():
        yticks.append(idx)
        yticklabels.append(f"Job {job.job_id}")

        for operation in job.operations.values():

            if operation.is_current_window:
                # Determine the resource and color
                machine_resource = f"Machine {operation.assigned_machine}"
                # resource_color_map[machine_resource] = (0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1.0)
                machine_color = resource_color_map[machine_resource]
                start_processing_time = operation.scheduled_start_processing_time
                processing_duration = operation.scheduled_finish_processing_time - operation.scheduled_start_processing_time
                # Plot a rectangle for the operation
                ax.barh(idx, processing_duration, left=start_processing_time, color=machine_color,
                        edgecolor="white", align="center")

                if operation.assigned_transbot is not None:
                    transbot_resource = f"Transbot {operation.assigned_transbot}"
                    # resource_color_map[transbot_resource] = (0.7803921568627451, 0.7803921568627451, 0.7803921568627451, 1.0)
                    transbot_color = resource_color_map[transbot_resource]
                    start_transporting_time = operation.scheduled_start_transporting_time
                    transporting_duration = operation.scheduled_finish_transporting_time - operation.scheduled_start_transporting_time
                    # Plot a rectangle for the operation
                    ax.barh(idx, transporting_duration, left=start_transporting_time, color=transbot_color,
                            edgecolor="white", align="center")
            elif not operation.is_lookahead:
                # Determine the resource and color
                machine_color = 'dimgrey'
                start_processing_time = operation.scheduled_start_processing_time
                processing_duration = operation.scheduled_finish_processing_time - operation.scheduled_start_processing_time
                # Plot a rectangle for the operation
                ax.barh(idx, processing_duration, left=start_processing_time, color=machine_color,
                        alpha=0.8,
                        edgecolor="white", align="center")

                if operation.assigned_transbot is not None:
                    transbot_color = 'dimgrey'
                    start_transporting_time = operation.scheduled_start_transporting_time
                    transporting_duration = operation.scheduled_finish_transporting_time - operation.scheduled_start_transporting_time
                    # Plot a rectangle for the operation
                    ax.barh(idx, transporting_duration, left=start_transporting_time, color=transbot_color,
                            alpha=0.8,
                            edgecolor="white", align="center")

            # if operation.is_addition_ops:
            #
            #     # Determine the resource and color
            #     machine_color = 'grey'
            #     start_processing_time = operation.scheduled_start_processing_time
            #     processing_duration = operation.scheduled_finish_processing_time - operation.scheduled_start_processing_time
            #     # Plot a rectangle for the operation
            #     ax.barh(idx, processing_duration, left=start_processing_time, color=machine_color,
            #             edgecolor="white", align="center")
            #
            #     if operation.assigned_transbot is not None:
            #         transbot_color = 'lightgrey'
            #         start_transporting_time = operation.scheduled_start_transporting_time
            #         transporting_duration = operation.scheduled_finish_transporting_time - operation.scheduled_start_transporting_time
            #         # Plot a rectangle for the operation
            #         ax.barh(idx, transporting_duration, left=start_transporting_time, color=transbot_color,
            #                 edgecolor="white", align="center")
            # else:
            #     # Determine the resource and color
            #     machine_resource = f"Machine {operation.assigned_machine}"
            #     machine_color = resource_color_map[machine_resource]
            #     start_processing_time = operation.scheduled_start_processing_time
            #     processing_duration = operation.scheduled_finish_processing_time - operation.scheduled_start_processing_time
            #     # Plot a rectangle for the operation
            #     ax.barh(idx, processing_duration, left=start_processing_time, color=machine_color,
            #             edgecolor="white", align="center")
            #
            #     if operation.assigned_transbot is not None:
            #         transbot_resource = f"Transbot {operation.assigned_transbot}"
            #         transbot_color = resource_color_map[transbot_resource]
            #         start_transporting_time = operation.scheduled_start_transporting_time
            #         transporting_duration = operation.scheduled_finish_transporting_time - operation.scheduled_start_transporting_time
            #         # Plot a rectangle for the operation
            #         ax.barh(idx, transporting_duration, left=start_transporting_time, color=transbot_color,
            #                 edgecolor="white", align="center")

    # Configure axes
    ax.set_xlim(time_window[0], max_end_time)
    ax.set_ylim(-0.5, len(local_schedule.jobs) - 0.5)
    # ax.set_xlim(time_window[0], 654)
    # ax.set_ylim(-0.5, 4.5)
    ax.set_xlabel("Time")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_title("Local Schedule Gantt Chart")
    ax.grid(True, alpha=0.3)

    if time_window[1] < max_end_time:
        # Add a red dashed line at the end of the time window
        ax.axvline(x=time_window[1], color="red", linestyle="--", linewidth=2, label="Time Window End")
    # ax.axvline(x=900, color="red", linestyle="--", linewidth=2, label="Time Window End")

    # Create a legend for the resources
    legend_patches = [Patch(color=color, label=resource) for resource, color in resource_color_map.items()]
    # ax.legend(handles=legend_patches, title="Resources", loc="upper right", bbox_to_anchor=(1.15, 1))
    ax.legend(handles=legend_patches, title="Resources", loc="upper right",
              bbox_to_anchor=(1.25, 1),
              borderaxespad=0.,
              ncol=1
              )

    plt.tight_layout()
    if save_fig_dir is not None:
        plt.savefig(save_fig_dir + "_gantt.png")
    plt.show()


def plot_local_gantt_by_resource(local_schedule, save_fig_dir=None):
    """
    Plot a Gantt chart with the y-axis representing manufacturing resources (e.g., machines or transbots),
    sorted by resource type and ID (e.g., Transbot 0, Transbot 1, ..., Machine 0, Machine 1, ...).

    Args:
        local_schedule (LocalSchedule): The local schedule containing jobs and operations.
    """

    # Generate unique colors for jobs using a colormap
    unique_job_ids = set(operation.job_id for job in local_schedule.jobs.values() for operation in job.operations.values())
    num_jobs = len(unique_job_ids)
    if num_jobs <= 20:
        # Use a predefined colormap if the number of jobs is small
        colormap = plt.colormaps["tab20"]
        # job_color_map = {job_id: colormap(i / 20) for i, job_id in enumerate(unique_job_ids)}
        job_color_map = {job_id: colormap(job_id / 20) for i, job_id in enumerate(unique_job_ids)}
    else:
        # Dynamically generate colors if the number of jobs exceeds 20
        colors = cm.get_cmap("hsv", num_jobs)
        job_color_map = {job_id: colors(i / num_jobs) for i, job_id in enumerate(unique_job_ids)}

    # Group operations by resource
    resource_operations = {}
    for job in local_schedule.jobs.values():
        for operation in job.operations.values():

            if operation.assigned_transbot is not None:
                transbot_resource = f"Transbot {operation.assigned_transbot}"
                if transbot_resource not in resource_operations:
                    resource_operations[transbot_resource] = []
                resource_operations[transbot_resource].append(operation)

            if operation.assigned_machine is not None:
                machine_resource = f"Machine {operation.assigned_machine}"
                if machine_resource not in resource_operations:
                    resource_operations[machine_resource] = []
                resource_operations[machine_resource].append(operation)

            # if operation.type == "Processing":
            #     resource = f"Machine {operation.machine_assigned}"
            # else:
            #     resource = f"Transbot {operation.transbot_assigned}"
            # if resource not in resource_operations:
            #     resource_operations[resource] = []
            # resource_operations[resource].append(operation)

    # Sort resources by type and ID
    sorted_resources = sorted(
        resource_operations.keys(),
        key=lambda x: (x.split()[0], int(x.split()[1]))
    )

    # Collect all end times to determine x-axis range
    max_end_time = local_schedule.time_window_end
    for job in local_schedule.jobs.values():
        for operation in job.operations.values():
            max_end_time = max(max_end_time, operation.scheduled_finish_processing_time)

    # Create the Gantt chart
    fig, ax = plt.subplots(figsize=(8, 6))

    yticks = []
    yticklabels = []

    # Plot each resource's operations
    for idx, resource in enumerate(sorted_resources):
        yticks.append(idx)
        yticklabels.append(resource)

        operations = resource_operations[resource]
        for operation in operations:
            job_id = operation.job_id
            color = job_color_map[job_id]

            start_processing_time = operation.scheduled_start_processing_time
            end_processing_time = operation.scheduled_finish_processing_time
            processing_duration = end_processing_time - start_processing_time

            # Plot a rectangle for the operation
            ax.barh(idx, processing_duration, left=start_processing_time, color=color,
                    edgecolor="white", align="center")
            # ax.barh(idx, duration, left=start_time, color=color, edgecolor=color, align="center")

            if operation.assigned_transbot is not None:
                start_transporting_time = operation.scheduled_start_transporting_time
                end_transporting_time = operation.scheduled_finish_transporting_time
                transporting_duration = end_transporting_time - start_transporting_time

                # Plot a rectangle for the operation
                ax.barh(idx, transporting_duration, left=start_transporting_time, color=color, edgecolor="white",
                        align="center")

    # Configure axes
    ax.set_xlim(local_schedule.time_window_start, max_end_time)
    ax.set_ylim(-0.5, len(sorted_resources) - 0.5)
    ax.set_xlabel("Time")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_title("Local Schedule Gantt Chart by Resource")

    # Add a red dashed line at the end of the time window
    ax.axvline(x=local_schedule.time_window_end, color="red", linestyle="--", linewidth=2, label="Time Window End")

    # Create a legend for the jobs
    legend_patches = [Patch(color=color, label=f"Job {job_id}") for job_id, color in job_color_map.items()]
    ax.legend(handles=legend_patches, title="Jobs", loc="upper right", bbox_to_anchor=(1.15, 1))

    current_ticks = list(plt.xticks()[0])
    current_labels = list(plt.xticks()[1])

    current_ticks.append(max_end_time)
    current_labels.append(str(int(max_end_time)))

    plt.xticks(current_ticks, current_labels)

    plt.xlim(local_schedule.time_window_start, max_end_time * 1.02)

    plt.tight_layout()
    if save_fig_dir is not None:
        plt.savefig(save_fig_dir + "_gantt_by_resource.png")
    plt.show()


# Example usage
if __name__ == "__main__":
    from configs import dfjspt_params
    import os

    n_jobs = 10
    instance_id = 100
    current_window = 0
    num_ops = 45
    for i in range(1):

        # Load the LocalSchedule from the pkl file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # local_schedule_file_dir = os.path.dirname(current_dir) + \
        #                       "/InterfaceWithGlobal/local_schedules/local_schedule_" + \
        #                       f"J{dfjspt_params.n_jobs}M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}" \
        #                       + f"I{dfjspt_params.current_scheduling_instance_id}_window{dfjspt_params.time_window_size}" \
        #                       + f"/window_{dfjspt_params.current_window}"

        filename = f'testing/local_schedule_J{n_jobs}I{instance_id}_{current_window}_ops{num_ops}'

        local_schedule_file_dir = os.path.dirname(current_dir) + \
            "/InterfaceWithGlobal/local_schedules" + \
            f"/M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}W{dfjspt_params.time_window_size}/" \
            + filename
            # + f"local_schedule_J{n_jobs}I{instance_id}" \
            # + f"_{current_window}_ops{86}"

        local_schedule_file = local_schedule_file_dir + f".pkl"

        with open(local_schedule_file,
                  "rb") as file:
            local_schedule = pickle.load(file)
        # plot_local_gantt_by_resource(local_schedule,
        #                              save_fig_dir=local_schedule_file_dir
        #                              )
        plot_local_gantt(local_schedule,
            # save_fig_dir=local_schedule_file_dir
        )

        current_window += 1
