import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.cm as cm

from local_realtime_scheduling.Environment.ExecutionResult import LocalResult, Local_Job_result,Operation_result

def plot_local_result_gantt(local_result, save_fig_dir=None):
    """
    Plot a Gantt chart with operations colored by their assigned resources (e.g., machines or transbots).

    Args:
        local_result (LocalResult): The local result containing jobs and operations.
    """
    # Generate unique colors for resources using a colormap
    machine_resource_ids = set(
        f"Machine {operation.assigned_machine}"
        for job in local_result.jobs.values()
        for operation in job.operations.values()
    )
    transbot_resource_ids = set(
        f"Transbot {operation.assigned_transbot}"
        for job in local_result.jobs.values()
        for operation in job.operations.values()
        if operation.assigned_transbot is not None
    )
    resource_ids = machine_resource_ids | transbot_resource_ids
    num_resources = len(resource_ids)
    colormap = plt.colormaps["tab20"] if num_resources <= 20 else cm.get_cmap("hsv", num_resources)
    resource_color_map = {resource: colormap(i / num_resources) for i, resource in enumerate(sorted(resource_ids))}

    time_window = [local_result.time_window_start, local_result.actual_local_makespan]

    # Collect all end times to determine x-axis range
    max_end_time = time_window[1]
    for job in local_result.jobs.values():
        for operation in job.operations.values():
            max_end_time = max(max_end_time, operation.actual_finish_processing_time)

    # Create the Gantt chart
    fig, ax = plt.subplots(figsize=(12, 8))

    yticks = []
    yticklabels = []

    for idx, job in local_result.jobs.items():
        yticks.append(idx)
        yticklabels.append(f"Job {job.job_id}")

        for operation in job.operations.values():
            # Determine the resource and color
            machine_resource = f"Machine {operation.assigned_machine}"
            machine_color = resource_color_map[machine_resource]
            start_processing_time = operation.actual_start_processing_time
            processing_duration = operation.actual_finish_processing_time - operation.actual_start_processing_time
            # Plot a rectangle for the operation
            ax.barh(idx, processing_duration, left=start_processing_time, color=machine_color,
                    edgecolor="white", align="center")

            if operation.assigned_transbot is not None:
                transbot_resource = f"Transbot {operation.assigned_transbot}"
                transbot_color = resource_color_map[transbot_resource]
                start_transporting_time = operation.actual_start_transporting_time
                transporting_duration = operation.actual_finish_transporting_time - operation.actual_start_transporting_time
                # Plot a rectangle for the operation
                ax.barh(idx, transporting_duration, left=start_transporting_time, color=transbot_color,
                        edgecolor="white", align="center")

    # Configure axes
    ax.set_xlim(time_window[0], max_end_time)
    ax.set_ylim(-0.5, len(local_result.jobs) - 0.5)
    ax.set_xlabel("Time")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_title("Local Result Gantt Chart")

    # # Add a red dashed line at the end of the time window
    # ax.axvline(x=time_window[1], color="red", linestyle="--", linewidth=2, label="Time Window End")

    # Create a legend for the resources
    legend_patches = [Patch(color=color, label=resource) for resource, color in resource_color_map.items()]
    ax.legend(handles=legend_patches, title="Resources", loc="upper right", bbox_to_anchor=(1.15, 1))

    plt.tight_layout()
    if save_fig_dir is not None:
        plt.savefig(save_fig_dir + "_gantt.png")
    plt.show()


def plot_local_result_gantt_by_resource(local_result, save_fig_dir=None):
    """
    Plot a Gantt chart with the y-axis representing manufacturing resources (e.g., machines or transbots),
    sorted by resource type and ID (e.g., Transbot 0, Transbot 1, ..., Machine 0, Machine 1, ...).

    Args:
        local_result (LocalResult): The local schedule containing jobs and operations.
    """

    # Generate unique colors for jobs using a colormap
    unique_job_ids = set(operation.job_id for job in local_result.jobs.values() for operation in job.operations.values())
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
    for job in local_result.jobs.values():
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

    # Sort resources by type and ID
    sorted_resources = sorted(
        resource_operations.keys(),
        key=lambda x: (x.split()[0], int(x.split()[1]))
    )

    # Collect all end times to determine x-axis range
    max_end_time = local_result.actual_local_makespan
    # for job in local_result.jobs.values():
    #     for operation in job.operations.values():
    #         max_end_time = max(max_end_time, operation.actual_finish_processing_time)

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

            start_processing_time = operation.actual_start_processing_time
            end_processing_time = operation.actual_finish_processing_time
            processing_duration = end_processing_time - start_processing_time

            # Plot a rectangle for the operation
            ax.barh(idx, processing_duration, left=start_processing_time, color=color,
                    edgecolor="white", align="center")
            # ax.barh(idx, duration, left=start_time, color=color, edgecolor=color, align="center")

            if operation.assigned_transbot is not None:
                start_transporting_time = operation.actual_start_transporting_time
                end_transporting_time = operation.actual_finish_transporting_time
                transporting_duration = end_transporting_time - start_transporting_time

                # Plot a rectangle for the operation
                ax.barh(idx, transporting_duration, left=start_transporting_time, color=color, edgecolor="white",
                        align="center")

    # Configure axes
    ax.set_xlim(local_result.time_window_start, max_end_time)
    ax.set_ylim(-0.5, len(sorted_resources) - 0.5)
    ax.set_xlabel("Time")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_title("Local Schedule Gantt Chart by Resource")

    # Create a legend for the jobs
    legend_patches = [Patch(color=color, label=f"Job {job_id}") for job_id, color in job_color_map.items()]
    ax.legend(handles=legend_patches, title="Jobs", loc="upper right", bbox_to_anchor=(1.15, 1))

    current_ticks = list(plt.xticks()[0])
    current_labels = list(plt.xticks()[1])

    current_ticks.append(max_end_time)
    current_labels.append(str(int(max_end_time)))

    plt.xticks(current_ticks, current_labels)

    plt.xlim(local_result.time_window_start, max_end_time * 1.02)

    plt.tight_layout()
    if save_fig_dir is not None:
        plt.savefig(save_fig_dir + "_gantt_by_resource.png")
    plt.show()


# Example usage
if __name__ == "__main__":
    from configs import dfjspt_params
    import os

    # Load the LocalSchedule from the pkl file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_result_file_dir = current_dir + \
                            "/local_result_" + \
                            f"J{dfjspt_params.n_jobs}M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}" \
                            + f"I0_window{dfjspt_params.time_window_size}/window_{dfjspt_params.current_window}"

    init_schedule_local_result_file_dir = current_dir + \
                            "/local_result_" + \
                            f"J{dfjspt_params.n_jobs}M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}" \
                            + f"I0_window{dfjspt_params.time_window_size}/window_{dfjspt_params.current_window}_init_schedule"

    drl_local_result_file_dir = current_dir + \
                                          "/local_result_" + \
                                          f"J{dfjspt_params.n_jobs}M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}" \
                                          + f"I0_window{dfjspt_params.time_window_size}/window_{dfjspt_params.current_window}_DRL"

    # result_file_name = local_result_file_dir + ".pkl"
    # result_file_name = init_schedule_local_result_file_dir + ".pkl"
    result_file_name = drl_local_result_file_dir + ".pkl"

    with open(result_file_name,
              "rb") as file:
        local_result = pickle.load(file)
    # plot_local_result_gantt_by_resource(local_result,
    #                                     # save_fig_dir=init_schedule_local_result_file_dir
    #                                     save_fig_dir=local_result_file_dir
    #                                     )
    plot_local_result_gantt(local_result,
                            # save_fig_dir=init_schedule_local_result_file_dir
                            # save_fig_dir=local_result_file_dir
                            save_fig_dir=drl_local_result_file_dir
                            )

