import pickle
import os
import re
from typing import List


class Local_Job_schedule:
    def __init__(self, job_id):
        self.job_id = job_id
        self.operations = {}
        self.available_time = None
        self.estimated_finish_time = None

    def add_operation(self, operation):
        self.operations[operation.operation_id] = operation

    def __repr__(self):
        return f"Job(Job_ID={self.job_id}, Operations={self.operations})"


class LocalSchedule:
    def __init__(self):
        self.jobs = {}
        self.global_n_jobs = None
        self.local_makespan = None
        self.time_window_start = None
        self.time_window_end = None
        self.n_ops_in_tw = None

    def add_job(self, job):
        self.jobs[job.job_id] = job

    def get_available_operations(self, current_time, include_lookahead=True):
        """
        Get operations that can be started at the current time.
        
        Args:
            current_time: The current simulation time
            include_lookahead: Whether to include lookahead operations
            
        Returns:
            List of operations that can be started
        """
        available_ops = []
        
        for job_id, job in self.jobs.items():
            for op_id, operation in job.operations.items():
                # Check if operation can be started
                if hasattr(operation, 'is_current_window') and operation.is_current_window:
                    # Current window operations can always be considered
                    if operation.scheduled_start_processing_time <= current_time:
                        available_ops.append(operation)
                elif include_lookahead and hasattr(operation, 'is_lookahead') and operation.is_lookahead:
                    # Lookahead operations can be started early if:
                    # 1. Current time is close to their scheduled start time (e.g., within a threshold)
                    # 2. All prerequisite operations are completed
                    threshold = (self.time_window_end - self.time_window_start) * 0.1  # 10% of window size
                    if operation.scheduled_start_processing_time <= current_time + threshold:
                        available_ops.append(operation)
        
        return available_ops
    
    def get_idle_resources_next_tasks(self, current_time, idle_machines=None, idle_transbots=None):
        """
        Get next available tasks for idle resources from lookahead operations.
        
        Args:
            current_time: Current simulation time
            idle_machines: List of idle machine IDs
            idle_transbots: List of idle transbot IDs
            
        Returns:
            Dictionary mapping resource IDs to their next available tasks
        """
        next_tasks = {'machines': {}, 'transbots': {}}
        
        if idle_machines is None:
            idle_machines = []
        if idle_transbots is None:
            idle_transbots = []
        
        # Find lookahead operations that can be assigned to idle resources
        for job_id, job in self.jobs.items():
            for op_id, operation in job.operations.items():
                if hasattr(operation, 'is_lookahead') and operation.is_lookahead:
                    # Check if assigned machine is idle
                    if (hasattr(operation, 'assigned_machine') and 
                        operation.assigned_machine in idle_machines):
                        if operation.assigned_machine not in next_tasks['machines']:
                            next_tasks['machines'][operation.assigned_machine] = []
                        next_tasks['machines'][operation.assigned_machine].append(operation)
                    
                    # Check if assigned transbot is idle
                    if (hasattr(operation, 'assigned_transbot') and 
                        operation.assigned_transbot in idle_transbots):
                        if operation.assigned_transbot not in next_tasks['transbots']:
                            next_tasks['transbots'][operation.assigned_transbot] = []
                        next_tasks['transbots'][operation.assigned_transbot].append(operation)
        
        # Sort tasks by scheduled start time for each resource
        for machine_id in next_tasks['machines']:
            next_tasks['machines'][machine_id].sort(
                key=lambda x: x.scheduled_start_processing_time
            )
        for transbot_id in next_tasks['transbots']:
            next_tasks['transbots'][transbot_id].sort(
                key=lambda x: x.scheduled_start_transporting_time 
                if hasattr(x, 'scheduled_start_transporting_time') 
                else x.scheduled_start_processing_time
            )
        
        return next_tasks

    def __repr__(self):
        return f"LocalSchedule(Jobs={self.jobs})"


# Function to divide global schedule into local schedules based on time windows
def divide_schedule_into_time_windows(
    global_schedule,
    time_window_size,
    output_filepath,
    lookahead_windows=1  # Number of future windows to include as lookahead
) -> List[LocalSchedule]:
    max_completion_time = global_schedule.makespan
    global_n_jobs = len(global_schedule.jobs)
    local_schedules = []

    idx = 0
    for start_time in range(0, int(max_completion_time) + 1, time_window_size):
        end_time = min(start_time + time_window_size, max_completion_time)
        lookahead_end_time = min(end_time + lookahead_windows * time_window_size, max_completion_time)
        
        print(f"Processing window: [{start_time}, {end_time}] with lookahead to {lookahead_end_time}")
        local_schedule = LocalSchedule()
        local_schedule.time_window_start = start_time
        local_schedule.time_window_end = end_time
        local_schedule.local_makespan = end_time
        local_schedule.global_n_jobs = global_n_jobs
        local_schedule.n_ops_in_tw = 0

        for job in global_schedule.jobs:
            total_n_ops = len(job.operations)
            relevant_job = None
            max_op_id = 0
            
            for operation in job.operations:
                is_current_window = (operation.scheduled_start_processing_time < end_time) \
                                   and (operation.scheduled_finish_processing_time >= start_time)
                is_lookahead = (operation.scheduled_start_processing_time >= end_time) \
                              and (operation.scheduled_start_processing_time < lookahead_end_time)
                
                if is_current_window or is_lookahead:
                    # If the job is not in the local schedule, add it
                    if relevant_job is None:
                        relevant_job = Local_Job_schedule(job_id=job.job_id)
                        if is_current_window and operation.scheduled_start_processing_time >= start_time:
                            relevant_job.available_time = start_time
                        elif not is_current_window:
                            relevant_job.available_time = operation.scheduled_start_processing_time
                        else:
                            relevant_job.available_time = operation.scheduled_finish_processing_time

                    # Add the operation to the relevant job
                    relevant_job.add_operation(operation)
                    
                    # Mark operations as current window or lookahead
                    if is_current_window and operation.scheduled_start_processing_time >= start_time:
                        relevant_job.operations[operation.operation_id].is_current_window = True
                        relevant_job.operations[operation.operation_id].is_lookahead = False
                        local_schedule.n_ops_in_tw += 1
                    elif is_lookahead:
                        relevant_job.operations[operation.operation_id].is_current_window = False
                        relevant_job.operations[operation.operation_id].is_lookahead = True
                    else:
                        # Operations that start before current window but finish within it
                        relevant_job.operations[operation.operation_id].is_current_window = False
                        relevant_job.operations[operation.operation_id].is_lookahead = False
                    
                    max_op_id = max(max_op_id, operation.operation_id)
                    
                    if operation.scheduled_finish_processing_time > local_schedule.local_makespan:
                        local_schedule.local_makespan = operation.scheduled_finish_processing_time

            if relevant_job is not None and len(relevant_job.operations) > 0:
                # Calculate finish time based on current window operations only
                current_window_finish_times = [
                    op.scheduled_finish_processing_time
                    for _, op in relevant_job.operations.items()
                    if hasattr(op, 'is_current_window') and op.is_current_window
                ]
                
                if current_window_finish_times:
                    relevant_job.estimated_finish_time = max(current_window_finish_times)
                else:
                    # If no current window operations, use the earliest available time
                    relevant_job.estimated_finish_time = relevant_job.available_time
                    
                local_schedule.add_job(relevant_job)

        local_schedules.append(local_schedule)
        if local_schedule.n_ops_in_tw > 0:
            save_local_schedule(local_schedule, output_filepath, idx)
        idx += 1
    return local_schedules


# Function to save LocalSchedule objects as .pkl files
# def save_local_schedules(local_schedules: List[LocalSchedule], output_folder: str):
#     for idx, local_schedule in enumerate(local_schedules):
#         # filename = os.path.join(output_folder, f"window_{idx}.pkl")
#         filename = output_folder + f"_{idx}.pkl"
#         os.makedirs(os.path.dirname(filename), exist_ok=True)
#
#         with open(filename, "wb") as file:
#             pickle.dump(local_schedule, file)
#         print(f"Local schedule for time window {idx} saved to {filename}")

def save_local_schedule(local_schedule: LocalSchedule, output_folder: str, idx: int):
    # filename = os.path.join(output_folder, f"window_{idx}.pkl")
    filename = output_folder + f"_{idx}_ops{local_schedule.n_ops_in_tw}.pkl"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "wb") as file:
        pickle.dump(local_schedule, file)
    print(f"Local schedule for time window {idx} saved to {filename}")


def generate_local_schedules(for_training: bool):
    from configs import dfjspt_params

    global_schedule_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) \
                          + f"/global_scheduling/InterfaceWithLocal/global_schedules" \
                          + f"/M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}"

    # # Define the path to the GlobalSchedule file and the output folder for local schedules
    # global_schedule_filename = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) \
    #     + "/global_scheduling/InterfaceWithLocal/global_schedules/global_schedule_" \
    #     + f"J{dfjspt_params.n_jobs}M{dfjspt_params.n_machines}" \
    #     + f"T{dfjspt_params.n_transbots}I{dfjspt_params.current_scheduling_instance_id}.pkl"
    # output_folder = os.path.dirname(os.path.abspath(__file__)) + \
    #                 "/local_schedules/local_schedule_" + \
    #                 f"J{dfjspt_params.n_jobs}M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}I{dfjspt_params.current_scheduling_instance_id}" \
    #                 + f"_window{dfjspt_params.time_window_size}"
    output_folder = os.path.dirname(os.path.abspath(__file__)) + \
                    f"/local_schedules/M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}W{dfjspt_params.time_window_size}"

    if for_training:
        output_folder = output_folder + f"/training"
    else:
        output_folder = output_folder + f"/testing"

    for global_schedule_filename in os.listdir(global_schedule_dir):
        if global_schedule_filename.endswith(".pkl"):  # Only processes .pkl files
            # Use regular expressions to extract the numbers after J and I
            match = re.match(r'global_schedule_J(\d+)I(\d+)\.pkl', global_schedule_filename)
            if match:
                num_jobs = int(match.group(1))
                current_instance_id = int(match.group(2))

                if for_training:
                    if current_instance_id <= 10:
                        print(f"Processing file: {global_schedule_filename}")

                        instance_name = f"local_schedule_J{num_jobs}I{current_instance_id}"
                        output_filepath = os.path.join(output_folder, instance_name)

                        # Load the GlobalSchedule from the pkl file
                        with open(os.path.join(global_schedule_dir, global_schedule_filename), "rb") as file:
                            global_schedule = pickle.load(file)

                        # Divide the global schedule into local schedules based on time windows
                        local_schedules = divide_schedule_into_time_windows(
                            global_schedule,
                            dfjspt_params.time_window_size,
                            output_filepath,
                            lookahead_windows=getattr(dfjspt_params, 'lookahead_windows', 1),  # Default to 1 if not defined
                        )
                else:
                    # if current_instance_id >= 100:
                    if current_instance_id >= 100:
                        print(f"Processing file: {global_schedule_filename}")

                        instance_name = f"local_schedule_J{num_jobs}I{current_instance_id}"
                        output_filepath = os.path.join(output_folder, instance_name)

                        # Load the GlobalSchedule from the pkl file
                        with open(os.path.join(global_schedule_dir, global_schedule_filename), "rb") as file:
                            global_schedule = pickle.load(file)

                        # Divide the global schedule into local schedules based on time windows
                        local_schedules = divide_schedule_into_time_windows(
                            global_schedule,
                            dfjspt_params.time_window_size,
                            output_filepath,
                            lookahead_windows=getattr(dfjspt_params, 'lookahead_windows', 1),  # Default to 1 if not defined
                        )

# Main execution
if __name__ == "__main__":

    generate_local_schedules(
        # for_training=True,
        for_training=False,
    )


