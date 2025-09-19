from memory_profiler import profile
@profile
def func(text):
    print(text)

func("before import:")

import os
import pickle
from configs import dfjspt_params
import numpy as np

func("after import:")


# Function to convert loaded GlobalSchedule back to Numpy data
def convert_class_to_numpy(global_schedule):
    # Extract makespan
    makespan = global_schedule.makespan

    # Determine the number of jobs and maximum number of operations
    n_jobs = len(global_schedule.jobs)
    max_n_operations = max(len(job.operations) for job in global_schedule.jobs)

    # Initialize Numpy arrays
    job_arrival_time = np.zeros(n_jobs)
    job_due_date = np.zeros(n_jobs)
    result_start_time_for_jobs = np.zeros((n_jobs, max_n_operations, 2))
    result_finish_time_for_jobs = np.zeros((n_jobs, max_n_operations, 2))

    # Initialize routes
    machine_routes = {}
    transbot_routes = {}

    # Populate Numpy arrays and routes
    for job in global_schedule.jobs:
        job_id = job.job_id
        job_arrival_time[job_id] = job.arrival_time
        job_due_date[job_id] = job.due_date

        for operation in job.operations:
            operation_id = operation.operation_id

            if operation.assigned_transbot is not None:
                # Transport operation details
                if operation.assigned_transbot not in transbot_routes:
                    transbot_routes[operation.assigned_transbot] = []
                transbot_routes[operation.assigned_transbot].append([
                    operation.job_id,
                    operation.operation_id,
                    # operation.transbot_source,
                    # operation.job_location,
                    # operation.destination,
                ])

                # Transport times
                result_start_time_for_jobs[job_id, operation_id, 0] = operation.scheduled_start_transporting_time
                result_finish_time_for_jobs[job_id, operation_id, 0] = operation.scheduled_finish_transporting_time

            # Processing operation details
            if operation.assigned_machine not in machine_routes:
                machine_routes[operation.assigned_machine] = []
            machine_routes[operation.assigned_machine].append([
                operation.job_id,
                operation.operation_id,
            ])

            # Processing times
            result_start_time_for_jobs[job_id, operation_id, 1] = operation.scheduled_start_processing_time
            result_finish_time_for_jobs[job_id, operation_id, 1] = operation.scheduled_finish_processing_time

    return (
        makespan,
        job_arrival_time,
        job_due_date,
        result_start_time_for_jobs,
        result_finish_time_for_jobs,
        machine_routes,
        transbot_routes,
    )


if __name__ == "__main__":
    n_jobs = 100
    n_machines = 36
    n_transbots = 20
    instance_id = 0

    global_schedule_file = os.path.dirname(os.path.abspath(__file__)) \
                           + "/global_schedules" \
                           + f"/M{n_machines}T{n_transbots}" \
                           + f"/global_schedule_J{n_jobs}I{instance_id}.pkl"

    func("before load:")

    with open(global_schedule_file, "rb") as file:
        loaded_schedule = pickle.load(file)
    # print(loaded_schedule)

    func("after load:")

    makespan, job_arrival_time, job_due_date, result_start_time_for_jobs, result_finish_time_for_jobs, machine_routes, transbot_routes = convert_class_to_numpy(loaded_schedule)
    print(f"makespan: {makespan}")
    print(f"job_arrival_time: {job_arrival_time}")
    print(f"job_due_date: {job_due_date}")
    print(f"result_start_time_for_jobs[0]: {result_start_time_for_jobs[0]}")
    print(f"result_finish_time_for_jobs[0]: {result_finish_time_for_jobs[0]}")
    print(f"machine_routes[0]: {machine_routes[0]}")
    print(f"transbot_routes[0]: {transbot_routes[0]}")
    func("after convert to numpy:")




