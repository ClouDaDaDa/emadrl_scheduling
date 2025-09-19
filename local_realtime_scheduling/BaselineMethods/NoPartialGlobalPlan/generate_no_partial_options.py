import copy
# import os
# import random
# import re
import numpy as np
# import pickle
# from ray import cloudpickle
from System.SchedulingInstance import SchedulingInstance
from System.FactoryInstance import FactoryInstance
from configs import dfjspt_params


MaxWindowDict = {
    "M10T10": {    
        10: 2,
        20: 3,
        30: 4,
        40: 5,
        50: 6,
        60: 6,
        70: 7,
        80: 8,
        90: 8,
        100: 8,
        150: 8,
        200: 8
    },
    "M36T20": {    
        10: 3,
        20: 3,
        30: 3,
        40: 3,
        50: 4,
        60: 4,
        70: 5,
        80: 5,
        90: 6,
        100: 6,
        150: 8,
        200: 8
    }
}

def select_local_instance_randomly(seed=None, for_training=True):
    if seed is not None:
        np.random.seed(seed)

    m_t = f"M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}"
    
    if for_training:
        n_jobs = np.random.choice([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])     
        scheduling_instance_id = np.random.randint(0, 10)
        max_window = MaxWindowDict[m_t][n_jobs]
        current_window = np.random.randint(0, max_window)
    else:
        n_jobs = np.random.choice([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200])     
        scheduling_instance_id = np.random.randint(100, 105)
        max_window = MaxWindowDict[m_t][n_jobs]
        current_window = np.random.randint(0, max_window)
    
    return n_jobs, scheduling_instance_id, max_window, current_window


def generate_no_partial_reset_options(
        n_jobs=None, scheduling_instance_id=None, 
        max_window=None, current_window=None,
        for_training=True,
):  

    if n_jobs is None or scheduling_instance_id is None or max_window is None or current_window is None:
        n_jobs, scheduling_instance_id, max_window, current_window = select_local_instance_randomly(
            seed=None,
            for_training=for_training,
        )

    if for_training:
        if current_window == 0:
            time_deviation = 0
        else:
            time_deviation = np.random.randint(0, 99)
    else:
        time_deviation = 0

    start_t_for_curr_time_window = dfjspt_params.time_window_size * current_window + time_deviation

    # recover the corresponding scheduling_instance from above information
    scheduling_instance = SchedulingInstance(
        seed=52 + scheduling_instance_id,
        n_jobs=n_jobs,
        n_machines=dfjspt_params.n_machines,
    )

    factory_instance = FactoryInstance(
        seed=dfjspt_params.factory_instance_seed,
        n_machines=dfjspt_params.n_machines,
        n_transbots=dfjspt_params.n_transbots,
    )

    if current_window == 0:
        pass
    else:
        # randomly initialize machines and transbots
        job_locations = ['warehouse']
        for machine in factory_instance.machines:
            job_locations.append(f'machine_{machine.machine_id}')
            machine.cumulative_total_time = start_t_for_curr_time_window
            machine.dummy_total_time = start_t_for_curr_time_window
            machine.cumulative_work_time = np.random.randint(0, 0.8 * start_t_for_curr_time_window)
            machine.dummy_work_time = np.random.randint(0, 500)
            machine.reliability = 1.0 - machine.degradation_model.degradation_function(current_life=machine.dummy_work_time)
            machine.cumulative_tasks = np.random.randint(0, np.max(scheduling_instance.n_operations_for_jobs))

        transbot_locations = copy.deepcopy(job_locations)
        transbot_locations.append("charging_0")
        transbot_locations.append("charging_1")
        for transbot in factory_instance.agv:
            transbot.cumulative_total_time = start_t_for_curr_time_window
            transbot.dummy_total_time = start_t_for_curr_time_window
            transbot.prev_loaded_finish_time = start_t_for_curr_time_window
            transbot.current_location = factory_instance.factory_graph.pickup_dropoff_points[
                np.random.choice(transbot_locations)]
            transbot.trajectory_hist[transbot.cumulative_total_time] = transbot.current_location
            transbot.battery.soc = np.random.uniform(0.31, 1)
            transbot.t_since_prev_r = 0.0
            transbot.cumulative_work_time = np.random.randint(0, start_t_for_curr_time_window)
            transbot.dummy_work_time = np.random.randint(0, 500)
            transbot.cumulative_tasks = np.random.randint(0, np.max(scheduling_instance.n_operations_for_jobs))

        # randomly initialize jobs
        for job in scheduling_instance.jobs:
            job.current_location = np.random.choice(job_locations)
            first_op_in_window = current_window * (job.operations_matrix.shape[0] // max_window)
            job.current_processing_operation = first_op_in_window
            job.n_p_ops_for_curr_tw = first_op_in_window

    # Convert to NoPartialGlobalPlan format
    no_partial_options = {
        "factory_instance": factory_instance,
        "scheduling_instance": scheduling_instance,
        "instance_n_jobs": n_jobs,
        "current_instance_id": scheduling_instance_id,
        "max_window": max_window,
        "current_window": current_window,
        "start_t_for_curr_window": start_t_for_curr_time_window
    }

    return no_partial_options