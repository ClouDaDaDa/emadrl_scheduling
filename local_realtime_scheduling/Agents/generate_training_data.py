import copy
import os
# import random
import re
import numpy as np
# import pickle
from ray import cloudpickle
from System.SchedulingInstance import SchedulingInstance
from System.FactoryInstance import FactoryInstance
from configs import dfjspt_params

def select_local_schedule_randomly(local_schedule_dir, seed=None):

    # Get all .pkl files in the directory
    local_schedule_files = [file_name for file_name in os.listdir(local_schedule_dir) if file_name.endswith(".pkl")]

    # # Sort files by the 'ops' number (from smallest to largest)
    # sorted_files = sorted(local_schedule_files, key=lambda f: int(re.search(r'ops(\d+)\.pkl', f).group(1)), reverse=True)

    # randomly pick a file
    if local_schedule_files:
        if seed is not None:
            np.random.seed(seed)
        local_schedule_filename = np.random.choice(local_schedule_files)

        return local_schedule_filename
    else:
        raise ValueError(f"Cannot find any local_schedule file in {local_schedule_dir}!")



def generate_reset_options_for_training(
        local_schedule_filename=None,
        for_training=True,
):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if for_training:
        local_schedule_dir = os.path.dirname(current_dir) + \
            "/InterfaceWithGlobal/local_schedules" + \
            f"/M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}W{dfjspt_params.time_window_size}/training"
    else:
        local_schedule_dir = os.path.dirname(current_dir) + \
            "/InterfaceWithGlobal/local_schedules" + \
            f"/M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}W{dfjspt_params.time_window_size}/testing"

    if local_schedule_filename is None:
        # randomly pick a file
        local_schedule_filename = select_local_schedule_randomly(
            local_schedule_dir=local_schedule_dir,
            seed=None,
        )

    # print(f"{local_schedule_filename} is selected as this training instance.")
    match = re.match(r'local_schedule_J(\d+)I(\d+)_(\d+)_ops(\d+)\.pkl', local_schedule_filename)
    if match:
        print(f"Processing file: {local_schedule_filename}")
        n_jobs = int(match.group(1))
        current_scheduling_instance_id = int(match.group(2))
        current_window = int(match.group(3))
        n_ops_for_tw = int(match.group(4))
        # read relevant information of this local_schedule

        with open(os.path.join(local_schedule_dir, local_schedule_filename),
                  "rb") as file:
            local_schedule = cloudpickle.load(file)

        if for_training:
            if current_window == 0:
                time_deviation = 0
            else:
                time_deviation = np.random.randint(0, 99)
        else:
            time_deviation = 0

        start_t_for_curr_time_window = local_schedule.time_window_start + time_deviation

        # recover the corresponding scheduling_instance from above information
        scheduling_instance = SchedulingInstance(
            seed=52 + current_scheduling_instance_id,
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
                if job.job_id in local_schedule.jobs and len(local_schedule.jobs[job.job_id].operations) > 0:
                    # Find the first operation in the current window
                    first_op_in_window = None
                    for operation in local_schedule.jobs[job.job_id].operations.values():
                        if operation.is_current_window:
                            first_op_in_window = operation.operation_id
                            break
                    
                    if first_op_in_window is not None:
                        # Set current_processing_operation to the first operation in the current window
                        # This ensures consistency between local_schedule and scheduling_instance
                        job.current_processing_operation = first_op_in_window
                        job.n_p_ops_for_curr_tw = first_op_in_window
                    else:
                        # No operations in current window, job is completed
                        job.current_processing_operation = len(job.operations_matrix)
                        job.n_p_ops_for_curr_tw = 0

        env_reset_options = {
            "factory_instance": factory_instance,
            "scheduling_instance": scheduling_instance,
            "local_schedule": local_schedule,
            "current_window": current_window,
            "start_t_for_curr_time_window": start_t_for_curr_time_window,
            "local_result_file": None,
        }

    else:
        raise ValueError(f"Cannot match {local_schedule_filename} with local_schedule_JxxIxx_xx_opsxx!")

    return env_reset_options


# Example usage
if __name__ == "__main__":
    from local_realtime_scheduling.Environment.LocalSchedulingMultiAgentEnv_v3_4 import LocalSchedulingMultiAgentEnv
    from local_realtime_scheduling.InterfaceWithGlobal.divide_global_schedule_to_local_from_pkl import LocalSchedule, \
        Local_Job_schedule

    env_config = {
        "n_machines": dfjspt_params.n_machines,
        "n_transbots": dfjspt_params.n_transbots,
        "factory_instance_seed": dfjspt_params.factory_instance_seed,
        "render_mode": "human",
    }

    for repeat in range(100):

        example_env = LocalSchedulingMultiAgentEnv(env_config)

        reset_options = generate_reset_options_for_training(for_training=True)

        observations, infos = example_env.reset(options=reset_options)
        example_env.render()

        done = {'__all__': False}
        truncated = {'__all__': False}
        total_rewards = {}
        for agent in example_env.agents:
            total_rewards[agent] = 0.0

        while (not done['__all__']) and (not truncated['__all__']):
            actions = {}
            for agent_id, obs in observations.items():
                # print(f"current agent = {agent_id}")
                action_mask = obs['action_mask']
                valid_actions = [i for i, valid in enumerate(action_mask) if valid == 1]
                if valid_actions:
                    if len(valid_actions) > 1:
                        valid_actions.pop(-1)
                    actions[agent_id] = np.random.choice(valid_actions)
                else:
                    raise Exception(f"No valid actions for agent {agent_id}!")
                    # actions[agent_id] = 0  # Default to a no-op if no valid actions

            observations, rewards, done, truncated, info = example_env.step(actions)
            example_env.render()
            # print(f"remaining operations = {example_env.remaining_operations}")

            for agent, reward in rewards.items():
                total_rewards[agent] += reward

        example_env.close()

        print(f"Actual makespan = {example_env.current_time_after_step}")
        print(f"Estimated makespan = {example_env.initial_estimated_makespan}")
        print(f"Total reward : {total_rewards['machine0']}")

        print("Local Scheduling completed.")

