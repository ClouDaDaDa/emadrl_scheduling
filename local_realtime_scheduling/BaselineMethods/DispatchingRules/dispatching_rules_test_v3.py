# from memory_profiler import profile
# @profile
def func(content: str):
    print(content)

import time
import os
import pickle
import numpy as np
from configs import dfjspt_params
from local_realtime_scheduling.Environment.LocalSchedulingMultiAgentEnv_v3_4 import LocalSchedulingMultiAgentEnv
from local_realtime_scheduling.BaselineMethods.DispatchingRules.machine_agent_heuristics import machine_agent_heuristics
from local_realtime_scheduling.BaselineMethods.DispatchingRules.transbot_agent_heuristics import transbot_agent_heuristics
from System.SchedulingInstance import SchedulingInstance
from local_realtime_scheduling.Agents.generate_training_data import generate_reset_options_for_training


def test_rule_for_one_instance(
    test_instance_filename: str,
    num_repeat=20,
    detailed_log=True,
    do_render=False,
    do_plot_gantt=True,
    do_plot_trajectories=False,
    current_window=None,
):
    print(f"\nStart testing dispatching rule policies:")

    config = {
        "n_machines": dfjspt_params.n_machines,
        "n_transbots": dfjspt_params.n_transbots,
        "factory_instance_seed": dfjspt_params.factory_instance_seed,
        "enable_dynamic_agent_filtering": getattr(dfjspt_params, 'enable_dynamic_agent_filtering', False),
    }

    scheduling_env = LocalSchedulingMultiAgentEnv(config)
    # func("Env instance created.")
    # print(f"Processing file: {test_instance_filename}")

    num_episodes = num_repeat
    makespans = []
    running_time = []

    for episode in range(num_episodes):
        print(f"\nStarting episode {episode + 1}")
        begin_time = time.time()
        decision_count = 0
        env_reset_options = generate_reset_options_for_training(
            local_schedule_filename=test_instance_filename,
            for_training=False,
            # for_training=True,
        )

        observations, infos = scheduling_env.reset(options=env_reset_options)
        if do_render:
            scheduling_env.render()
        # print(f"decision_count = {decision_count}")
        decision_count += 1
        done = {'__all__': False}
        truncated = {'__all__': False}
        total_rewards = {}
        for agent in scheduling_env.agents:
            total_rewards[agent] = 0.0

        while (not done['__all__']) and (not truncated['__all__']):
            actions = {}
            for agent_id, obs in observations.items():
                # print(f"current agent = {agent_id}")

                if agent_id.startswith("machine"):
                    actions[agent_id] = machine_agent_heuristics(machine_obs=obs)

                elif agent_id.startswith("transbot"):
                    actions[agent_id] = transbot_agent_heuristics(transbot_obs=obs)

            observations, rewards, done, truncated, info = scheduling_env.step(actions)
            if do_render:
                scheduling_env.render()
            decision_count += 1

            for agent, reward in rewards.items():
                total_rewards[agent] += reward

        if do_render:
            scheduling_env.close()
        if do_plot_gantt:
            scheduling_env.plot_jobs_gantt(
                plot_end_time=scheduling_env.current_time_after_step,
                save_fig_dir=None,
            )
            scheduling_env.plot_machines_gantt(
                plot_end_time=scheduling_env.current_time_after_step,
                save_fig_dir=None,
            )
            scheduling_env.plot_transbots_gantt(
                plot_end_time=scheduling_env.current_time_after_step,
                save_fig_dir=None,
            )
        if do_plot_trajectories:
            scheduling_env.plot_transbot_trajectory(
                0,
                scheduling_env.local_result.time_window_start,
                scheduling_env.current_time_after_step,
            )
            scheduling_env.plot_transbot_trajectory(
                1,
                scheduling_env.local_result.time_window_start,
                scheduling_env.current_time_after_step,
            )


        if detailed_log:
            print(f"Current timestep is {scheduling_env.current_time_after_step}.")
            if scheduling_env.local_result.actual_local_makespan is not None:
                print(f"Actual makespan = {scheduling_env.local_result.actual_local_makespan}")
                print(
                    f"Actual delta makespan = {scheduling_env.local_result.actual_local_makespan - scheduling_env.local_result.time_window_start}")
                makespans.append(
                    scheduling_env.local_result.actual_local_makespan - scheduling_env.local_result.time_window_start)
            else:
                makespans.append(scheduling_env.current_time_after_step - scheduling_env.local_result.time_window_start)
                print(f"Truncated makespan = {scheduling_env.current_time_after_step}")
                print(
                    f"Truncated delta makespan = {scheduling_env.current_time_after_step - scheduling_env.local_result.time_window_start}")
            print(f"Estimated makespan = {scheduling_env.initial_estimated_makespan}")
            print(
                f"Estimated delta makespan = {scheduling_env.initial_estimated_makespan - scheduling_env.local_result.time_window_start}")
            print(f"Total reward for episode {episode + 1}: {total_rewards['machine0']}")

            end_time = time.time()
            print(f"Running time for episode {episode + 1} is {end_time - begin_time}")
            running_time.append(end_time - begin_time)

            func("Local Scheduling completed.")

            print(f"Min delta makespan up to now is {np.min(makespans)}.")
            print(f"Average delta makespan up to now is {np.average(makespans)}.")
        else:
            if scheduling_env.local_result.actual_local_makespan is not None:
                makespans.append(
                    scheduling_env.local_result.actual_local_makespan - scheduling_env.local_result.time_window_start)
            else:
                makespans.append(scheduling_env.current_time_after_step - scheduling_env.local_result.time_window_start)
            end_time = time.time()
            running_time.append(end_time - begin_time)

    print(f"\nMin makespan across {num_episodes} episodes is {np.min(makespans)}.")
    print(f"Average makespan across {num_episodes} episodes is {np.average(makespans)}.")
    print(f"Average running time across {num_episodes} episodes is {np.average(running_time)}.")

    return makespans


def test_rule_for_all_instances(
    num_repeat=20,
    detailed_log=True,
):
    import re
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_schedule_dir = os.path.dirname(os.path.dirname(current_dir)) + \
        "/InterfaceWithGlobal/local_schedules" + \
        f"/M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}W{dfjspt_params.time_window_size}/testing"

    # Get all .pkl files in the directory
    local_schedule_files = [file_name for file_name in os.listdir(local_schedule_dir) if file_name.endswith(".pkl")]

    # Sort files by the 'ops' number (from smallest to largest)
    sorted_files = sorted(local_schedule_files, key=lambda f: int(re.search(r'ops(\d+)\.pkl', f).group(1)), reverse=True)

    total_num_instance = len(sorted_files)

    for instance_id in range(total_num_instance):
        test_instance_filename = sorted_files[instance_id]

        match = re.match(r'local_schedule_J(\d+)I(\d+)_(\d+)_ops(\d+)\.pkl', test_instance_filename)
        if match:
            # n_jobs = int(match.group(1))
            current_scheduling_instance_id = int(match.group(2))
            # current_window = int(match.group(3))
            # n_ops_for_tw = int(match.group(4))

            if current_scheduling_instance_id >= 100:
                makespans = test_rule_for_one_instance(
                    test_instance_filename=test_instance_filename,
                    num_repeat=num_repeat,
                    detailed_log=detailed_log,
                )

def test_rule_for_one_global_instance(
    n_jobs,
    instance_id,
    num_repeat=1,
    detailed_log=True,
):
    import re
    current_dir = os.path.dirname(os.path.abspath(__file__))

    config = {
        "n_machines": dfjspt_params.n_machines,
        "n_transbots": dfjspt_params.n_transbots,
        "factory_instance_seed": dfjspt_params.factory_instance_seed,
    }

    local_schedule_dir = os.path.dirname(os.path.dirname(current_dir)) + \
        "/InterfaceWithGlobal/local_schedules" + \
        f"/M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}W{dfjspt_params.time_window_size}/testing"

    pattern = re.compile(f"local_schedule_J{n_jobs}I{instance_id}_.*\.pkl$")


    local_instances = sorted(
        [f for f in os.listdir(local_schedule_dir) if pattern.match(f)],
        key=lambda x: int(re.search(r'_(\d+)_ops', x).group(1))
    )
    num_windows = len(local_instances)

    num_episodes = num_repeat
    global_makespans = []
    running_time = []

    for episode in range(num_episodes):
        print(f"\nStarting episode {episode + 1}")
        begin_time = time.time()

        scheduling_env = LocalSchedulingMultiAgentEnv(config)

        # for local_schedule_file in local_instances:
        for current_window in range(num_windows):
            local_schedule_file = local_schedule_dir + '/' + local_instances[current_window]

            with open(local_schedule_file,
                      "rb") as file:
                local_schedule = pickle.load(file)
            print(f"Processing file: {local_instances[current_window]}")

            if current_window == 0:
                reset_options = {
                    "factory_instance": scheduling_env.factory_instance,
                    "scheduling_instance": SchedulingInstance(
                        seed=52 + instance_id,
                        n_jobs=n_jobs,
                        n_machines=scheduling_env.num_machines,
                    ),
                    "local_schedule": local_schedule,
                    "current_window": 0,
                    "instance_n_jobs": n_jobs,
                    "current_instance_id": instance_id,
                    "start_t_for_curr_time_window": 0,
                    "local_result_file": None,
                }

            else:

                instance_snapshot_dir = os.path.dirname(os.path.dirname(current_dir)) + \
                    "/Environment/instance_snapshots" + \
                    f"/M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}W{dfjspt_params.time_window_size}" \
                    + f"/snapshot_J{n_jobs}I{instance_id}" \
                    + f"_{current_window - 1}.pkl"
                with open(instance_snapshot_dir,
                          "rb") as file:
                    prev_instance_snapshot = pickle.load(file)

                reset_options = {
                    "factory_instance": prev_instance_snapshot["factory_instance"],
                    "scheduling_instance": prev_instance_snapshot["scheduling_instance"],
                    "local_schedule": local_schedule,
                    "current_window": current_window,
                    "instance_n_jobs": n_jobs,
                    "current_instance_id": instance_id,
                    "start_t_for_curr_time_window": prev_instance_snapshot["start_t_for_curr_time_window"],
                    "local_result_file": None,
                }

            observations, infos = scheduling_env.reset(options=reset_options)
            # scheduling_env.render()
            # print(f"remaining operations = {scheduling_env.remaining_operations}")
            done = {'__all__': False}
            truncated = {'__all__': False}
            total_rewards = {}
            for agent in scheduling_env.agents:
                total_rewards[agent] = 0.0

            while (not done['__all__']) and (not truncated['__all__']):
                actions = {}
                for agent_id, obs in observations.items():
                    # print(f"current agent = {agent_id}")
                    if agent_id.startswith("machine"):
                        actions[agent_id] = machine_agent_heuristics(machine_obs=obs)

                    elif agent_id.startswith("transbot"):
                        actions[agent_id] = transbot_agent_heuristics(transbot_obs=obs)

                observations, rewards, done, truncated, info = scheduling_env.step(actions)

                for agent, reward in rewards.items():
                    total_rewards[agent] += reward

            # the last local time window has finished:
            if current_window == num_windows - 1:
                scheduling_env._save_instance_snapshot(final=True)
                if detailed_log:
                    print(f"Current timestep is {scheduling_env.current_time_after_step}.")
                    if scheduling_env.local_result.actual_local_makespan is not None:
                        print(f"Actual makespan = {scheduling_env.local_result.actual_local_makespan}")
                        global_makespans.append(scheduling_env.local_result.actual_local_makespan)
                    else:
                        global_makespans.append(scheduling_env.current_time_after_step)
                        print(f"Truncated makespan = {scheduling_env.current_time_after_step}")
                    print(f"Estimated makespan = {scheduling_env.initial_estimated_makespan}")
                    print(f"Total reward for episode {episode + 1}: {total_rewards['machine0']}")

                    end_time = time.time()
                    print(f"Running time for episode {episode + 1} is {end_time - begin_time}")
                    running_time.append(end_time - begin_time)

                    func("Local Scheduling completed.")

                    print(f"Min delta makespan up to now is {np.min(global_makespans)}.")
                    print(f"Average delta makespan up to now is {np.average(global_makespans)}.")
                else:
                    if scheduling_env.local_result.actual_local_makespan is not None:
                        global_makespans.append(scheduling_env.local_result.actual_local_makespan)
                    else:
                        global_makespans.append(scheduling_env.current_time_after_step)
                    end_time = time.time()
                    running_time.append(end_time - begin_time)

                print(f"\nMin makespan across {num_episodes} episodes is {np.min(global_makespans)}.")
                print(f"Average makespan across {num_episodes} episodes is {np.average(global_makespans)}.")
                print(f"Average running time across {num_episodes} episodes is {np.average(running_time)}.")

    return global_makespans


def test_rule_for_all_global_instances(
    num_repeat=1,
    detailed_log=True,
):
    global_makespans_for_all_instances = {}
    for i in range(9, 11):
        n_jobs = i * 10
        for instance_id in range(100, 110):
            global_makespans_list = test_rule_for_one_global_instance(
                n_jobs=n_jobs,
                instance_id=instance_id,
                num_repeat=num_repeat,
                detailed_log=detailed_log,
            )
            print(
                f"Global makespans list for /M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}W{dfjspt_params.time_window_size}/J{n_jobs}I{instance_id} is: ")
            print(global_makespans_list)
            global_makespans_for_all_instances[f"J{n_jobs}I{instance_id}"] = global_makespans_list[0]

            print(global_makespans_for_all_instances)



# Example usage
if __name__ == "__main__":
    from local_realtime_scheduling.InterfaceWithGlobal.divide_global_schedule_to_local_from_pkl import LocalSchedule, \
        Local_Job_schedule

    # current_dir = os.path.dirname(os.path.abspath(__file__))

    test_instance_filename = 'local_schedule_J100I107_0_ops112.pkl'  # M10T10
    test_rule_for_one_instance(test_instance_filename=test_instance_filename)


