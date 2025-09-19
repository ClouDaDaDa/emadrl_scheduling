import os
import pickle
import numpy as np
from configs import dfjspt_params
from local_realtime_scheduling.BaselineMethods.InitialSchedule.InitialScheduleEnv_v3_3 import InitialScheduleEnv
from local_realtime_scheduling.Agents.generate_training_data import generate_reset_options_for_training
from local_realtime_scheduling.InterfaceWithGlobal.divide_global_schedule_to_local_from_pkl import LocalSchedule, \
    Local_Job_schedule

def run_one_local_instance(
        local_instance_file, num_episodes=10,
        # do_render=False,
        do_render=True,
        do_plot_gantt=True,
        do_plot_trajectories=False,
        # do_plot_gantt=False,
):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    local_schedule_file = os.path.dirname(os.path.dirname(current_dir)) + \
                          "/InterfaceWithGlobal/local_schedules" + \
                          f"/M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}W{dfjspt_params.time_window_size}/testing" \
                          + '/' + local_instance_file
    with open(local_schedule_file,
              "rb") as file:
        local_schedule = pickle.load(file)
    # print(vars(local_schedule))
    print(f"Starting instance {local_instance_file}:\n")

    config = {
        "n_machines": dfjspt_params.n_machines,
        "n_transbots": dfjspt_params.n_transbots,
        "factory_instance_seed": dfjspt_params.factory_instance_seed,
        "render_mode": "human",
    }
    scheduling_env = InitialScheduleEnv(config)

    makespans = []

    for episode_id in range(num_episodes):
        makespans = run_one_episode(episode_id=episode_id,
                                    scheduling_env=scheduling_env,
                                    local_schedule=local_schedule,
                                    makespans=makespans,
                                    do_render=do_render,
                                    do_plot_gantt=do_plot_gantt,
                                    do_plot_trajectories=do_plot_trajectories,
                                    local_instance_file=local_instance_file,
                                    )

    print(f"\nMin makespan across {num_episodes} episodes is {np.min(makespans)}.")
    print(f"Average makespan across {num_episodes} episodes is {np.average(makespans)}.")


def run_one_episode(
        episode_id, scheduling_env, local_schedule, makespans,
        # do_render=False,
        do_render=True,
        do_plot_gantt=True,
        # do_plot_gantt=False,
        do_plot_trajectories=False,
        local_instance_file=None,
):
    print(f"\nStarting episode {episode_id}:")
    decision_count = 0

    env_reset_options = generate_reset_options_for_training(
        local_schedule_filename=local_instance_file,
        for_training=False,
        # for_training=True,
    )

    observations, infos = scheduling_env.reset(options=env_reset_options)
    if do_render:
        scheduling_env.render()
    # print(f"decision_count = {decision_count}")
    decision_count += 1
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
            action_mask = obs['action_mask']
            valid_actions = [i for i, valid in enumerate(action_mask) if valid == 1]
            if valid_actions:
                if len(valid_actions) > 1:
                    valid_actions.pop(-1)
                actions[agent_id] = np.random.choice(valid_actions)
            else:
                raise Exception(f"No valid actions for agent {agent_id}!")
                # actions[agent_id] = 0  # Default to a no-op if no valid actions

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
    print(f"Total reward for episode {episode_id}: {total_rewards['machine0']}")

    print(f"Min makespan up to now is {np.min(makespans)}.")
    print(f"Average makespan up to now is {np.average(makespans)}.")

    return makespans


# Example usage
if __name__ == "__main__":
    instance_n_jobs = 5
    scheduling_instance_id = 100
    current_window =0
    num_ops = 31

    num_instances = 1
    for local_instance in range(num_instances):
        file_name = f"local_schedule_J{instance_n_jobs}I{scheduling_instance_id}_{current_window}_ops{num_ops}.pkl"
        run_one_local_instance(
            local_instance_file=file_name,
            # do_render=False,
            do_render=True,
            do_plot_gantt=True,
            # do_plot_gantt=False,
        )