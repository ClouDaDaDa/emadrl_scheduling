import copy
import numpy as np
import time
import json
from global_scheduling.dfjspt_env import DfjsptMaEnv
from configs import dfjspt_params
from global_scheduling.dfjspt_rule.job_selection_rules import job_EST_action
from global_scheduling.dfjspt_rule.machine_selection_rules import machine_EET_action, transbot_EET_action


def rule1_mean_makespan(
    instance_id,
    train_or_eval_or_test=None,
):

    config = {
        "train_or_eval_or_test": train_or_eval_or_test,
    }
    env = DfjsptMaEnv(config)
    makespan_list = []
    for _ in range(100):
        observation, info = env.reset(options={
            "instance_id": instance_id,
        })
        # env.render()
        done = False
        count = 0
        stage = next(iter(info["agent0"].values()), None)
        total_reward = 0

        while not done:
            if stage == 0:
                legal_job_actions = copy.deepcopy(observation["agent0"]["action_mask"])
                real_job_attrs = copy.deepcopy(observation["agent0"]["observation"])
                EST_job_action = job_EST_action(legal_job_actions=legal_job_actions, real_job_attrs=real_job_attrs)
                observation, reward, terminated, truncated, info = env.step(EST_job_action)
                stage = next(iter(info["agent1"].values()), None)

            elif stage == 1:
                legal_machine_actions = copy.deepcopy(observation["agent1"]["action_mask"])
                real_machine_attrs = copy.deepcopy(observation["agent1"]["observation"])
                EET_machine_action = machine_EET_action(legal_machine_actions=legal_machine_actions,
                                                        real_machine_attrs=real_machine_attrs)
                observation, reward, terminated, truncated, info = env.step(EET_machine_action)
                stage = next(iter(info["agent2"].values()), None)

            else:
                legal_transbot_actions = copy.deepcopy(observation["agent2"]["action_mask"])
                real_transbot_attrs = copy.deepcopy(observation["agent2"]["observation"])
                EET_transbot_action = transbot_EET_action(
                    legal_transbot_actions=legal_transbot_actions,
                    real_transbot_attrs=real_transbot_attrs)
                observation, reward, terminated, truncated, info = env.step(EET_transbot_action)
                stage = next(iter(info["agent0"].values()), None)
                done = terminated["__all__"]
                count += 1
                total_reward += reward["agent0"]

        make_span = env.final_makespan
        makespan_list.append(make_span)
    mean_makespan = np.mean(makespan_list)
    return makespan_list, mean_makespan


def rule1_single_makespan(
    instance_id,
    train_or_eval_or_test=None,
):

    config = {
        "train_or_eval_or_test": train_or_eval_or_test,
    }
    env = DfjsptMaEnv(config)

    observation, info = env.reset(options={
        "instance_id": instance_id,
    })
    # env.render()
    done = False
    count = 0
    stage = next(iter(info["agent0"].values()), None)
    total_reward = 0

    while not done:
        if stage == 0:
            legal_job_actions = copy.deepcopy(observation["agent0"]["action_mask"])
            real_job_attrs = copy.deepcopy(observation["agent0"]["observation"])
            EST_job_action = job_EST_action(legal_job_actions=legal_job_actions, real_job_attrs=real_job_attrs)
            observation, reward, terminated, truncated, info = env.step(EST_job_action)
            stage = next(iter(info["agent1"].values()), None)

        elif stage == 1:
            legal_machine_actions = copy.deepcopy(observation["agent1"]["action_mask"])
            real_machine_attrs = copy.deepcopy(observation["agent1"]["observation"])
            EET_machine_action = machine_EET_action(legal_machine_actions=legal_machine_actions,
                                                    real_machine_attrs=real_machine_attrs)
            observation, reward, terminated, truncated, info = env.step(EET_machine_action)
            stage = next(iter(info["agent2"].values()), None)

        else:
            legal_transbot_actions = copy.deepcopy(observation["agent2"]["action_mask"])
            real_transbot_attrs = copy.deepcopy(observation["agent2"]["observation"])
            EET_transbot_action = transbot_EET_action(
                legal_transbot_actions=legal_transbot_actions,
                real_transbot_attrs=real_transbot_attrs
            )
            observation, reward, terminated, truncated, info = env.step(EET_transbot_action)
            stage = next(iter(info["agent0"].values()), None)
            done = terminated["__all__"]
            count += 1
            total_reward += reward["agent0"]

    global_schedule = env.global_schedule
    return global_schedule


if __name__ == '__main__':
    # for n_job_id in range(1, 11):
    # for n_job_id in [5, 10, 15, 20]:
    for n_job_id in [1, 2]:
    # for n_job_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]:
    # for n_job_id in [1,2,3,4,5, 6, 7,8, 9, 10]:
        dfjspt_params.max_n_jobs = n_job_id * 10
        dfjspt_params.n_jobs = dfjspt_params.max_n_jobs
        print(f"\nProcessing instances with {dfjspt_params.n_jobs} jobs: ")
        rule_makespan_results = []
        time_0 = time.time()
        for i in range(100, 103):
        # for i in range(100, 105):
        # for i in range(10):
        # for i in range(dfjspt_params.n_instances):
            global_schedule_i = rule1_single_makespan(
                instance_id=i,
                train_or_eval_or_test=None,
            )
            rule_makespan_results.append(global_schedule_i.makespan)
            print(f"Makespan of instance {i} = {global_schedule_i.makespan}.")
            # print(f"Global Schedule of instance {i} is:")
            # print(global_schedule_i)
        time_1 = time.time()
        total_time = time_1 - time_0
        print(f"Total time = {total_time}.")
        # print(f"Average running time per instance = {total_time / dfjspt_params.n_instances}")
        average = 0 if not rule_makespan_results else (sum(rule_makespan_results) / len(rule_makespan_results))
        print(f"Average makespan = {average}.")




