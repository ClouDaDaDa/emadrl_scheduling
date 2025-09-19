import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
import os
import re
import numpy as np
import argparse
import time
from typing import Dict, List, Optional, Type, Union
from ray.air.constants import TRAINING_ITERATION
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    EVALUATION_RESULTS,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from ray.rllib.utils.typing import ResultDict
from ray.tune import CLIReporter
from ray.rllib.algorithms import AlgorithmConfig

from local_realtime_scheduling.Environment.LocalSchedulingMultiAgentEnv_v3_4 import LocalSchedulingMultiAgentEnv
from local_realtime_scheduling.InterfaceWithGlobal.divide_global_schedule_to_local_from_pkl import LocalSchedule, Local_Job_schedule
from local_realtime_scheduling.Agents.args_parser import add_default_args
from local_realtime_scheduling.Agents.action_mask_module import ActionMaskingTorchRLModule
from local_realtime_scheduling.Agents.customized_callback import MyCallbacks

parser = add_default_args(
    default_iters=2000,
    default_reward=1000,
)


@ray.remote
class CurriculumTaskManager:
    def __init__(self,
                 task_id_min,
                 task_id_max,
                 task_range,
                 task_pool_length,
                 sorted_task_pool,
                 ):
        self.task_id_min = task_id_min
        self.task_id_max = task_id_max
        self.task_range = task_range
        self.task_pool_length = task_pool_length
        self.sorted_task_pool = sorted_task_pool
        self.current_task_level = 0
        self.arrive_max_level = False

    def increase_task_level(self):
        if not self.arrive_max_level:
            self.current_task_level += 1
            self.task_id_min = self.task_id_max
            self.task_id_max = min(
                self.task_id_max + self.task_range,
                self.task_pool_length)

        if self.task_id_max == self.task_pool_length:
            self.arrive_max_level = True
            self.task_id_min = max(0, self.task_id_max - 3000)

    def generate_task_filename(self):
        task_id = np.random.randint(self.task_id_min, self.task_id_max)
        task_filename = self.sorted_task_pool[task_id]

        return task_filename

    def get_current_level(self):
        return self.current_task_level

def train_with_tune_pipeline(
    base_config: "AlgorithmConfig",
    args: Optional[argparse.Namespace] = None,
    *,
    stop: Optional[Dict] = None,
    success_metric: Optional[Dict] = None,
    trainable: Optional[Type] = None,
    tune_callbacks: Optional[List] = None,
    keep_config: bool = False,
    scheduler=None,
    progress_reporter=None,
) -> Union[ResultDict, tune.result_grid.ResultGrid]:
    """Run the training pipeline with the given configuration."""
    if args is None:
        parser = add_default_args()
        args = parser.parse_args()

    if args.as_release_test:
        args.as_test = True

    log_dir = (os.path.dirname(os.path.abspath(__file__))
        + f'/ray_results/M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}W{dfjspt_params.time_window_size}')

    args.num_cpus = 16
    ray.init(
        num_cpus=args.num_cpus or None,
        local_mode=args.local_mode,
        ignore_reinit_error=True,
    )

    if stop is None:
        stop = {
            f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": args.stop_reward,
            TRAINING_ITERATION: args.stop_iters,
        }

    config = base_config

    if not keep_config:
        config.framework(args.framework)

        if args.env is not None and config.env is None:
            config.environment(args.env)

        if args.enable_new_api_stack:
            config.api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )

        if args.num_env_runners is not None:
            config.env_runners(num_env_runners=args.num_env_runners)

    if args.no_tune:
        assert not args.as_test and not args.as_release_test
        algo = config.build()
        for i in range(stop.get(TRAINING_ITERATION, args.stop_iters)):
            results = algo.train()
            if ENV_RUNNER_RESULTS in results:
                mean_return = results[ENV_RUNNER_RESULTS].get(
                    EPISODE_RETURN_MEAN, np.nan
                )
                print(f"iter={i} R={mean_return}", end="")
            if EVALUATION_RESULTS in results:
                Reval = results[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][
                    EPISODE_RETURN_MEAN
                ]
                print(f" R(eval)={Reval}", end="")
            print()
            for key, threshold in stop.items():
                val = results
                for k in key.split("/"):
                    try:
                        val = val[k]
                    except KeyError:
                        val = None
                        break
                if val is not None and not np.isnan(val) and val >= threshold:
                    print(f"Stop criterium ({key}={threshold}) fulfilled!")
                    ray.shutdown()
                    return results

        ray.shutdown()
        return results

    tune_callbacks = tune_callbacks or []

    if progress_reporter is None and args.num_agents > 0:
        progress_reporter = CLIReporter(
            metric_columns={
                **{
                    TRAINING_ITERATION: "iter",
                    "time_total_s": "total time (s)",
                    NUM_ENV_STEPS_SAMPLED_LIFETIME: "ts",
                    f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": "combined return",
                },
                **{
                    (
                        f"{ENV_RUNNER_RESULTS}/module_episode_returns_mean/" f"{pid}"
                    ): f"return {pid}"
                    for pid in config.policies
                },
            },
        )

    os.environ["RAY_AIR_NEW_OUTPUT"] = "0"

    start_time = time.time()
    results = tune.Tuner(
        trainable or config.algo_class,
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            stop=stop,
            verbose=args.verbose,
            callbacks=tune_callbacks,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=args.checkpoint_freq,
                checkpoint_at_end=args.checkpoint_at_end,
            ),
            progress_reporter=progress_reporter,
            name=log_dir,
        ),
        tune_config=tune.TuneConfig(
            num_samples=args.num_samples,
            max_concurrent_trials=args.max_concurrent_trials,
            scheduler=scheduler,
        ),
    ).fit()
    time_taken = time.time() - start_time

    ray.shutdown()

    return results

if __name__ == "__main__":
    from configs import dfjspt_params
    args = parser.parse_args()

    if args.algo != "PPO":
        raise ValueError("This example only supports PPO. Please use --algo=PPO.")

    current_dir = os.path.dirname(os.path.abspath(__file__))

    env_config = {
        "n_machines": dfjspt_params.n_machines,
        "n_transbots": dfjspt_params.n_transbots,
        "factory_instance_seed": dfjspt_params.factory_instance_seed,
        # Add dynamic agent filtering configuration with safer defaults for training
        "enable_dynamic_agent_filtering": getattr(dfjspt_params, 'enable_dynamic_agent_filtering', False),
        "no_obs_solution_type": getattr(dfjspt_params, 'no_obs_solution_type', "dummy_agent"),
    }

    example_env = LocalSchedulingMultiAgentEnv(env_config)

    if dfjspt_params.enable_curriculum:
        local_schedule_dir = os.path.dirname(current_dir) + \
             "/InterfaceWithGlobal/local_schedules" + \
             f"/M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}W{dfjspt_params.time_window_size}/training"
        # Get all .pkl files in the directory
        local_schedule_files = [file_name for file_name in os.listdir(local_schedule_dir) if file_name.endswith(".pkl")]

        if len(local_schedule_files) > 0:
            # Sort files by the 'ops' number (from smallest to largest)
            sorted_files = sorted(local_schedule_files, key=lambda f: int(re.search(r'ops(\d+)\.pkl', f).group(1)))
            curriculum_task_manager = CurriculumTaskManager.options(name="curriculum_task_manager").remote(
                task_id_min=0,
                task_id_max=min(dfjspt_params.task_id_range, len(sorted_files)),
                task_range=dfjspt_params.task_id_range,
                task_pool_length=len(sorted_files),
                sorted_task_pool=sorted_files
            )
        else:
            raise ValueError(f"Cannot find any local_schedule file in {local_schedule_dir}!")

    max_episode_length = 2 * dfjspt_params.episode_time_upper_bound
    train_batch_size = max(1, dfjspt_params.num_env_runners) * max_episode_length

    if dfjspt_params.use_lstm:
        model_config = {
            # "vf_share_layers": False,
            "lstm_cell_size": 64,
            "use_lstm": True,
            "max_seq_len": (example_env.num_machines + example_env.num_transbots) * 5,
        }
    else:
        model_config = {}

    base_config = (
        PPOConfig()
        .environment(
            env=LocalSchedulingMultiAgentEnv,
            env_config=env_config,
            disable_env_checking=True,
        )
        .env_runners(
            num_env_runners=dfjspt_params.num_env_runners,
            num_envs_per_env_runner=1,
            batch_mode="complete_episodes",
            rollout_fragment_length="auto",
            sample_timeout_s=6000,
            observation_filter="MeanStdFilter",
        )
        .training(
            train_batch_size_per_learner=train_batch_size,
            minibatch_size=max_episode_length,
            entropy_coeff=0.001,

            num_epochs=5,
            # lr=1e-5,
            lr=[
                [0, 5e-5],
                # [train_batch_size * 10, 5e-5],
                [train_batch_size * 50, 1e-5],
            ],
        )
        .learners(
            num_learners=dfjspt_params.num_learners,
            num_cpus_per_learner=1,
            num_gpus_per_learner=0,
        )
        .checkpointing(
            checkpoint_trainable_policies_only=True,
        )
        .rl_module(

            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={

                    "p_machine": RLModuleSpec(
                        module_class=ActionMaskingTorchRLModule,
                        observation_space=example_env.observation_spaces[example_env.machine_agents[0]],
                        action_space=example_env.action_spaces[example_env.machine_agents[0]],
                        model_config=model_config,
                    ),
                    "p_transbot": RLModuleSpec(
                        module_class=ActionMaskingTorchRLModule,
                        observation_space=example_env.observation_spaces[example_env.transbot_agents[0]],
                        action_space=example_env.action_spaces[example_env.transbot_agents[0]],
                        model_config=model_config,
                    ),
                },
            ),
        )
        .multi_agent(
            policies={"p_machine", "p_transbot"},
            policy_mapping_fn=lambda agent_id, *a, **kw: "p_machine" if agent_id.startswith("machine") else "p_transbot",
        )

        .callbacks(MyCallbacks)
    )

    args.no_tune = dfjspt_params.no_tune

    print(f"num_learners = {base_config.num_learners}")
    print(f"train_batch_size_per_learner = {base_config.train_batch_size_per_learner}")
    print(f"total_train_batch_size = {base_config.total_train_batch_size}")
    print(f"sample_timeout_s = {base_config.sample_timeout_s}")

    # Run the example (with Tune).
    train_with_tune_pipeline(base_config, args)

