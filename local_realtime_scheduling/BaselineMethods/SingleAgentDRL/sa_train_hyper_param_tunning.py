import ray
from ray import tune, air
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
import os
import numpy as np
import argparse
import time
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

from local_realtime_scheduling.BaselineMethods.SingleAgentDRL.LocalSchedulingSingleAgentEnv import LocalSchedulingSingleAgentEnv
from local_realtime_scheduling.BaselineMethods.SingleAgentDRL.sa_action_mask_module import SAActionMaskingTorchRLModule
from local_realtime_scheduling.BaselineMethods.SingleAgentDRL.sa_customized_callback import SA_Callbacks
from local_realtime_scheduling.InterfaceWithGlobal.divide_global_schedule_to_local_from_pkl import LocalSchedule, \
    Local_Job_schedule
from configs import dfjspt_params

def add_sa_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop_iters", type=int, default=2000)
    parser.add_argument("--stop_reward", type=float, default=1000)
    parser.add_argument("--num_cpus", type=int, default=8)
    parser.add_argument("--local_mode", action="store_true")
    parser.add_argument("--framework", type=str, default="torch")
    parser.add_argument("--no_tune", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)
    parser.add_argument("--checkpoint_freq", type=int, default=10)
    parser.add_argument("--checkpoint_at_end", action="store_true")
    return parser

def train_with_tune_pipeline(
    base_config: "AlgorithmConfig",
    args: argparse.Namespace,
    stop: dict = None,
    trainable = None,
    tune_callbacks = None,
    scheduler=None,
    progress_reporter=None,
) -> ResultDict:
    log_dir = (os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        + f'/Agents/ray_results/SA_M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}W{dfjspt_params.time_window_size}')
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
    args.no_tune = dfjspt_params.no_tune
    if args.no_tune:
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
    if progress_reporter is None:
        progress_reporter = CLIReporter(
            metric_columns={
                TRAINING_ITERATION: "iter",
                "time_total_s": "total time (s)",
                NUM_ENV_STEPS_SAMPLED_LIFETIME: "ts",
                f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": "return",
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
            num_samples=1,
            max_concurrent_trials=1,
            scheduler=scheduler,
        ),
    ).fit()
    time_taken = time.time() - start_time
    ray.shutdown()
    return results

if __name__ == "__main__":
    parser = add_sa_args()
    args = parser.parse_args()
    env_config = {
        "n_machines": dfjspt_params.n_machines,
        "n_transbots": dfjspt_params.n_transbots,
        "factory_instance_seed": dfjspt_params.factory_instance_seed,
    }
    example_env = LocalSchedulingSingleAgentEnv(env_config)
    max_episode_length = 2 * dfjspt_params.episode_time_upper_bound
    train_batch_size = max(1, dfjspt_params.num_env_runners) * max_episode_length
    if dfjspt_params.use_lstm:
        model_config = {
            "lstm_cell_size": 64,
            "use_lstm": True,
            "max_seq_len": (example_env.num_machines + example_env.num_transbots) * 5,
        }
    else:
        model_config = {}
    base_config = (
        PPOConfig()
        .environment(
            env=LocalSchedulingSingleAgentEnv,
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
            lr=[
                [0, 5e-5],
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
            rl_module_spec=RLModuleSpec(
                module_class=SAActionMaskingTorchRLModule,
                observation_space=example_env.observation_space,
                action_space=example_env.action_space,
                model_config=model_config,
            ),
        )
        .callbacks(SA_Callbacks)
    )
    print(f"num_learners = {base_config.num_learners}")
    print(f"train_batch_size_per_learner = {base_config.train_batch_size_per_learner}")
    print(f"total_train_batch_size = {base_config.total_train_batch_size}")
    print(f"sample_timeout_s = {base_config.sample_timeout_s}")
    train_with_tune_pipeline(base_config, args) 