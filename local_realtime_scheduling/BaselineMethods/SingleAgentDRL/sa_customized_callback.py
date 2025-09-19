import gymnasium as gym
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import EPISODE_RETURN_MEAN, ENV_RUNNER_RESULTS
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.env_context import EnvContext
from typing import Optional, Union, Dict
from local_realtime_scheduling.Agents.generate_training_data import generate_reset_options_for_training
from local_realtime_scheduling.Environment.ExecutionResult import LocalResult


class SA_Callbacks(DefaultCallbacks):
    def __init__(self):
        super().__init__()
        self.best_performance = float('inf')
        self.best_performance_iteration = 0
        self.patience_counter = 0
        self.performance_history = []
        print("SA_Callbacks initialized for single-agent performance monitoring.")

    @override(DefaultCallbacks)
    def on_environment_created(
        self,
        *,
        env_runner,
        metrics_logger: Optional[MetricsLogger] = None,
        env: gym.Env,
        env_context: EnvContext,
        **kwargs,
    ) -> None:
        print("Single-agent Environment created!")

    # @override(DefaultCallbacks)
    # def on_episode_created(
    #     self,
    #     *,
    #     episode: Union[EpisodeV2],
    #     env_runner = None,
    #     metrics_logger: Optional[MetricsLogger] = None,
    #     env: Optional[gym.Env] = None,
    #     rl_module: Optional[RLModule] = None,
    #     env_index: int = 0,
    #     **kwargs,
    # ):
    #     print("Single-agent Episode created!")
    #     # No curriculum logic, just return None
    #     # return None
    #     reset_options = generate_reset_options_for_training(
    #         local_schedule_filename=None,
    #         for_training=True,
    #     )
    #
    #     return reset_options

    @override(DefaultCallbacks)
    def on_episode_start(
        self,
        *,
        episode: Union[EpisodeV2],
        env_runner = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int = 0,
        rl_module: Optional[RLModule] = None,
        **kwargs,
    ) -> None:
        print("Single-agent Episode start!")
        scheduling_env = env.envs[0].env.env
        if scheduling_env is not None:
            print(f"Start timestep is {getattr(scheduling_env, 'current_time_before_step', None)}.")



    @override(DefaultCallbacks)
    def on_episode_step(
        self,
        *,
        episode: Union[EpisodeV2],
        env_runner = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int = 0,
        rl_module: Optional[RLModule] = None,
        **kwargs,
    ) -> None:
        pass

    @override(DefaultCallbacks)
    def on_episode_end(
        self,
        *,
        episode: Union[EpisodeV2],
        env_runner = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int = 0,
        rl_module: Optional[RLModule] = None,
        **kwargs,
    ) -> None:
        print("Single-agent Episode end!")
        env = env.envs[0].env.env
        if env is not None:
            print(f"Current timestep is {getattr(env, 'current_time_after_step', None)}.")
        # Compute makespan metrics
        if env is not None and hasattr(env, 'local_result') and hasattr(env.local_result, 'actual_local_makespan'):
            delta_makespan = env.local_result.actual_local_makespan - env.local_result.time_window_start
            episode_status = "completed"
            print(f"Actual makespan for this episode is {env.local_result.actual_local_makespan}.")
            print(f"Actual delta makespan for this episode is {delta_makespan}.")
        else:
            delta_makespan = getattr(env, 'current_time_after_step', 0) - getattr(env.local_result, 'time_window_start', 0)
            episode_status = "truncated"
            print("This episode is truncated.")
            print(f"Truncated delta makespan for this episode is {delta_makespan}.")
        # Log metrics
        if metrics_logger is not None:
            metrics_logger.log_value(key="actual_delta_makespan_ema", value=delta_makespan, reduce="mean", ema_coeff=0.1)
            metrics_logger.log_value(key="actual_delta_makespan_recent", value=delta_makespan, reduce="mean", window=100)
            metrics_logger.log_value(key="actual_delta_makespan_min", value=delta_makespan, reduce="min", window=100)
            metrics_logger.log_value(key="actual_delta_makespan_max", value=delta_makespan, reduce="max", window=100)
            metrics_logger.log_value(key="actual_delta_makespan_lifetime_mean", value=delta_makespan, reduce="mean", window=None)
            if episode_status == "completed":
                metrics_logger.log_value(key="actual_delta_makespan_completed_ema", value=delta_makespan, reduce="mean", ema_coeff=0.1)
                metrics_logger.log_value(key="episodes_completed_count", value=1, reduce="sum", clear_on_reduce=False)
            else:
                metrics_logger.log_value(key="actual_delta_makespan_truncated_ema", value=delta_makespan, reduce="mean", ema_coeff=0.1)
                metrics_logger.log_value(key="episodes_truncated_count", value=1, reduce="sum", clear_on_reduce=False)
            metrics_logger.log_value(key="episodes_total_count", value=1, reduce="sum", clear_on_reduce=False)

    def on_train_result(
        self,
        *,
        algorithm,
        metrics_logger=None,
        result: dict,
        **kwargs,
    ) -> None:
        training_iteration = result.get('training_iteration', 0)
        env_runner_results = result.get(ENV_RUNNER_RESULTS, {})
        if 'actual_delta_makespan_ema' in env_runner_results:
            current_ema = env_runner_results['actual_delta_makespan_ema']
            if metrics_logger is not None:
                metrics_logger.log_value(
                    key="training_performance_ema_by_iteration",
                    value=current_ema,
                    reduce="mean",
                    ema_coeff=0.05
                ) 