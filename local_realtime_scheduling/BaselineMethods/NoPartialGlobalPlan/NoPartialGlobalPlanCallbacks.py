# import copy
# import os
# import pickle
# import pathlib
# import random
# import numpy as np
# from functools import partial
# from ray.rllib.utils.numpy import convert_to_numpy
# from ray.rllib.utils.test_utils import check
# from ray import cloudpickle
import gymnasium as gym
import ray
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.evaluation.episode_v2 import EpisodeV2
# from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy import Policy
# from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils import check
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import EPISODE_RETURN_MEAN, ENV_RUNNER_RESULTS
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import AgentID, EnvType, EpisodeType, PolicyID
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, Union
if TYPE_CHECKING:
    # from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.env.env_runner import EnvRunner
    # from ray.rllib.env.env_runner_group import EnvRunnerGroup

# from System.SchedulingInstance import SchedulingInstance
# from System.FactoryInstance import FactoryInstance
from configs import dfjspt_params
from local_realtime_scheduling.BaselineMethods.NoPartialGlobalPlan.generate_no_partial_options import generate_no_partial_reset_options
# from local_realtime_scheduling.InterfaceWithGlobal.divide_global_schedule_to_local_from_pkl import LocalSchedule, Local_Job_schedule


class MyCallbacks(DefaultCallbacks):
    
    def __init__(self):
        """Initialize callbacks."""
        super().__init__()

        self.best_performance = float('inf')
        self.best_performance_iteration = 0
        self.patience_counter = 0
        self.performance_history = []
        
        print("MyCallbacks initialized with performance monitoring.")

    # @override(DefaultCallbacks)
    # def on_algorithm_init(
    #         self,
    #         *,
    #         algorithm: "Algorithm",
    #         metrics_logger: Optional[MetricsLogger] = None,
    #         **kwargs,
    # ) -> None:
    #     multi_rl_module_component_tree = "learner_group/learner/rl_module"
    #     checkpoint_dir = '/Users/cloudadastudio/Downloads/multi_embodied_agent_realtime_scheduling/local_realtime_scheduling/Agents/ray_results/M10T10W300/PPO_LocalSchedulingMultiAgentEnv_90ef6_00000_0_2025-08-31_12-47-15/checkpoint_000008'
    #
    #     module_p_machine = algorithm.get_module("p_machine")
    #     weight_before = convert_to_numpy(next(iter(module_p_machine.parameters())))
    #
    #     algorithm.restore_from_path(
    #         # path=pathlib.Path(checkpoint_dir) / multi_rl_module_component_tree,
    #         path=checkpoint_dir,
    #         # # Algo is multi-agent (has "p0" and "p1" subdirs).
    #         # component=multi_rl_module_component_tree + "/p1",
    #     )
    #     module_p_machine = algorithm.get_module("p_machine")
    #
    #     # Make sure weights were restored (changed).
    #     weight_after = convert_to_numpy(next(iter(module_p_machine.parameters())))
    #     check(weight_before, weight_after, false=True)
    #
    #     print(f"Algorithm has restored from checkpoint {checkpoint_dir}.")


    @override(DefaultCallbacks)
    def on_environment_created(
        self,
        *,
        env_runner: "EnvRunner",
        metrics_logger: Optional[MetricsLogger] = None,
        env: gym.Env,
        env_context: EnvContext,
        **kwargs,
    ) -> None:
        print("Environment created!")

    @override(DefaultCallbacks)
    def on_episode_created(
        self,
        *,
        # TODO (sven): Deprecate Episode/EpisodeV2 with new API stack.
        episode: Union[EpisodeType, EpisodeV2],
        # TODO (sven): Deprecate this arg new API stack (in favor of `env_runner`).
        # worker: Optional["EnvRunner"] = None,
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        # TODO (sven): Deprecate this arg new API stack (in favor of `env`).
        # base_env: Optional[BaseEnv] = None,
        env: Optional[gym.Env] = None,
        # TODO (sven): Deprecate this arg new API stack (in favor of `rl_module`).
        # policies: Optional[Dict[PolicyID, Policy]] = None,
        rl_module: Optional[RLModule] = None,
        env_index: int,
        **kwargs,
    ):
        print("Episode created!")

        reset_options = generate_no_partial_reset_options(
            for_training=True,
        )

        return reset_options

    @override(DefaultCallbacks)
    def on_episode_start(
        self,
        *,
        episode: Union[EpisodeType, EpisodeV2],
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        # TODO (sven): Deprecate these args.
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        **kwargs,
    ) -> None:
        print("Episode start!")
        print(f"Start timestep is {env.current_time_before_step}.")

    @override(DefaultCallbacks)
    def on_episode_step(
        self,
        *,
        episode: Union[EpisodeType, EpisodeV2],
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        # TODO (sven): Deprecate these args.
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        **kwargs,
    ) -> None:
        pass
        # print("Episode step!")
        # if env.all_agents_have_made_decisions:
        #     print(f"Current timestep is {env.current_time_after_step}.")
        #     print(f"Remaining operations is {env.remaining_operations}.")

    @override(DefaultCallbacks)
    def on_episode_end(
        self,
        *,
        episode: Union[EpisodeType, EpisodeV2],
        env_runner: Optional["EnvRunner"] = None,
        metrics_logger: Optional[MetricsLogger] = None,
        env: Optional[gym.Env] = None,
        env_index: int,
        rl_module: Optional[RLModule] = None,
        # TODO (sven): Deprecate these args.
        worker: Optional["EnvRunner"] = None,
        base_env: Optional[BaseEnv] = None,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        **kwargs,
    ) -> None:
        print("Episode end!")
        print(f"Current timestep is {env.current_time_after_step}.")
        
        if env.local_result.actual_local_makespan is not None:
            delta_makespan = env.local_result.actual_local_makespan - env.local_result.time_window_start
            episode_status = "completed"
            print(f"Actual makespan for this episode is {env.local_result.actual_local_makespan}.")
            print(f"Actual delta makespan for this episode is {delta_makespan}.")
        else:
            delta_makespan = env.current_time_after_step - env.local_result.time_window_start
            episode_status = "truncated"
            print("This episode is truncated.")
            print(f"Truncated delta makespan for this episode is {delta_makespan}.")

        metrics_logger.log_value(
            key="actual_delta_makespan_ema", 
            value=delta_makespan,
            reduce="mean", 
            ema_coeff=0.1
        )
        
        metrics_logger.log_value(
            key="actual_delta_makespan_recent", 
            value=delta_makespan,
            reduce="mean", 
            window=100
        )
        
        metrics_logger.log_value(
            key="actual_delta_makespan_min", 
            value=delta_makespan,
            reduce="min", 
            window=100
        )
        
        metrics_logger.log_value(
            key="actual_delta_makespan_max", 
            value=delta_makespan,
            reduce="max", 
            window=100
        )
        
        metrics_logger.log_value(
            key="actual_delta_makespan_lifetime_mean", 
            value=delta_makespan,
            reduce="mean", 
            window=None  
        )
        
        if episode_status == "completed":
            metrics_logger.log_value(
                key="actual_delta_makespan_completed_ema", 
                value=delta_makespan,
                reduce="mean", 
                ema_coeff=0.1
            )
            metrics_logger.log_value(
                key="episodes_completed_count", 
                value=1,
                reduce="sum", 
                clear_on_reduce=False  
            )
        else:
            metrics_logger.log_value(
                key="actual_delta_makespan_truncated_ema", 
                value=delta_makespan,
                reduce="mean", 
                ema_coeff=0.1
            )
            metrics_logger.log_value(
                key="episodes_truncated_count", 
                value=1,
                reduce="sum", 
                clear_on_reduce=False  
            )
        
        metrics_logger.log_value(
            key="episodes_total_count", 
            value=1,
            reduce="sum", 
            clear_on_reduce=False  
        )

    def on_train_result(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger=None,
        result: dict,
        **kwargs,
    ) -> None:

        # # Check if the policy network was updated in the current training iteration
        # if "policy_updated" in result:
        #     policy_updated = result["policy_updated"]
        #     print(f"Policy network updated: {policy_updated}")


        training_iteration = result.get('training_iteration', 0)
        
        env_runner_results = result.get(ENV_RUNNER_RESULTS, {})
        
        if 'actual_delta_makespan_ema' in env_runner_results:
            current_ema = env_runner_results['actual_delta_makespan_ema']
            metrics_logger.log_value(
                key="training_performance_ema_by_iteration",
                value=current_ema,
                reduce="mean",
                ema_coeff=0.05  
            )

    def _log_performance_summary(self, training_iteration: int, env_runner_results: dict):
        print(f"\n=== Performance Summary at Training Iteration {training_iteration} ===")
        
        metrics_to_report = [
            ("EMA (Real-time trend)", "actual_delta_makespan_ema"),
            ("Recent 100 episodes", "actual_delta_makespan_recent"),
            ("Best in recent 100", "actual_delta_makespan_min"),
            ("Worst in recent 100", "actual_delta_makespan_max"),
            ("Lifetime average", "actual_delta_makespan_lifetime_mean"),
            ("Completed episodes EMA", "actual_delta_makespan_completed_ema"),
            ("Truncated episodes EMA", "actual_delta_makespan_truncated_ema"),
        ]
        
        for description, key in metrics_to_report:
            value = env_runner_results.get(key)
            if value is not None:
                print(f"  {description:<25}: {value:.4f}")
        
        stats_to_report = [
            ("Total episodes", "episodes_total_count"),
            ("Completed episodes", "episodes_completed_count"),
            ("Truncated episodes", "episodes_truncated_count"),
        ]
        
        for description, key in stats_to_report:
            value = env_runner_results.get(key)
            if value is not None:
                print(f"  {description:<25}: {int(value)}")
        
        completed = env_runner_results.get("episodes_completed_count", 0)
        total = env_runner_results.get("episodes_total_count", 1)
        completion_rate = completed / total if total > 0 else 0
        print(f"  {'Completion rate':<25}: {completion_rate:.4f}")
        
        recent_mean = env_runner_results.get("actual_delta_makespan_recent")
        recent_min = env_runner_results.get("actual_delta_makespan_min")
        if recent_mean is not None and recent_min is not None and recent_mean > 0:
            improvement_ratio = recent_min / recent_mean
            print(f"  {'Improvement ratio':<25}: {improvement_ratio:.4f}")
        
        print("=" * 65)


