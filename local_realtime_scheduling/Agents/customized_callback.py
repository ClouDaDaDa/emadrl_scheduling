
import gymnasium as gym
import ray
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils import check
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import EPISODE_RETURN_MEAN, ENV_RUNNER_RESULTS
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import AgentID, EnvType, EpisodeType, PolicyID
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, Union
if TYPE_CHECKING:
    from ray.rllib.env.env_runner import EnvRunner

from configs import dfjspt_params
from local_realtime_scheduling.Agents.generate_training_data import generate_reset_options_for_training


class MyCallbacks(DefaultCallbacks):
    
    def __init__(self):
        """Initialize callbacks."""
        super().__init__()

        self.best_performance = float('inf')
        self.best_performance_iteration = 0
        self.patience_counter = 0
        self.performance_history = []
        
        print("MyCallbacks initialized with performance monitoring.")

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
        """Callback run when a new environment object has been created.

        Note: This only applies to the new API stack. The env used is usually a
        gym.Env (or more specifically a gym.vector.Env).

        Args:
            env_runner: Reference to the current EnvRunner instance.
            metrics_logger: The MetricsLogger object inside the `env_runner`. Can be
                used to log custom metrics after environment creation.
            env: The environment object that has been created on `env_runner`. This is
                usually a gym.Env (or a gym.vector.Env) object.
            env_context: The `EnvContext` object that has been passed to the
                `gym.make()` call as kwargs (and to the gym.Env as `config`). It should
                have all the config key/value pairs in it as well as the
                EnvContext-typical properties: `worker_index`, `num_workers`, and
                `remote`.
            kwargs: Forward compatibility placeholder.
        """
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
        """Callback run when a new episode is created (but has not started yet!).

        This method gets called after a new Episode(V2) (old stack) or
        MultiAgentEpisode instance has been created.
        This happens before the respective sub-environment's (usually a gym.Env)
        `reset()` is called by RLlib.

        Note, at the moment this callback does not get called in the new API stack
        and single-agent mode.

        1) Episode(V2)/MultiAgentEpisode created: This callback is called.
        2) Respective sub-environment (gym.Env) is `reset()`.
        3) Callback `on_episode_start` is called.
        4) Stepping through sub-environment/episode commences.

        Args:
            episode: The newly created episode. On the new API stack, this will be a
                MultiAgentEpisode object. On the old API stack, this will be a
                Episode or EpisodeV2 object.
                This is the episode that is about to be started with an upcoming
                `env.reset()`. Only after this reset call, the `on_episode_start`
                callback will be called.
            env_runner: Replaces `worker` arg. Reference to the current EnvRunner.
            metrics_logger: The MetricsLogger object inside the `env_runner`. Can be
                used to log custom metrics after Episode creation.
            env: Replaces `base_env` arg.  The gym.Env (new API stack) or RLlib
                BaseEnv (old API stack) running the episode. On the old stack, the
                underlying sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            rl_module: Replaces `policies` arg. Either the RLModule (new API stack) or a
                dict mapping policy IDs to policy objects (old stack). In single agent
                mode there will only be a single policy/RLModule under the
                `rl_module["default_policy"]` key.
            env_index: The index of the sub-environment that is about to be reset
                (within the vector of sub-environments of the BaseEnv).
            kwargs: Forward compatibility placeholder.
        """
        print("Episode created!")

        if dfjspt_params.enable_curriculum:
            curriculum_task_manager = ray.get_actor("curriculum_task_manager")
            current_task_level = ray.get(curriculum_task_manager.get_current_level.remote())
            metrics_logger.log_value(key="current_task_level",
                                     value=current_task_level,
                                     reduce="max")
            print(f"Current task level is {current_task_level}")

            task_filename = ray.get(curriculum_task_manager.generate_task_filename.remote())
            reset_options = generate_reset_options_for_training(
                local_schedule_filename=task_filename,
                for_training=True,
            )
        else:
            reset_options = generate_reset_options_for_training(
                local_schedule_filename=None,
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
        """Callback run right after an Episode has been started.

        This method gets called after a SingleAgentEpisode or MultiAgentEpisode instance
        has been reset with a call to `env.reset()` by the EnvRunner.

        1) Single-/MultiAgentEpisode created: `on_episode_created()` is called.
        2) Respective sub-environment (gym.Env) is `reset()`.
        3) Single-/MultiAgentEpisode starts: This callback is called.
        4) Stepping through sub-environment/episode commences.

        Args:
            episode: The just started (after `env.reset()`) SingleAgentEpisode or
                MultiAgentEpisode object.
            env_runner: Reference to the EnvRunner running the env and episode.
            metrics_logger: The MetricsLogger object inside the `env_runner`. Can be
                used to log custom metrics during env/episode stepping.
            env: The gym.Env or gym.vector.Env object running the started episode.
            env_index: The index of the sub-environment that is about to be reset
                (within the vector of sub-environments of the BaseEnv).
            rl_module: The RLModule used to compute actions for stepping the env.
                In a single-agent setup, this is a (single-agent) RLModule, in a multi-
                agent setup, this will be a MultiRLModule.
            kwargs: Forward compatibility placeholder.
        """
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
        """Called on each episode step (after the action(s) has/have been logged).

        Note that on the new API stack, this callback is also called after the final
        step of an episode, meaning when terminated/truncated are returned as True
        from the `env.step()` call, but is still provided with the non-finalized
        episode object (meaning the data has NOT been converted to numpy arrays yet).

        The exact time of the call of this callback is after `env.step([action])` and
        also after the results of this step (observation, reward, terminated, truncated,
        infos) have been logged to the given `episode` object.

        Args:
            episode: The just stepped SingleAgentEpisode or MultiAgentEpisode object
                (after `env.step()` and after returned obs, rewards, etc.. have been
                logged to the episode object).
            env_runner: Reference to the EnvRunner running the env and episode.
            metrics_logger: The MetricsLogger object inside the `env_runner`. Can be
                used to log custom metrics during env/episode stepping.
            env: The gym.Env or gym.vector.Env object running the started episode.
            env_index: The index of the sub-environment that has just been stepped.
            rl_module: The RLModule used to compute actions for stepping the env.
                In a single-agent setup, this is a (single-agent) RLModule, in a multi-
                agent setup, this will be a MultiRLModule.
            kwargs: Forward compatibility placeholder.
        """
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
        """Called when an episode is done (after terminated/truncated have been logged).

        The exact time of the call of this callback is after `env.step([action])` and
        also after the results of this step (observation, reward, terminated, truncated,
        infos) have been logged to the given `episode` object, where either terminated
        or truncated were True:

        - The env is stepped: `final_obs, rewards, ... = env.step([action])`

        - The step results are logged `episode.add_env_step(final_obs, rewards)`

        - Callback `on_episode_step` is fired.

        - Another env-to-module connector call is made (even though we won't need any
          RLModule forward pass anymore). We make this additional call to ensure that in
          case users use the connector pipeline to process observations (and write them
          back into the episode), the episode object has all observations - even the
          terminal one - properly processed.

        - ---> This callback `on_episode_end()` is fired. <---

        - The episode is finalized (i.e. lists of obs/rewards/actions/etc.. are
          converted into numpy arrays).

        Args:
            episode: The terminated/truncated SingleAgent- or MultiAgentEpisode object
                (after `env.step()` that returned terminated=True OR truncated=True and
                after the returned obs, rewards, etc.. have been logged to the episode
                object). Note that this method is still called before(!) the episode
                object is finalized, meaning all its timestep data is still present in
                lists of individual timestep data.
            env_runner: Reference to the EnvRunner running the env and episode.
            metrics_logger: The MetricsLogger object inside the `env_runner`. Can be
                used to log custom metrics during env/episode stepping.
            env: The gym.Env or gym.vector.Env object running the started episode.
            env_index: The index of the sub-environment that has just been terminated
                or truncated.
            rl_module: The RLModule used to compute actions for stepping the env.
                In a single-agent setup, this is a (single-agent) RLModule, in a multi-
                agent setup, this will be a MultiRLModule.
            kwargs: Forward compatibility placeholder.
        """
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

        if dfjspt_params.enable_curriculum:
            curriculum_task_manager = ray.get_actor("curriculum_task_manager")
            current_task_level = ray.get(curriculum_task_manager.get_current_level.remote())
            metrics_logger.log_value(key="current_task_level",
                                     value=current_task_level,
                                     reduce="max")
            if result['training_iteration'] % 20 == 0:
                curriculum_task_manager.increase_task_level.remote()

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

