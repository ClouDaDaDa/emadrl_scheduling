
## Changes to source code:

### 1. 

The source code of RLlib currently does not support observation input of dict type with action_mask, so the following corresponding modifications are required:

Add the following code into `/python3.10/site-packages/ray/rllib/core/models/catalog.py`:

1) Before Class `catalog`:

```
from gymnasium import spaces
```

```
class NestedModelConfig(ModelConfig):
    """Custom ModelConfig for handling nested observation spaces (Dict)."""

    def __init__(self, observation_space: spaces.Space, model_config_dict: dict):

        if not isinstance(observation_space, spaces.Dict):
            raise ValueError(f"Expected a Dict observation space, but got {type(observation_space)}")

        self.input_dims = self.flatten_and_concat(observation_space)

        self.mlp_config = MLPEncoderConfig(
            input_dims=(self.input_dims,),  
            hidden_layer_dims=model_config_dict.get("fcnet_hiddens", [256, 256]),
            hidden_layer_use_bias=model_config_dict.get("fcnet_use_bias", True),
            hidden_layer_activation=model_config_dict.get("fcnet_activation", "relu")
        )

    def flatten_and_concat(self, observation_space: spaces.Dict) -> int:
        total_dims = 0
        for key, space in observation_space.spaces.items():
            if isinstance(space, spaces.Box):
                total_dims += np.prod(space.shape)
            else:
                raise ValueError(f"Unsupported space type {type(space)} for key {key}")
        return total_dims

    def build(self, framework: str = "torch"):

        return self.mlp_config.build(framework=framework)

    @property
    def output_dims(self):

        return self.mlp_config.output_dims
```

2) in `_get_encoder_config()`:

```
    # NestedModelConfig
    encoder_config = NestedModelConfig(
        observation_space=observation_space,
        model_config_dict=model_config_dict
    )
    # raise ValueError(
    #     f"No default encoder config for obs space={observation_space},"
    #     f" lstm={use_lstm} found."
    # )
```


### 2.

The source code of RLlib currently does not support adding options data to the env.reset() method, so the following corresponding modifications are required:

Modify `_sample_episodes()` in `/python3.10/site-packages/ray/rllib/env/multi_agent_env_runner.py`:

```
        # Create a new multi-agent episode.
        _episode = self._new_episode()
        # self._make_on_episode_callback("on_episode_created", _episode)
        # Call the `on_episode_created` callback.
        env_options = self._callbacks.on_episode_created(
            episode=_episode,
            env_runner=self,
            metrics_logger=self.metrics,
            env=self.env.unwrapped,
            rl_module=self.module,
            env_index=0,
        )
        _shared_data = {
            "agent_to_module_mapping_fn": self.config.policy_mapping_fn,
        }

        # Try resetting the environment.
        # TODO (simon): Check, if we need here the seed from the config.
        obs, infos = self._try_env_reset(options=env_options)
```

in line `447`, and

```
            # Create a new episode instance.
            _episode = self._new_episode()
            # self._make_on_episode_callback("on_episode_created", _episode)
            # Call the `on_episode_created` callback.
            env_options = self._callbacks.on_episode_created(
                episode=_episode,
                env_runner=self,
                metrics_logger=self.metrics,
                env=self.env.unwrapped,
                rl_module=self.module,
                env_index=0,
            )
            
            # Try resetting the environment.
            obs, infos = self._try_env_reset(options=env_options)
```
in line `609`.

Then, modify `_try_env_reset()` in `/python3.10/site-packages/ray/rllib/env/env_runner.py`:

```
    def _try_env_reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
    # def _try_env_reset(self):
        """Tries resetting the env and - if an error orrurs - handles it gracefully."""
        # Try to reset.
        try:
            # obs, infos = self.env.reset()
            obs, infos = self.env.reset(seed=seed, options=options)
            # Everything ok -> return.
            return obs, infos
        # Error.
        except Exception as e:
            # If user wants to simply restart the env -> recreate env and try again
            # (calling this method recursively until success).
            if self.config.restart_failed_sub_environments:
                logger.exception(
                    "Resetting the env resulted in an error! The original error "
                    f"is: {e.args[0]}"
                )
                # Recreate the env and simply try again.
                self.make_env()
                # return self._try_env_reset()
                return self._try_env_reset(seed=seed, options=options)
            else:
                raise e
```

### 3.

The source code of RLlib currently has trouble with handling customized Class in ray.cloudpickle.load().

Add the following code in `/python3.10/site-packages/ray/_private/workers/default_worker.py`:

```
class Local_Job_schedule:
    def __init__(self, job_id):
        self.job_id = job_id
        self.operations = {}
        self.available_time = None
        self.estimated_finish_time = None

    def add_operation(self, operation):
        self.operations[operation.operation_id] = operation

    def __repr__(self):
        return f"Job(Job_ID={self.job_id}, Operations={self.operations})"


class LocalSchedule:
    def __init__(self):
        self.jobs = {}
        self.local_makespan = None
        self.time_window_start = None
        self.time_window_end = None

    def add_job(self, job):
        self.jobs[job.job_id] = job

    def __repr__(self):
        return f"LocalSchedule(Jobs={self.jobs})"
```
