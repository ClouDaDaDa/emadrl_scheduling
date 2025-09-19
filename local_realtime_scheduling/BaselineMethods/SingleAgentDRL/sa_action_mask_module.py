import gymnasium as gym
from typing import Dict, Optional, Tuple, Union

from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.rl_module import RLModule, RLModuleConfig
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.typing import TensorType
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

torch, nn = try_import_torch()

class SAActionMaskingRLModule(RLModule):
    """Single-agent RLModule with action masking for safe RL.

    This RLModule implements action masking for a single agent that controls all resources.
    The action mask is extracted from the environment's observation and applied to the action logits.
    """
    @override(RLModule)
    def __init__(
        self,
        config=-1,
        *,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        inference_only: Optional[bool] = None,
        learner_only: bool = False,
        model_config: Optional[Union[dict, DefaultModelConfig]] = None,
        catalog_class=None,
    ):
        if not isinstance(observation_space, gym.spaces.Dict):
            raise ValueError(
                "This RLModule requires the environment to provide a "
                "gym.spaces.Dict observation space of the form: \n"
                " {'action_mask': Box(0.0, 1.0, shape=(n,)), 'observation': Box(...)}"
            )
        self.observation_space_with_mask = observation_space
        observation_space = observation_space["observation"]
        self._checked_observations = False
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            inference_only=inference_only,
            learner_only=learner_only,
            model_config=model_config,
            catalog_class=catalog_class,
        )

class SAActionMaskingTorchRLModule(SAActionMaskingRLModule, PPOTorchRLModule):
    @override(PPOTorchRLModule)
    def setup(self):
        super().setup()
        self.observation_space = self.observation_space_with_mask

    @override(PPOTorchRLModule)
    def _forward_inference(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        action_mask, batch = self._preprocess_batch(batch)
        outs = super()._forward_inference(batch, **kwargs)
        return self._mask_action_logits(outs, action_mask)

    @override(PPOTorchRLModule)
    def _forward_exploration(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        action_mask, batch = self._preprocess_batch(batch)
        outs = super()._forward_exploration(batch, **kwargs)
        return self._mask_action_logits(outs, action_mask)

    @override(PPOTorchRLModule)
    def _forward_train(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Dict[str, TensorType]:
        outs = super()._forward_train(batch, **kwargs)
        return self._mask_action_logits(outs, batch["action_mask"])

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, TensorType], embeddings=None):
        action_mask, batch = self._preprocess_batch(batch)
        batch["action_mask"] = action_mask
        return super().compute_values(batch, embeddings)

    def _preprocess_batch(
        self, batch: Dict[str, TensorType], **kwargs
    ) -> Tuple[TensorType, Dict[str, TensorType]]:
        self._check_batch(batch)
        if type(batch[Columns.OBS]) is dict:
            action_mask = batch[Columns.OBS].pop("action_mask")
            observation = batch[Columns.OBS].pop("observation")
            # For single agent, observation is already flat
            batch[Columns.OBS] = observation
            batch['action_mask'] = action_mask
        else:
            action_mask = batch['action_mask']
        return action_mask, batch

    def _mask_action_logits(
        self, batch: Dict[str, TensorType], action_mask: TensorType
    ) -> Dict[str, TensorType]:
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        batch[Columns.ACTION_DIST_INPUTS] += inf_mask
        return batch

    def _check_batch(self, batch: Dict[str, TensorType]) -> Optional[ValueError]:
        if not self._checked_observations:
            if "action_mask" not in batch[Columns.OBS]:
                raise ValueError(
                    "No action mask found in observation. This RLModule requires "
                    "the environment to provide observations that include an "
                    "action mask (i.e. an observation space of the Dict space "
                    "type that looks as follows: \n"
                    "{'action_mask': Box(0.0, 1.0, shape=(n,)), 'observation': Box(...) }"
                )
            if "observation" not in batch[Columns.OBS]:
                raise ValueError(
                    "No observations found in observation. This RLModule requires "
                    "the environment to provide observations that include the original "
                    "observations under a key 'observation' in a dict."
                )
            self._checked_observations = True 