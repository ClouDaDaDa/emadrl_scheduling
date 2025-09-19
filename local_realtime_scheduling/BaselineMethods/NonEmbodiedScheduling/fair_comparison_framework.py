"""
Fair Comparison Framework for Embodied vs Non-Embodied Scheduling

This module provides a framework for fair comparison between embodied and non-embodied
scheduling approaches by addressing the key challenge: how to evaluate both methods
on the same underlying scheduling problem while respecting their different modeling assumptions.

Key Design Principles:
1. Same problem instances and global schedules
2. Same evaluation metrics (final makespan on the SAME physical system)
3. Different execution environments (embodied vs abstract) but same final validation
4. Controlled experimental conditions

The framework implements a "dual-layer" evaluation approach:
- Training Layer: Each method trains in its native environment
- Evaluation Layer: Both methods are evaluated on the SAME embodied system
"""

import os
import sys
import pickle
import numpy as np
import copy
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import torch
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils.numpy import convert_to_numpy, softmax

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from configs import dfjspt_params
from System.FactoryInstance import FactoryInstance
from System.SchedulingInstance import SchedulingInstance
from local_realtime_scheduling.Environment.LocalSchedulingMultiAgentEnv_v3_4 import LocalSchedulingMultiAgentEnv
from local_realtime_scheduling.Agents.generate_training_data import generate_reset_options_for_training
# from local_realtime_scheduling.BaselineMethods.NonEmbodiedScheduling import NonEmbodiedSchedulingMultiAgentEnv
from local_realtime_scheduling.InterfaceWithGlobal.divide_global_schedule_to_local_from_pkl import LocalSchedule, Local_Job_schedule


@dataclass
class ComparisonResult:
    """Results from comparing embodied vs non-embodied methods."""
    instance_name: str
    embodied_makespan: float
    non_embodied_makespan: float
    embodied_execution_time: float
    non_embodied_execution_time: float
    embodied_decisions: int
    non_embodied_decisions: int
    embodied_success: bool
    non_embodied_success: bool
    
    @property
    def makespan_improvement(self) -> float:
        """Improvement of embodied over non-embodied (positive = embodied better)."""
        return (self.non_embodied_makespan - self.embodied_makespan) / self.non_embodied_makespan * 100


class FairComparisonFramework:
    """
    Framework for fair comparison between embodied and non-embodied scheduling.
    
    The key insight is that we need to separate:
    1. Training environments (can be different)
    2. Evaluation environment (must be the same - the embodied one represents "reality")
    
    The comparison works as follows:
    1. Train both methods in their respective environments
    2. Extract decision policies from both trained models
    3. Evaluate both policies on the SAME embodied environment
    4. Compare final makespans achieved in the real (embodied) system
    """
    
    def __init__(
        self, 
        config: Dict[str, Any], 
        embodied_checkpoint_dir: str, 
        non_embodied_checkpoint_dir: str,
    ):
        self.config = config
        self.n_machines = config["n_machines"]
        self.n_transbots = config["n_transbots"] 
        self.factory_seed = config["factory_instance_seed"]
        
        # Create the "ground truth" embodied environment for evaluation
        self.embodied_env = LocalSchedulingMultiAgentEnv({
            "n_machines": self.n_machines,
            "n_transbots": self.n_transbots,
            "factory_instance_seed": self.factory_seed,
            "enable_dynamic_agent_filtering": False,  # Disable for consistent comparison
        })
        
        # # Create the abstract non-embodied environment for training/testing
        # self.non_embodied_env = NonEmbodiedSchedulingMultiAgentEnv({
        #     "n_machines": self.n_machines,
        #     "n_transbots": self.n_transbots,
        #     "factory_instance_seed": self.factory_seed,
        # })

        self.embodied_policies = self._get_madrl_policies(embodied_checkpoint_dir)
        self.non_embodied_policies = self._get_madrl_policies(non_embodied_checkpoint_dir)
     
    
    def _get_madrl_policies(self, checkpoint_dir: str) -> Tuple[RLModule, RLModule]:
        """Get MADRL policies from checkpoint directory."""
        machine_rl_module_checkpoint_dir = Path(checkpoint_dir) / "learner_group" / "learner" / "rl_module" / "p_machine"
        transbot_rl_module_checkpoint_dir = Path(checkpoint_dir) / "learner_group" / "learner" / "rl_module" / "p_transbot"
        
        machine_rl_module = RLModule.from_checkpoint(machine_rl_module_checkpoint_dir)
        transbot_rl_module = RLModule.from_checkpoint(transbot_rl_module_checkpoint_dir)
        
        return (machine_rl_module, transbot_rl_module)

    def _get_madrl_actions(self, policy_type: str, observations: Dict[str, Any]) -> Dict[str, Any]:
        """Get MADRL actions from observations."""
        actions = {}
        for agent_id, obs in observations.items():
            try:
                if policy_type == "embodied":
                    if agent_id.startswith("machine"):
                        input_dict = {
                            "obs": {
                                "action_mask": torch.tensor(obs["action_mask"]).unsqueeze(0),
                                "observation": {
                                    "job_features": torch.tensor(obs["observation"]["job_features"]).unsqueeze(0),
                                    "neighbor_machines_features": torch.tensor(
                                        obs["observation"]["neighbor_machines_features"]).unsqueeze(0),
                                    "machine_features": torch.tensor(obs["observation"]["machine_features"]).unsqueeze(0),
                                }
                            }
                        }
                        rl_module_out = self.embodied_policies[0].forward_inference(input_dict)
                    else:
                        input_dict = {
                            "obs": {
                                "action_mask": torch.tensor(obs["action_mask"]).unsqueeze(0),
                                "observation": {
                                    "job_features": torch.tensor(obs["observation"]["job_features"]).unsqueeze(0),
                                    "neighbor_transbots_features": torch.tensor(
                                        obs["observation"]["neighbor_transbots_features"]).unsqueeze(0),
                                    "transbot_features": torch.tensor(obs["observation"]["transbot_features"]).unsqueeze(0),
                                }
                            }
                        }
                        rl_module_out = self.embodied_policies[1].forward_inference(input_dict)
                else:
                    # non-embodied policy expects global_features + job_features only
                    if agent_id.startswith("machine"):
                        input_dict = {
                            "obs": {
                                "action_mask": torch.tensor(obs["action_mask"]).unsqueeze(0),
                                "observation": {
                                    "global_features": torch.tensor(obs["observation"]["global_features"]).unsqueeze(0),
                                    "job_features": torch.tensor(obs["observation"]["job_features"]).unsqueeze(0),
                                }
                            }
                        }
                        rl_module_out = self.non_embodied_policies[0].forward_inference(input_dict)
                    else:
                        input_dict = {
                            "obs": {
                                "action_mask": torch.tensor(obs["action_mask"]).unsqueeze(0),
                                "observation": {
                                    "global_features": torch.tensor(obs["observation"]["global_features"]).unsqueeze(0),
                                    "job_features": torch.tensor(obs["observation"]["job_features"]).unsqueeze(0),
                                }
                            }
                        }
                        rl_module_out = self.non_embodied_policies[1].forward_inference(input_dict)
            except Exception as e:
                print(f"[FairCompare] Inference failed for agent {agent_id} (policy={policy_type}): {e}")
                if "observation" in obs:
                    try:
                        keys = list(obs["observation"].keys())
                        print(f"[FairCompare] Obs keys: {keys}")
                        for k in keys:
                            arr = obs["observation"].get(k)
                            if hasattr(arr, 'shape'):
                                print(f"[FairCompare] {agent_id}.{k}.shape={arr.shape}")
                    except Exception:
                        pass
                am = obs.get("action_mask")
                if am is not None:
                    try:
                        print(f"[FairCompare] action_mask.len={len(am)} enabled={sum(1 for x in am if x==1)}")
                    except Exception:
                        pass
                raise
            
            logits = convert_to_numpy(rl_module_out['action_dist_inputs'])[0]
            # Mask invalid logits using action_mask if provided
            am = obs.get("action_mask")
            if am is not None:
                invalid = (np.arange(len(logits)) >= len(am)) | (np.array(am) == 0)
                logits = np.where(invalid, -1e9, logits)
            # Numerically stable softmax
            finite_logits = np.where(np.isfinite(logits), logits, -1e9)
            max_logit = np.max(finite_logits)
            shifted = finite_logits - max_logit
            with np.errstate(over='ignore', invalid='ignore'):
                x_exp = np.exp(shifted)
                x_exp = np.where(np.isfinite(x_exp), x_exp, 0.0)
                denom = np.sum(x_exp)
            if not np.isfinite(denom) or denom <= 0:
                # Fallback: pick best valid index or first enabled
                if am is not None:
                    enabled = [i for i,v in enumerate(am) if v==1]
                    chosen = int(np.argmax(finite_logits[enabled])) if enabled else 0
                    actions[agent_id] = enabled[chosen] if enabled else 0
                else:
                    actions[agent_id] = int(np.argmax(finite_logits)) if len(finite_logits)>0 else 0
            else:
                probs = x_exp / denom
                if am is not None:
                    # Zero out any residual invalid mass and renormalize
                    mask_arr = np.array(am, dtype=np.float32)
                    mask_arr = np.pad(mask_arr, (0, max(0, len(probs)-len(mask_arr))), constant_values=0)
                    probs = probs * mask_arr[:len(probs)]
                    s = probs.sum()
                    if s <= 0 or not np.isfinite(s):
                        enabled = [i for i,v in enumerate(am) if v==1]
                        actions[agent_id] = enabled[0] if enabled else 0
                        continue
                    probs = probs / s
                # Safety: replace tiny negatives and NaNs
                probs = np.where(np.isfinite(probs), probs, 0.0)
                probs = np.clip(probs, 0.0, 1.0)
                s = probs.sum()
                if s <= 0:
                    # Final fallback
                    if am is not None:
                        enabled = [i for i,v in enumerate(am) if v==1]
                        actions[agent_id] = enabled[0] if enabled else 0
                    else:
                        actions[agent_id] = int(np.argmax(finite_logits)) if len(finite_logits)>0 else 0
                else:
                    probs = probs / s
                    actions[agent_id] = int(np.random.choice(len(probs), p=probs))
        return actions
    
    def compare_on_instance(
        self,
        reset_options: Dict[str, Any],
        num_episodes: int = 10,
        detailed_log: bool = False
    ) -> List[ComparisonResult]:
        """
        Compare embodied vs non-embodied methods on a single instance.
        
        Both methods are ultimately evaluated on the SAME
        embodied environment to ensure fair comparison of final outcomes.
        
        Args:
            reset_options: Reset options
            num_episodes: Number of evaluation episodes
            detailed_log: Whether to print detailed logs
            
        Returns:
            List of comparison results for each episode
        """
        results = []
        
        for episode in range(num_episodes):
            if detailed_log:
                print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
            
            # Evaluate embodied method on embodied environment (native)
            embodied_result = self._evaluate_embodied_method(
                reset_options=copy.deepcopy(reset_options),
                detailed_log=detailed_log
            )
            
            # Evaluate non-embodied method on embodied environment (cross-evaluation)
            non_embodied_result = self._evaluate_non_embodied_on_embodied(
                reset_options=copy.deepcopy(reset_options),
                detailed_log=detailed_log
            )
            
            result = ComparisonResult(
                instance_name=f"episode_{episode}",
                embodied_makespan=embodied_result["makespan"],
                non_embodied_makespan=non_embodied_result["makespan"],
                embodied_execution_time=embodied_result["execution_time"],
                non_embodied_execution_time=non_embodied_result["execution_time"],
                embodied_decisions=embodied_result["decisions"],
                non_embodied_decisions=non_embodied_result["decisions"],
                embodied_success=embodied_result["success"],
                non_embodied_success=non_embodied_result["success"]
            )
            
            results.append(result)
            
            if detailed_log:
                print(f"Embodied makespan: {result.embodied_makespan:.2f}")
                print(f"Non-embodied makespan: {result.non_embodied_makespan:.2f}")
                print(f"Improvement: {result.makespan_improvement:.2f}%")
        
        return results
    
    def _evaluate_embodied_method(
        self,
        reset_options: Dict[str, Any],
        detailed_log: bool = False
    ) -> Dict[str, Any]:
        """Evaluate embodied method on embodied environment."""

        print(f"\n[FairCompare] Evaluating embodied method on embodied environment...")
        
        start_exec_time = time.time()
        observations, _ = self.embodied_env.reset(options=reset_options)
        
        decision_count = 0
        terminated = {"__all__": False}
        truncated = {"__all__": False}
        
        while not (terminated["__all__"] or truncated["__all__"]):  
            # Get MADRL actions from embodied policy
            actions = self._get_madrl_actions(policy_type="embodied", observations=observations)
            # Step the environment
            observations, rewards, terminated, truncated, infos = self.embodied_env.step(actions)
            decision_count += 1
        
        execution_time = time.time() - start_exec_time
        # Extract results - for local instances, we use delta makespan
        if self.embodied_env.local_result.actual_local_makespan is not None:
            delta_makespan = self.embodied_env.local_result.actual_local_makespan - self.embodied_env.local_result.time_window_start
        else:
            delta_makespan = self.embodied_env.current_time_after_step - self.embodied_env.local_result.time_window_start
        success = terminated["__all__"]
        
        return {
            "makespan": delta_makespan,
            "execution_time": execution_time,
            "decisions": decision_count,
            "success": success
        }
    
    def _evaluate_non_embodied_on_embodied(
        self,
        reset_options: Dict[str, Any],
        detailed_log: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate non-embodied method on embodied environment.
        """

        print(f"\n[FairCompare] Evaluating non-embodied method on embodied environment...")

        start_exec_time = time.time()
        observations, _ = self.embodied_env.reset(options=reset_options)
        
        decision_count = 0
        terminated = {"__all__": False}
        truncated = {"__all__": False}
        # Thresholds for heuristic safety actions
        reliability_threshold = 0.5
        soc_threshold = 0.5

        # Initialize previous status snapshots for event detection
        prev_machine_status = {m.machine_id: m.machine_status for m in self.embodied_env.factory_instance.machines}
        prev_transbot_status = {t.agv_id: t.agv_status for t in self.embodied_env.factory_instance.agv}

        # On the very first decision we allow policy actions (treat as event step)
        has_event_step = True

        print(f"Remaining operations: {self.embodied_env.remaining_operations}")
        
        while not (terminated["__all__"] or truncated["__all__"]):  
            # Build actions dict for currently acting agents only (respect current decision stage)
            actions: Dict[str, Any] = {}

            # Stage awareness: 0 -> machines act; 1 -> transbots act
            current_stage = self.embodied_env.decision_stage

            # If no completion event previously detected, but there are feasible job actions now,
            # allow policy decisions to kick off work (avoid deadlock of perpetual do-nothing)
            if not has_event_step:
                for agent_id, obs in observations.items():
                    if current_stage == 0 and not agent_id.startswith("machine"):
                        continue
                    if current_stage == 1 and not agent_id.startswith("transbot"):
                        continue
                    am = obs.get("action_mask")
                    if am is None:
                        continue
                    # job slots are the first K entries of job_features length
                    if "observation" in obs and "job_features" in obs["observation"]:
                        k = obs["observation"]["job_features"].shape[0]
                        if any((i < len(am) and am[i] == 1) for i in range(min(k, len(am)))):
                            has_event_step = True
                            break

            if has_event_step:
                # Step 1: Map embodied observations to non-embodied format (only for current stage agents)
                non_obs, map_ctx = self._map_embodied_to_non_embodied_obs(observations, current_stage)

                # Step 2: Get actions from non-embodied policy (for the provided agents only)
                non_actions = self._get_madrl_actions(policy_type="non_embodied", observations=non_obs)

                # Step 3: Map actions back to embodied format using current embodied observations and masks
                actions = self._map_non_embodied_to_embodied_actions(non_actions, observations, map_ctx)
            else:
                # No event: default to do-nothing for currently acting agents
                for agent_id, obs in observations.items():
                    if current_stage == 0 and not agent_id.startswith("machine"):
                        continue
                    if current_stage == 1 and not agent_id.startswith("transbot"):
                        continue
                    action_mask = obs.get("action_mask")
                    if agent_id.startswith("machine"):
                        # Do-nothing index is len(job_features) + 4
                        k = obs["observation"]["job_features"].shape[0] if "observation" in obs else 0
                        actions[agent_id] = k + 4 if action_mask is not None and len(action_mask) > k + 4 else 0
                    else:
                        # Do-nothing index is len(job_features) + 1
                        k = obs["observation"]["job_features"].shape[0] if "observation" in obs else 0
                        actions[agent_id] = k + 1 if action_mask is not None and len(action_mask) > k + 1 else 0

            # Safety override: maintenance/charging when below thresholds
            for agent_id, obs in observations.items():
                if current_stage == 0 and not agent_id.startswith("machine"):
                    continue
                if current_stage == 1 and not agent_id.startswith("transbot"):
                    continue
                action_mask = obs.get("action_mask")
                if agent_id.startswith("machine"):
                    m_idx = int(agent_id.lstrip("machine"))
                    m = self.embodied_env.factory_instance.machines[m_idx]
                    if m.reliability < reliability_threshold:
                        k = obs["observation"]["job_features"].shape[0] if "observation" in obs else 0
                        # Maintenance indices: k..k+3; prefer CM if faulty, else LM
                        cm_idx = k + 3
                        lm_idx = k + 0
                        chosen = cm_idx if m.machine_status == 3 else lm_idx
                        if action_mask is not None and 0 <= chosen < len(action_mask) and action_mask[chosen] == 1:
                            actions[agent_id] = chosen
                else:
                    t_idx = int(agent_id.lstrip("transbot"))
                    t = self.embodied_env.factory_instance.agv[t_idx]
                    if t.battery.soc < soc_threshold:
                        k = obs["observation"]["job_features"].shape[0] if "observation" in obs else 0
                        charge_idx = k  # charging is at index k
                        if action_mask is not None and 0 <= charge_idx < len(action_mask) and action_mask[charge_idx] == 1:
                            actions[agent_id] = charge_idx

            # Final fallback: ensure each acting agent has an action; if not, set do-nothing
            for agent_id, obs in observations.items():
                if current_stage == 0 and not agent_id.startswith("machine"):
                    continue
                if current_stage == 1 and not agent_id.startswith("transbot"):
                    continue
                if agent_id not in actions:
                    action_mask = obs.get("action_mask")
                    if agent_id.startswith("machine"):
                        k = obs["observation"]["job_features"].shape[0] if "observation" in obs else 0
                        dn = k + 4
                        actions[agent_id] = dn if action_mask is None or (0 <= dn < len(action_mask) and action_mask[dn] == 1) else 0
                    else:
                        k = obs["observation"]["job_features"].shape[0] if "observation" in obs else 0
                        dn = k + 1
                        actions[agent_id] = dn if action_mask is None or (0 <= dn < len(action_mask) and action_mask[dn] == 1) else 0

            # Validate actions against masks and coerce invalid ones to do-nothing with diagnostics
            for agent_id, obs in observations.items():
                if current_stage == 0 and not agent_id.startswith("machine"):
                    continue
                if current_stage == 1 and not agent_id.startswith("transbot"):
                    continue
                if agent_id not in actions:
                    continue
                am = obs.get("action_mask")
                act = int(actions[agent_id])
                if am is not None and (act < 0 or act >= len(am) or am[act] != 1):
                    # Coerce to do-nothing
                    if agent_id.startswith("machine"):
                        k = obs["observation"]["job_features"].shape[0] if "observation" in obs else 0
                        dn = k + 4
                    else:
                        k = obs["observation"]["job_features"].shape[0] if "observation" in obs else 0
                        dn = k + 1
                    fallback = dn if 0 <= dn < len(am) and am[dn] == 1 else 0
                    print(f"[FairCompare] Invalid action {act} for {agent_id} at stage {current_stage}; fallback -> {fallback}")
                    actions[agent_id] = fallback

            # Step the embodied environment
            try:
                observations, rewards, terminated, truncated, infos = self.embodied_env.step(actions)
            except Exception as e:
                # Print rich diagnostics and re-raise
                print(f"[FairCompare] Step failed at stage {current_stage}: {e}")
                print(f"[FairCompare] Actions summary: { {aid:int(a) for aid,a in actions.items()} }")
                for aid, obs in observations.items():
                    if current_stage == 0 and not aid.startswith("machine"):
                        continue
                    if current_stage == 1 and not aid.startswith("transbot"):
                        continue
                    am = obs.get("action_mask")
                    if am is not None:
                        enabled = [i for i,x in enumerate(am) if x==1]
                        print(f"[FairCompare] {aid} enabled={enabled[:10]}... total={len(enabled)}")
                raise
            decision_count += 1
            
            # Update event detection for next iteration
            cur_machine_status = {m.machine_id: m.machine_status for m in self.embodied_env.factory_instance.machines}
            cur_transbot_status = {t.agv_id: t.agv_status for t in self.embodied_env.factory_instance.agv}

            event_detected = False
            # Machine completes processing/maintenance -> status transitions to idle (0) from {1,2}
            for mid, st in cur_machine_status.items():
                pst = prev_machine_status.get(mid, st)
                if st == 0 and pst in (1, 2):
                    event_detected = True
                    break
            # Transbot completes moving/charging -> status transitions to idle (0) from {1,2,3}
            if not event_detected:
                for tid, st in cur_transbot_status.items():
                    pst = prev_transbot_status.get(tid, st)
                    if st == 0 and pst in (1, 2, 3):
                        event_detected = True
                        break

            has_event_step = event_detected
            prev_machine_status = cur_machine_status
            prev_transbot_status = cur_transbot_status

            # print(f"Remaining operations: {self.embodied_env.remaining_operations}")
        
        execution_time = time.time() - start_exec_time
        # Extract results - for local instances, we use delta makespan
        if self.embodied_env.local_result.actual_local_makespan is not None:
            delta_makespan = self.embodied_env.local_result.actual_local_makespan - self.embodied_env.local_result.time_window_start
        else:
            delta_makespan = self.embodied_env.current_time_after_step - self.embodied_env.local_result.time_window_start
        success = terminated["__all__"]
        
        return {
            "makespan": delta_makespan,
            "execution_time": execution_time,
            "decisions": decision_count,
            "success": success
        }
        
    def _map_embodied_to_non_embodied_obs(self, embodied_observations: Dict[str, Any], current_stage: int):
        """
        Map embodied observations to non-embodied observation structure for the acting agents only.
        Returns (non_embodied_observations, context) where context stores per-agent job ordering for action mapping.
        """
        # Non-embodied observation specs
        max_jobs_in_obs = 20
        global_features_dim = 10
        job_features_dim = 8

        # Helpers
        fi = self.embodied_env.factory_instance
        si = self.embodied_env.scheduling_instance
        graph = fi.factory_graph
        unload_time_matrix = graph.unload_transport_time_matrix
        location_map = graph.location_index_map

        # Gather current job ids in this local schedule
        all_job_ids = [jid for jid in self.embodied_env.local_schedule.jobs.keys()]

        # Construct global features
        def calc_global_features() -> np.ndarray:
            g = np.zeros(global_features_dim, dtype=np.float32)
            g[0] = self.embodied_env.current_time_after_step
            g[1] = 0.0  # placeholder for event_step (unused by policy at inference)
            g[2] = float(self.embodied_env.remaining_operations)
            g[3] = float(self.embodied_env.total_n_ops_for_curr_tw)
            g[4] = g[2] / max(g[3], 1.0)
            busy = sum(1 for m in fi.machines if m.machine_status == 1)
            g[5] = busy / max(len(fi.machines), 1)
            # Job status counts
            ready_jobs = 0
            processing_jobs = 0
            completed_jobs = 0
            for jid in all_job_ids:
                jj = si.jobs[jid]
                if jj.job_status == 3:
                    completed_jobs += 1
                elif jj.job_status == 1:
                    processing_jobs += 1
                else:
                    # treat as ready if not completed and has pending op
                    if jj.current_processing_operation < len(jj.operations_matrix):
                        ready_jobs += 1
            g[6] = ready_jobs
            g[7] = processing_jobs
            g[8] = completed_jobs
            g[9] = max(0.0, (self.embodied_env.initial_estimated_makespan or g[0]) - g[0])
            return g

        global_features = calc_global_features()

        non_obs: Dict[str, Any] = {}
        map_ctx: Dict[str, Any] = {"available_jobs": {}}

        for agent_id, obs in embodied_observations.items():
            if current_stage == 0 and not agent_id.startswith("machine"):
                continue
            if current_stage == 1 and not agent_id.startswith("transbot"):
                continue

            # Determine per-agent feasible jobs directly from embodied Top-K job_features to ensure index alignment
            available_jobs: List[int] = []
            if "observation" in obs and "job_features" in obs["observation"]:
                jf_emb = obs["observation"]["job_features"]
                am = obs.get("action_mask")
                # Only consider the job slots in 0..K-1 that are enabled by mask
                for idx in range(jf_emb.shape[0]):
                    jid = int(jf_emb[idx, 0])
                    if jid < 0:
                        continue
                    if am is not None and (idx >= len(am) or am[idx] != 1):
                        continue
                    available_jobs.append(jid)

            available_jobs = available_jobs[:max_jobs_in_obs]
            map_ctx["available_jobs"][agent_id] = available_jobs

            # Build job_features matrix (non-embodied spec: id, current_op, progress, pt(min/max/avg), tt(min/avg))
            jf = np.full((max_jobs_in_obs, job_features_dim), -1, dtype=np.float32)
            for i, jid in enumerate(available_jobs):
                job = si.jobs[jid]
                cur_op = job.current_processing_operation
                pt_row = job.operations_matrix[cur_op, :] if cur_op < len(job.operations_matrix) else np.zeros((self.n_machines,))
                valid_pts = [float(pt) for pt in pt_row if pt > 0]
                min_pt = min(valid_pts) if valid_pts else 0.0
                max_pt = max(valid_pts) if valid_pts else 0.0
                avg_pt = float(sum(valid_pts) / len(valid_pts)) if valid_pts else 0.0
                jloc_idx = location_map[job.current_location]
                ts = [float(unload_time_matrix[jloc_idx, m]) for m in range(self.n_machines) if (cur_op < len(job.operations_matrix) and job.operations_matrix[cur_op, m] > 0)]
                min_tt = min(ts) if ts else 0.0
                avg_tt = float(sum(ts) / len(ts)) if ts else 0.0
                jf[i] = np.array([
                    float(jid),
                    float(cur_op),
                    float(getattr(job, 'job_progress_for_current_time_window', 0.0)),
                    min_pt, max_pt, avg_pt,
                    min_tt, avg_tt
                ], dtype=np.float32)

            # Build action mask for non-embodied: allow the listed jobs + do-nothing
            am = np.zeros((max_jobs_in_obs + 1,), dtype=np.int32)
            if available_jobs:
                am[:len(available_jobs)] = 1
            am[-1] = 1

            non_obs[agent_id] = {
                "action_mask": am,
                "observation": {
                    "global_features": global_features.copy(),
                    "job_features": jf
                }
            }

        return non_obs, map_ctx

    def _map_non_embodied_to_embodied_actions(self, non_actions: Dict[str, Any], embodied_observations: Dict[str, Any], map_ctx: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map non-embodied actions (indices over available_jobs + do-nothing) back to embodied action indices.
        Use current embodied observations' job_features to locate the correct action index; fallback to do-nothing if not found or masked.
        """
        actions: Dict[str, Any] = {}
        for agent_id, na in non_actions.items():
            obs = embodied_observations.get(agent_id)
            if obs is None:
                continue
            action_mask = obs.get("action_mask")
            # Do-nothing in non-embodied is the last index (max_jobs_in_obs)
            # Our constructed obs had size 20+1; interpret generally: if chosen >= len(available_jobs) -> do-nothing
            avail_jobs = map_ctx.get("available_jobs", {}).get(agent_id, [])
            if na >= len(avail_jobs):
                # Map to embodied do-nothing
                if agent_id.startswith("machine"):
                    k = obs["observation"]["job_features"].shape[0]
                    embodied_idx = k + 4
                else:
                    k = obs["observation"]["job_features"].shape[0]
                    embodied_idx = k + 1
                if action_mask is None or (0 <= embodied_idx < len(action_mask) and action_mask[embodied_idx] == 1):
                    actions[agent_id] = embodied_idx
                else:
                    actions[agent_id] = 0
                continue

            # Non-embodied chose a job
            chosen_job_id = avail_jobs[na]
            # Find this job in embodied job_features to get the correct action index
            if "observation" in obs and "job_features" in obs["observation"]:
                jf = obs["observation"]["job_features"]
                embodied_idx = None
                for j in range(jf.shape[0]):
                    if int(jf[j, 0]) == int(chosen_job_id):
                        embodied_idx = j
                        break
                if embodied_idx is not None and (action_mask is None or (0 <= embodied_idx < len(action_mask) and action_mask[embodied_idx] == 1)):
                    actions[agent_id] = embodied_idx
                else:
                    # If job not in Top-K or masked, fallback to do-nothing
                    if agent_id.startswith("machine"):
                        k = jf.shape[0]
                        dn = k + 4
                    else:
                        k = jf.shape[0]
                        dn = k + 1
                    actions[agent_id] = dn if action_mask is None or (0 <= dn < len(action_mask) and action_mask[dn] == 1) else 0
            else:
                # No observation details; fallback to do-nothing
                if agent_id.startswith("machine"):
                    actions[agent_id] = 4
                else:
                    actions[agent_id] = 1

        return actions

    
    def analyze_results(self, results: List[ComparisonResult]) -> Dict[str, Any]:
        """Analyze comparison results and generate summary statistics."""
        
        if not results:
            # Return a full-shaped summary with zeros to avoid KeyError at print time
            return {
                "total_episodes": 0,
                "embodied_success_rate": 0.0,
                "non_embodied_success_rate": 0.0,
                "embodied_makespan_mean": float('inf'),
                "embodied_makespan_std": 0.0,
                "non_embodied_makespan_mean": float('inf'),
                "non_embodied_makespan_std": 0.0,
                "improvement_mean": 0.0,
                "improvement_std": 0.0,
                "improvement_median": 0.0,
                "embodied_wins": 0,
                "non_embodied_wins": 0,
                "ties": 0,
                "statistically_significant": False,
            }
        
        # Basic statistics
        embodied_makespans = [r.embodied_makespan for r in results if r.embodied_success]
        non_embodied_makespans = [r.non_embodied_makespan for r in results if r.non_embodied_success]
        improvements = [r.makespan_improvement for r in results if r.embodied_success and r.non_embodied_success]
        
        analysis = {
            "total_episodes": len(results),
            "embodied_success_rate": sum(r.embodied_success for r in results) / len(results),
            "non_embodied_success_rate": sum(r.non_embodied_success for r in results) / len(results),
            
            "embodied_makespan_mean": np.mean(embodied_makespans) if embodied_makespans else float('inf'),
            "embodied_makespan_std": np.std(embodied_makespans) if embodied_makespans else 0,
            "non_embodied_makespan_mean": np.mean(non_embodied_makespans) if non_embodied_makespans else float('inf'),
            "non_embodied_makespan_std": np.std(non_embodied_makespans) if non_embodied_makespans else 0,
            
            "improvement_mean": np.mean(improvements) if improvements else 0,
            "improvement_std": np.std(improvements) if improvements else 0,
            "improvement_median": np.median(improvements) if improvements else 0,
            
            "embodied_wins": sum(1 for r in results if r.embodied_success and r.non_embodied_success and r.makespan_improvement > 0),
            "non_embodied_wins": sum(1 for r in results if r.embodied_success and r.non_embodied_success and r.makespan_improvement < 0),
            "ties": sum(1 for r in results if r.embodied_success and r.non_embodied_success and abs(r.makespan_improvement) < 0.1),
        }
        
        # Statistical significance (placeholder)
        analysis["statistically_significant"] = len(improvements) > 10 and abs(analysis["improvement_mean"]) > analysis["improvement_std"]
        
        return analysis


def run_comparison_experiment(
    local_schedule_files: List[str],
    embodied_checkpoint_dir: str,
    non_embodied_checkpoint_dir: str,
    num_episodes_per_instance: int = 10,
    detailed_log: bool = False
) -> Tuple[List[ComparisonResult], Dict[str, Any]]:
    """
    Run a comprehensive comparison experiment.
    
    Args:
        local_schedule_files: List of local schedule file paths
        embodied_checkpoint_dir: Embodied checkpoint directory
        non_embodied_checkpoint_dir: Non-embodied checkpoint directory
        num_episodes_per_instance: Number of episodes per instance
        detailed_log: Whether to print detailed logs
        
    Returns:
        Tuple of (all results, summary analysis)
    """
    
    config = {
        "n_machines": dfjspt_params.n_machines,
        "n_transbots": dfjspt_params.n_transbots,
        "factory_instance_seed": dfjspt_params.factory_instance_seed,
    }

    framework = FairComparisonFramework(config, embodied_checkpoint_dir, non_embodied_checkpoint_dir)

    all_results = []
    
    print(f"Starting comparison experiment with {len(local_schedule_files)} instances")
    print(f"Episodes per instance: {num_episodes_per_instance}")
    print("=" * 60)
    
    for i, schedule_file in enumerate(local_schedule_files):
        print(f"\nProcessing instance {i+1}/{len(local_schedule_files)}: {schedule_file}")
        
        try:
            # # Load the instance
            # with open(schedule_file, "rb") as f:
            #     local_schedule = pickle.load(f)
            
            # # Create factory and scheduling instances (simplified for demo)
            # factory_instance = FactoryInstance(
            #     seed=config["factory_instance_seed"],
            #     n_machines=config["n_machines"],
            #     n_transbots=config["n_transbots"]
            # )
            
            # # Create a minimal scheduling instance
            # scheduling_instance = SchedulingInstance(
            #     seed=42,
            #     n_jobs=len(local_schedule.jobs),
            #     n_machines=config["n_machines"]
            # )
            
            # start_time = local_schedule.time_window_start

            reset_options = generate_reset_options_for_training(
                local_schedule_filename=schedule_file,
                for_training=False,
            )
            
            # Run comparison
            instance_results = framework.compare_on_instance(
                reset_options=reset_options,
                num_episodes=num_episodes_per_instance,
                detailed_log=detailed_log
            )
            
            all_results.extend(instance_results)
            
            # Print instance summary
            avg_improvement = np.mean([r.makespan_improvement for r in instance_results 
                                    if r.embodied_success and r.non_embodied_success])
            print(f"Instance average improvement: {avg_improvement:.2f}%")
            
        except Exception as e:
            print(f"Error processing {schedule_file}: {e}")
            continue
    
    # Analyze overall results
    analysis = framework.analyze_results(all_results)
    
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Total episodes: {analysis['total_episodes']}")
    print(f"Embodied success rate: {analysis['embodied_success_rate']:.2%}")
    print(f"Non-embodied success rate: {analysis['non_embodied_success_rate']:.2%}")
    print(f"Average improvement (embodied vs non-embodied): {analysis['improvement_mean']:.2f}%")
    print(f"Embodied wins: {analysis['embodied_wins']}")
    print(f"Non-embodied wins: {analysis['non_embodied_wins']}")
    print(f"Ties: {analysis['ties']}")
    
    return all_results, analysis


if __name__ == "__main__":
    # Example usage
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    EMBODIED_CHECKPOINT_DIR = ROOT_DIR + "/Agents/ray_results/M36T20W300/PPO_LocalSchedulingMultiAgentEnv_6589e_00000_0_2025-08-28_21-53-37/checkpoint_000049"

    # NON_EMBODIED_CHECKPOINT_DIR = ROOT_DIR + "/BaselineMethods/NonEmbodiedScheduling/ray_results_non_embodied/M36T20W300/PPO_NonEmbodiedSchedulingMultiAgentEnv_cc493_00000_0_2025-09-02_16-06-58/checkpoint_000103"
    NON_EMBODIED_CHECKPOINT_DIR = ROOT_DIR + "/BaselineMethods/NonEmbodiedScheduling/ray_results_non_embodied/M36T20W300/PPO_NonEmbodiedSchedulingMultiAgentEnv_0fabf_00000_0_2025-09-04_23-51-46/checkpoint_000199"

    LOCAL_SCHEDULE_FILES = [
        "local_schedule_J10I100_0_ops45.pkl",
        "local_schedule_J10I100_1_ops42.pkl",
        "local_schedule_J10I100_2_ops15.pkl",
        "local_schedule_J60I104_0_ops174.pkl",
        "local_schedule_J60I104_1_ops166.pkl",
        "local_schedule_J60I104_2_ops170.pkl",
        "local_schedule_J60I104_3_ops90.pkl",
        "local_schedule_J80I100_4_ops121.pkl",
        "local_schedule_J10I101_2_ops10.pkl"
    ]
    # n_jobs = 10
    # instance_id = 100
    # window_id = 0
    # n_ops = 45

    # LOCAL_SCHEDULE_FILES.append(
    #     Path(ROOT_DIR + "/local_realtime_scheduling/InterfaceWithGlobal/local_schedules") / f"M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}W{dfjspt_params.time_window_size}" / "testing" / f"local_schedule_J{n_jobs}I{instance_id}_{window_id}_ops{n_ops}.pkl"
    # )

    # # For demonstration, create a simple test case
    # config = {
    #     "n_machines": dfjspt_params.n_machines,
    #     "n_transbots": dfjspt_params.n_transbots,
    #     "factory_instance_seed": 42,
    # }
    
    # framework = FairComparisonFramework(config)
    # print(f"\nFramework initialized with {config['n_machines']} machines and {config['n_transbots']} transbots")
    # print("Ready for comparison experiments!")

    all_results, analysis = run_comparison_experiment(
        local_schedule_files=LOCAL_SCHEDULE_FILES,
        embodied_checkpoint_dir=EMBODIED_CHECKPOINT_DIR,
        non_embodied_checkpoint_dir=NON_EMBODIED_CHECKPOINT_DIR,
        num_episodes_per_instance=2,
        detailed_log=True
    )




