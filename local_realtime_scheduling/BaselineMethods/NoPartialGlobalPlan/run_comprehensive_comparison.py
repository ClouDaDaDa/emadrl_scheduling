#!/usr/bin/env python3
"""
Comprehensive comparison between EMADRL with Partial Global Plans vs No Partial Global Plans.
This script tests both methods on global instances and compares their performance.
"""

import os
import sys
import pickle
import re
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Any
import argparse
from datetime import datetime
import torch
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils.numpy import convert_to_numpy, softmax
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add the project root to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from System.FactoryInstance import FactoryInstance
from System.SchedulingInstance import SchedulingInstance
from local_realtime_scheduling.Environment.LocalSchedulingMultiAgentEnv_v3_4 import LocalSchedulingMultiAgentEnv
from local_realtime_scheduling.BaselineMethods.NoPartialGlobalPlan.NoPartialGlobalPlanMultiAgentEnv import NoPartialGlobalPlanMultiAgentEnv
from local_realtime_scheduling.BaselineMethods.NoPartialGlobalPlan.generate_no_partial_options import generate_no_partial_reset_options, MaxWindowDict
from local_realtime_scheduling.InterfaceWithGlobal.divide_global_schedule_to_local_from_pkl import LocalSchedule, Local_Job_schedule
from configs import dfjspt_params


def get_local_schedule_files_for_global_instance(n_jobs: int, instance_id: int, testing_dir: str) -> Tuple[List[str], int]:
    """Get all local schedule files for a global instance, sorted by window ID, and calculate total ops"""
    pattern = re.compile(f"local_schedule_J{n_jobs}I{instance_id}_.*\.pkl$")
    
    # Get all matching files
    matching_files = [f for f in os.listdir(testing_dir) if pattern.match(f)]
    
    # Sort by window ID
    sorted_files = sorted(
        matching_files,
        key=lambda x: int(re.search(r'_(\d+)_ops', x).group(1))
    )
    
    # Calculate total number of operations across all windows
    total_ops = 0
    ops_pattern = re.compile(r'_ops(\d+)\.pkl$')
    for filename in sorted_files:
        ops_match = ops_pattern.search(filename)
        if ops_match:
            total_ops += int(ops_match.group(1))
    
    return sorted_files, total_ops


class ComprehensiveComparison:
    def __init__(
        self, 
        n_machines: int = 36,
        n_transbots: int = 20,
        factory_seed: int = 42,
        max_instances: int = 1e8,
        num_episodes: int = 1,
        num_workers: int = 1,
        results_dir: str = None,
        partial_global_plan_checkpoint_dir: str = None,
        no_partial_global_plan_checkpoint_dir: str = None,
    ):
        """
        Initialize the comprehensive comparison framework.
        
        Args:
            n_machines: Number of machines
            n_transbots: Number of transbots
            factory_seed: Seed for factory instance generation
            results_dir: Directory to save results
        """
        self.n_machines = n_machines
        self.n_transbots = n_transbots
        self.factory_seed = factory_seed
        self.num_episodes = max(1, int(num_episodes))
        self.num_workers = max(1, int(num_workers))
        self.results_dir = results_dir or os.path.dirname(os.path.abspath(__file__)) + f"/compare_no_partial_results_M{n_machines}T{n_transbots}_{datetime.now().strftime('%Y%m%d_%H%M%S')}" 
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)

        self.test_n_jobs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]
        # self.test_n_jobs = [200]
        self.test_instance_id = [100, 101, 102, 103, 104]
        # self.test_instance_id = [100, 101, 102]
        self.total_test_instances = min(max_instances, len(self.test_n_jobs) * len(self.test_instance_id))
        
        # Results storage
        self.results = {
            'partial_global_plan': [],
            'no_partial_global_plan': []
        }

        self.partial_global_plan_policies = self._get_madrl_policies(partial_global_plan_checkpoint_dir)
        self.no_partial_global_plan_policies = self._get_madrl_policies(no_partial_global_plan_checkpoint_dir)

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
                if policy_type == "partial_global_plan":
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
                        rl_module_out = self.partial_global_plan_policies[0].forward_inference(input_dict)
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
                        rl_module_out = self.partial_global_plan_policies[1].forward_inference(input_dict)
                elif policy_type == "no_partial_global_plan":
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
                        rl_module_out = self.partial_global_plan_policies[0].forward_inference(input_dict)
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
                        rl_module_out = self.partial_global_plan_policies[1].forward_inference(input_dict)
                else:
                    raise ValueError(f"Invalid policy type: {policy_type}")
            except Exception as e:
                print(f"Inference failed for agent {agent_id} (policy={policy_type}): {e}")
                if "observation" in obs:
                    try:
                        keys = list(obs["observation"].keys())
                        print(f"Obs keys: {keys}")
                        for k in keys:
                            arr = obs["observation"].get(k)
                            if hasattr(arr, 'shape'):
                                print(f"{agent_id}.{k}.shape={arr.shape}")
                    except Exception:
                        pass
                am = obs.get("action_mask")
                if am is not None:
                    try:
                        print(f"action_mask.len={len(am)} enabled={sum(1 for x in am if x==1)}")
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
    
    def test_partial_global_plan_method(
        self, 
        n_job: int,
        instance_id: int,
        scheduling_instance: SchedulingInstance, 
        episode_id: str,
    ) -> Dict[str, Any]:
        """
        Test the EMADRL method with Partial Global Plans.
        
        Args:
            n_job: Number of jobs for this instance
            instance_id: Instance ID
            scheduling_instance: The SchedulingInstance for this instance
            
        Returns:
            Dictionary containing performance metrics
        """
        print("Testing Partial Global Plan method...")
        
        total_makespan = 0
        window_results = []
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # For the EMADRL method with Partial Global Plans, we use the local schedules from "local_realtime_scheduling/InterfaceWithLocal/local_schedules"
        local_schedules_dir = os.path.dirname(os.path.dirname(current_dir)) \
                          + f"/InterfaceWithGlobal/local_schedules" \
                          + f"/M{self.n_machines}T{self.n_transbots}W{dfjspt_params.time_window_size}/testing"

        local_files, total_ops = get_local_schedule_files_for_global_instance(
            n_job, instance_id, local_schedules_dir
        )
        if not local_files:
            raise ValueError(f"No local schedule files found for M{self.n_machines}T{self.n_transbots}W{dfjspt_params.time_window_size}/J{n_job}I{instance_id}")
        num_windows = len(local_files)
        
        partial_global_plan_env = LocalSchedulingMultiAgentEnv(
            config={
                "n_machines": self.n_machines,
                "n_transbots": self.n_transbots,
                "factory_instance_seed": self.factory_seed,
                "enable_dynamic_agent_filtering": False,
            }
        )
 
        for window_idx, local_filename in enumerate(local_files):

            local_file_path = os.path.join(local_schedules_dir, local_filename)
            # Deep copy the local schedule to avoid race conditions
            with open(local_file_path, "rb") as file:
                original_local_schedule = pickle.load(file)
            local_schedule = copy.deepcopy(original_local_schedule)

            print(f"  Processing window {window_idx + 1}/{num_windows}")
            
            if window_idx == 0:
                # Reset environment with local schedule
                reset_options = {
                    "factory_instance": partial_global_plan_env.factory_instance,
                    "scheduling_instance": copy.deepcopy(scheduling_instance),
                    "local_schedule": local_schedule,
                    "current_window": window_idx,
                    "start_t_for_curr_time_window": 0,
                    "instance_n_jobs": n_job,
                    "current_instance_id": instance_id
                }
            else:
                # Restore FactoryInstance and SchedulingInstance from snapshot
                # Load snapshot from previous window - use unique episode-based path to avoid conflicts
                snapshot_dir = current_dir + "/instance_snapshots/partial" + \
                    f"/M{self.n_machines}T{self.n_transbots}W{dfjspt_params.time_window_size}" \
                    + f"/snapshot_J{n_job}I{instance_id}_{window_idx - 1}" \
                    + f"_ep{episode_id}.pkl"
                
                try:
                    with open(snapshot_dir, "rb") as file:
                        prev_snapshot = pickle.load(file)
                except FileNotFoundError:
                    raise FileNotFoundError(f"Snapshot {snapshot_dir} not found")
                else:
                    reset_options = {
                        "factory_instance": prev_snapshot["factory_instance"],
                        "scheduling_instance": prev_snapshot["scheduling_instance"],
                        "local_schedule": local_schedule,  # Use the deep-copied local_schedule
                        "current_window": window_idx,
                        "instance_n_jobs": n_job,
                        "current_instance_id": instance_id,
                        "start_t_for_curr_time_window": prev_snapshot["start_t_for_curr_time_window"],
                        "local_result_file": None,
                    }
            
            # Retry logic for truncated episodes
            max_retries = 3
            retry_count = 0
            window_success = False
            success = False
            delta_makespan = 0.0
            execution_time = 0.0
            
            while retry_count < max_retries and not window_success:
                try:
                    start_exec_time = time.time()
                    observations, _ = partial_global_plan_env.reset(options=copy.deepcopy(reset_options))
                    
                    decision_count = 0
                    terminated = {"__all__": False}
                    truncated = {"__all__": False}
                    
                    while not (terminated["__all__"] or truncated["__all__"]):  
                        # Get MADRL actions from policy with partial global plan
                        actions = self._get_madrl_actions(policy_type="partial_global_plan", observations=observations)
                        # Step the environment
                        observations, rewards, terminated, truncated, infos = partial_global_plan_env.step(actions)
                        decision_count += 1

                        # Save intermediate snapshots with unique names to avoid conflicts
                        if (partial_global_plan_env.remaining_operations == 0 
                            and not partial_global_plan_env.has_saved_instance_snapshot):
                            
                            snapshot_dir = current_dir + "/instance_snapshots/partial" + \
                                f"/M{self.n_machines}T{self.n_transbots}W{dfjspt_params.time_window_size}" \
                                + f"/snapshot_J{n_job}I{instance_id}" \
                                + f"_{window_idx}"
                            
                            partial_global_plan_env.has_saved_instance_snapshot = True
                            for job_id in partial_global_plan_env.local_schedule.jobs:
                                this_job = partial_global_plan_env.scheduling_instance.jobs[job_id]
                                this_job.p_ops_for_cur_tw = []
                            
                            # Save with unique episode identifier to avoid conflicts
                            partial_global_plan_env._save_instance_snapshot(
                                final=False,
                                episode_id=f"{episode_id}",
                                instance_snapshot_dir=snapshot_dir,
                            )
                     
                    execution_time = time.time() - start_exec_time
                    # Extract results - for local instances, we use delta makespan
                    if partial_global_plan_env.local_result.actual_local_makespan is not None:
                        delta_makespan = partial_global_plan_env.local_result.actual_local_makespan - partial_global_plan_env.local_result.time_window_start
                        total_makespan = partial_global_plan_env.local_result.actual_local_makespan
                        window_success = True
                        success = terminated["__all__"]
                    else:
                        raise ValueError("The episode is truncated!")
                        
                except ValueError as e:
                    if "truncated" in str(e).lower():
                        retry_count += 1
                        print(f"    Window {window_idx + 1} truncated, retry {retry_count}/{max_retries}")
                        if retry_count < max_retries:
                            continue
                        else:
                            print(f"    Window {window_idx + 1} failed after {max_retries} retries, using truncated result")
                            # Use truncated result as fallback
                            delta_makespan = partial_global_plan_env.current_time_after_step - partial_global_plan_env.local_result.time_window_start
                            total_makespan = partial_global_plan_env.current_time_after_step
                            window_success = True
                            success = False
                    else:
                        raise e
                except Exception as e:
                    retry_count += 1
                    print(f"    Window {window_idx + 1} error: {e}, retry {retry_count}/{max_retries}")
                    if retry_count >= max_retries:
                        raise e

            if success and window_idx == num_windows - 1:
                # Save final snapshot for this instance with unique episode identifier to avoid conflicts
                snapshot_dir = current_dir + "/instance_snapshots/partial" + \
                    f"/M{self.n_machines}T{self.n_transbots}W{dfjspt_params.time_window_size}" \
                    + f"/snapshot_J{n_job}I{instance_id}" \
                    + f"_{window_idx}"
                    
                partial_global_plan_env._save_instance_snapshot(
                    final=True,
                    episode_id=f"{episode_id}",
                    instance_snapshot_dir=snapshot_dir,
                )
            
            window_results.append({
                'window_idx': partial_global_plan_env.current_window,
                'makespan': delta_makespan,
                "execution_time": execution_time,
                "success": success,
            })
        
        return {
            'method': 'partial_global_plan',
            'total_makespan': total_makespan,
            'num_windows': len(window_results),
            'window_results': window_results
        }
    
    def test_no_partial_global_plan_method(
        self, 
        n_job: int,
        instance_id: int,
        scheduling_instance: SchedulingInstance,
        checkpoint_dir: str = None,
        episode_id: str = "0",
    ) -> Dict[str, Any]:
        """
        Test the EMADRL method without Partial Global Plans (rough partitioning).
        
        Args:
            n_job: Number of jobs for this instance
            instance_id: Instance ID
            scheduling_instance: The SchedulingInstance for this instance
            checkpoint_dir: Path to trained EMADRL policy
            
        Returns:
            Dictionary containing performance metrics
        """
        print("Testing No Partial Global Plan method...")
        
        max_window = MaxWindowDict[f"M{self.n_machines}T{self.n_transbots}"][n_job]
        current_dir = os.path.dirname(os.path.abspath(__file__))

        no_partial_global_plan_env = NoPartialGlobalPlanMultiAgentEnv(
            config={
                "n_machines": self.n_machines,
                "n_transbots": self.n_transbots,
                "factory_instance_seed": self.factory_seed,
                "enable_dynamic_agent_filtering": False,
            }
        )

        total_makespan = 0
        window_results = []
        
        # Test with max_window segments (rough partitioning)
        for window_idx in range(max_window):
            print(f"  Processing window {window_idx + 1}/{max_window}")
            
            if window_idx == 0:
                no_partial_options = {
                    "factory_instance": no_partial_global_plan_env.factory_instance,
                    "scheduling_instance": copy.deepcopy(scheduling_instance),
                    "instance_n_jobs": n_job,
                    "current_instance_id": instance_id,
                    "max_window": max_window,
                    "current_window": window_idx,
                    "start_t_for_curr_window": 0
                }
            else:
                # Restore FactoryInstance and SchedulingInstance from snapshot
                # Load snapshot from previous window - use unique episode-based path to avoid conflicts
                snapshot_dir = current_dir + "/instance_snapshots/no_partial" + \
                    f"/M{self.n_machines}T{self.n_transbots}W{dfjspt_params.time_window_size}" \
                    + f"/snapshot_J{n_job}I{instance_id}_{window_idx - 1}" \
                    + f"_ep{episode_id}.pkl"
                
                try:
                    with open(snapshot_dir, "rb") as file:
                        prev_snapshot = pickle.load(file)
                except FileNotFoundError:
                    raise FileNotFoundError(f"Snapshot {snapshot_dir} not found")
                else:
                    no_partial_options = {
                        "factory_instance": prev_snapshot["factory_instance"],
                        "scheduling_instance": prev_snapshot["scheduling_instance"],
                        "instance_n_jobs": n_job,
                        "current_instance_id": instance_id,
                        "max_window": max_window,
                        "current_window": window_idx,
                        "start_t_for_curr_window": prev_snapshot["start_t_for_curr_time_window"],
                    }
            
            # Retry logic for truncated episodes
            max_retries = 3
            retry_count = 0
            window_success = False
            delta_makespan = 0.0
            execution_time = 0.0
            success = False
            
            while retry_count < max_retries and not window_success:
                try:
                    start_exec_time = time.time()
                    observations, _ = no_partial_global_plan_env.reset(options=copy.deepcopy(no_partial_options))
                    
                    decision_count = 0
                    terminated = {"__all__": False}
                    truncated = {"__all__": False}
                    
                    while not (terminated["__all__"] or truncated["__all__"]):
                        # Get MADRL actions from policy without partial global plan
                        actions = self._get_madrl_actions(policy_type="no_partial_global_plan", observations=observations)
                        # Step the environment
                        observations, rewards, terminated, truncated, infos = no_partial_global_plan_env.step(actions)
                        decision_count += 1

                        # Save intermediate snapshots with unique names to avoid conflicts
                        if (no_partial_global_plan_env.remaining_operations == 0 
                            and not no_partial_global_plan_env.has_saved_instance_snapshot):
                            
                            snapshot_dir = current_dir + "/instance_snapshots/no_partial" + \
                                f"/M{self.n_machines}T{self.n_transbots}W{dfjspt_params.time_window_size}" \
                                + f"/snapshot_J{n_job}I{instance_id}" \
                                + f"_{window_idx}"
                            
                            no_partial_global_plan_env.has_saved_instance_snapshot = True
                            for job_id in no_partial_global_plan_env.local_schedule.jobs:
                                this_job = no_partial_global_plan_env.scheduling_instance.jobs[job_id]
                                this_job.p_ops_for_cur_tw = []
                            
                            # Save with unique episode identifier to avoid conflicts
                            no_partial_global_plan_env._save_instance_snapshot(
                                final=False,
                                episode_id=f"{episode_id}",
                                instance_snapshot_dir=snapshot_dir,
                            )
                    
                    execution_time = time.time() - start_exec_time
                    # Extract results - for local instances, we use delta makespan
                    if no_partial_global_plan_env.local_result.actual_local_makespan is not None:
                        delta_makespan = no_partial_global_plan_env.local_result.actual_local_makespan - no_partial_global_plan_env.local_result.time_window_start
                        total_makespan = no_partial_global_plan_env.local_result.actual_local_makespan
                        window_success = True
                        success = terminated["__all__"]
                    else:
                        raise ValueError("The episode is truncated!")
                        
                except ValueError as e:
                    if "truncated" in str(e).lower():
                        retry_count += 1
                        print(f"    Window {window_idx + 1} truncated, retry {retry_count}/{max_retries}")
                        if retry_count < max_retries:
                            continue
                        else:
                            print(f"    Window {window_idx + 1} failed after {max_retries} retries, using truncated result")
                            # Use truncated result as fallback
                            delta_makespan = no_partial_global_plan_env.current_time_after_step - no_partial_global_plan_env.local_result.time_window_start
                            total_makespan = no_partial_global_plan_env.current_time_after_step
                            window_success = True
                            success = False
                    else:
                        raise e
                except Exception as e:
                    retry_count += 1
                    print(f"    Window {window_idx + 1} error: {e}, retry {retry_count}/{max_retries}")
                    if retry_count >= max_retries:
                        raise e

            if success and window_idx == max_window - 1:
                # Save final snapshot for this instance with unique episode identifier to avoid conflicts
                snapshot_dir = current_dir + "/instance_snapshots/no_partial" + \
                    f"/M{self.n_machines}T{self.n_transbots}W{dfjspt_params.time_window_size}" \
                    + f"/snapshot_J{n_job}I{instance_id}" \
                    + f"_{window_idx}"
                    
                no_partial_global_plan_env._save_instance_snapshot(
                    final=True,
                    episode_id=f"{episode_id}",
                    instance_snapshot_dir=snapshot_dir,
                )
            
            window_results.append({
                'window_idx': no_partial_global_plan_env.current_window,
                'makespan': delta_makespan,
                "execution_time": execution_time,
                "success": success,
            })
        
        return {
            'method': 'no_partial_global_plan',
            'total_makespan': total_makespan,
            'num_windows': max_window,
            'window_results': window_results
        }
    
    def _run_single_episode(self, n_job: int, instance_id: int, episode_index: int) -> Dict[str, Any]:
        """Run both methods for a single (instance, episode)."""
        scheduling_instance = SchedulingInstance(
            seed=52 + instance_id,
            n_jobs=n_job,
            n_machines=self.n_machines,
        )
        ep_id = f"{os.getpid()}_{episode_index}"
        partial_results = self.test_partial_global_plan_method(
            n_job=n_job,
            instance_id=instance_id,
            scheduling_instance=scheduling_instance,
            episode_id=ep_id,
        )
        partial_results['instance_id'] = f"J{n_job}I{instance_id}_ep{episode_index}"
        no_partial_results = self.test_no_partial_global_plan_method(
            n_job=n_job,
            instance_id=instance_id,
            scheduling_instance=scheduling_instance,
            episode_id=ep_id,
        )
        no_partial_results['instance_id'] = f"J{n_job}I{instance_id}_ep{episode_index}"
        return {
            'partial_global_plan': partial_results,
            'no_partial_global_plan': no_partial_results,
        }

    def run_comparison(self) -> Dict[str, Any]:
        """
        Run comprehensive comparison on multiple global instances.
        
        Returns:
            Dictionary containing all comparison results
        """
        print(f"Starting comprehensive comparison on {self.total_test_instances} instances...")
        print(f"Results will be saved to: {self.results_dir}")
        
        all_results = {
            'partial_global_plan': [],
            'no_partial_global_plan': []
        }
        
        # Prepare task list
        tasks: List[Tuple[int,int,int]] = []
        for n_job in self.test_n_jobs:
            for instance_id in self.test_instance_id:
                for ep in range(self.num_episodes):
                    tasks.append((n_job, instance_id, ep))
                    if len(tasks) >= self.total_test_instances * self.num_episodes:
                        break
        print(f"Scheduling {len(tasks)} episodes across {self.num_workers} workers...")

        if self.num_workers <= 1:
            for (n_job, instance_id, ep) in tasks:
                print(f"\n{'='*60}")
                print(f"Processing instance J{n_job}I{instance_id} episode {ep}")
                print(f"{'='*60}")
                try:
                    ep_results = self._run_single_episode(n_job, instance_id, ep)
                    pr = ep_results['partial_global_plan']
                    nr = ep_results['no_partial_global_plan']
                    all_results['partial_global_plan'].append(pr)
                    all_results['no_partial_global_plan'].append(nr)
                    print(f"\nInstance {pr['instance_id']} Results:")
                    print(f"  Partial Global Plan - Makespan: {pr['total_makespan']:.2f}")
                    print(f"  No Partial Global Plan - Makespan: {nr['total_makespan']:.2f}")
                    print(f"  Improvement: {((nr['total_makespan'] - pr['total_makespan']) / nr['total_makespan'] * 100):.2f}%")
                except Exception as e:
                    print(f"Error processing instance J{n_job}I{instance_id} ep{ep}: {e}")
                    continue
        else:
            with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
                future_to_task = {
                    ex.submit(self._run_single_episode, n_job, instance_id, ep): (n_job, instance_id, ep)
                    for (n_job, instance_id, ep) in tasks
                }
                for fut in as_completed(future_to_task):
                    n_job, instance_id, ep = future_to_task[fut]
                    try:
                        ep_results = fut.result()
                        pr = ep_results['partial_global_plan']
                        nr = ep_results['no_partial_global_plan']
                        all_results['partial_global_plan'].append(pr)
                        all_results['no_partial_global_plan'].append(nr)
                        print(f"\nInstance {pr['instance_id']} Results:")
                        print(f"  Partial Global Plan - Makespan: {pr['total_makespan']:.2f}")
                        print(f"  No Partial Global Plan - Makespan: {nr['total_makespan']:.2f}")
                        print(f"  Improvement: {((nr['total_makespan'] - pr['total_makespan']) / nr['total_makespan'] * 100):.2f}%")
                    except Exception as e:
                        print(f"Error (parallel) processing instance J{n_job}I{instance_id} ep{ep}: {e}")
                        continue
        
        # Save results
        self.save_results(all_results)
        
        # Generate comparison report
        self.generate_comparison_report(all_results)
        
        return all_results
    
    def save_results(self, results: Dict[str, Any]):
        """Save results to files."""
        # Save raw results
        with open(os.path.join(self.results_dir, 'raw_results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        
        # Save summary CSV
        summary_data = []
        for method, method_results in results.items():
            for result in method_results:
                summary_data.append({
                    'method': method,
                    'instance_id': result['instance_id'],
                    'total_makespan': result['total_makespan'],
                    'num_windows_segments': result['num_windows'],
                })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(self.results_dir, 'summary.csv'), index=False)
        
        print(f"Results saved to {self.results_dir}")
    
    def generate_comparison_report(self, results: Dict[str, Any]):
        """Generate a comprehensive comparison report."""
        partial_results = results['partial_global_plan']
        no_partial_results = results['no_partial_global_plan']
        
        if not partial_results or not no_partial_results:
            print("No results to compare!")
            return
        
        # Calculate statistics
        partial_makespans = [r['total_makespan'] for r in partial_results]
        no_partial_makespans = [r['total_makespan'] for r in no_partial_results]
        
        # Calculate improvements
        improvements = []
        for i in range(min(len(partial_makespans), len(no_partial_makespans))):
            if no_partial_makespans[i] > 0:
                improvement = (no_partial_makespans[i] - partial_makespans[i]) / no_partial_makespans[i] * 100
                improvements.append(improvement)
        
        # Generate report
        report = f"""
COMPREHENSIVE COMPARISON REPORT
===============================

Configuration:
- Machines: {self.n_machines}
- Transbots: {self.n_transbots}
- Factory Seed: {self.factory_seed}
- Test Instances: {len(partial_results)}

RESULTS SUMMARY:
===============

Partial Global Plan Method:
- Average Makespan: {np.mean(partial_makespans):.2f} ± {np.std(partial_makespans):.2f}
- Min Makespan: {np.min(partial_makespans):.2f}
- Max Makespan: {np.max(partial_makespans):.2f}

No Partial Global Plan Method:
- Average Makespan: {np.mean(no_partial_makespans):.2f} ± {np.std(no_partial_makespans):.2f}
- Min Makespan: {np.min(no_partial_makespans):.2f}
- Max Makespan: {np.max(no_partial_makespans):.2f}

IMPROVEMENT ANALYSIS:
===================
- Average Improvement: {np.mean(improvements):.2f}%
- Median Improvement: {np.median(improvements):.2f}%
- Min Improvement: {np.min(improvements):.2f}%
- Max Improvement: {np.max(improvements):.2f}%
- Instances with Improvement: {sum(1 for imp in improvements if imp > 0)}/{len(improvements)}

CONCLUSION:
===========
The Partial Global Plan method shows {'better' if np.mean(improvements) > 0 else 'worse'} performance
with an average makespan improvement of {np.mean(improvements):.2f}% over the No Partial Global Plan method.
"""
        
        # Save report
        with open(os.path.join(self.results_dir, 'comparison_report.txt'), 'w') as f:
            f.write(report)
        
        print(report)
        
        # Generate plots
        self.generate_plots(partial_makespans, no_partial_makespans, improvements)
    
    def generate_plots(self, partial_makespans, no_partial_makespans, improvements):
        """Generate comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Makespan comparison
        axes[0, 0].scatter(range(len(partial_makespans)), partial_makespans, 
                          label='Partial Global Plan', alpha=0.7, color='blue')
        axes[0, 0].scatter(range(len(no_partial_makespans)), no_partial_makespans, 
                          label='No Partial Global Plan', alpha=0.7, color='red')
        axes[0, 0].set_xlabel('Instance Index')
        axes[0, 0].set_ylabel('Makespan')
        axes[0, 0].set_title('Makespan Comparison by Instance')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Improvement histogram
        axes[0, 1].hist(improvements, bins=20, alpha=0.7, color='green')
        axes[0, 1].axvline(np.mean(improvements), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(improvements):.2f}%')
        axes[0, 1].set_xlabel('Improvement (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Makespan Improvements')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box plot comparison
        data_to_plot = [partial_makespans, no_partial_makespans]
        axes[1, 0].boxplot(data_to_plot, labels=['Partial Global Plan', 'No Partial Global Plan'])
        axes[1, 0].set_ylabel('Makespan')
        axes[1, 0].set_title('Makespan Distribution Comparison')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Improvement over instances
        axes[1, 1].plot(range(len(improvements)), improvements, 'o-', alpha=0.7, color='purple')
        axes[1, 1].axhline(0, color='black', linestyle='-', alpha=0.5)
        axes[1, 1].set_xlabel('Instance Index')
        axes[1, 1].set_ylabel('Improvement (%)')
        axes[1, 1].set_title('Improvement Trend Across Instances')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'comparison_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {os.path.join(self.results_dir, 'comparison_plots.png')}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive comparison between EMADRL methods')
    parser.add_argument('--machines', type=int, default=10, help='Number of machines')
    parser.add_argument('--transbots', type=int, default=10, help='Number of transbots')
    parser.add_argument('--factory_seed', type=int, default=42, help='Factory instance seed')
    parser.add_argument('--max_instances', type=int, default=1000,
                       help='Maximum number of instances to test')
    parser.add_argument('--num_episodes', type=int, default=5,
                       help='Repeat episodes per global instance')
    parser.add_argument('--num_workers', type=int, default=5,
                       help='Number of parallel workers')
    
    args = parser.parse_args()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if args.machines == 10 and args.transbots == 10:
        NO_PARTIAL_GP_CHECKPOINT_DIR = current_dir + "/ray_results/M10T10W300_NoPartial/PPO_NoPartialGlobalPlanMultiAgentEnv_3abf7_00000_0_2025-09-09_13-38-35/checkpoint_000012"
        PARTIAL_GP_CHECKPOINT_DIR = os.path.dirname(os.path.dirname(current_dir)) + "/Agents/ray_results/M10T10W300/PPO_LocalSchedulingMultiAgentEnv_59efc_00000_0_2025-08-31_15-23-11/checkpoint_000046"
    elif args.machines == 36 and args.transbots == 20:
        NO_PARTIAL_GP_CHECKPOINT_DIR = current_dir + "/ray_results/M36T20W300_NoPartial/PPO_NoPartialGlobalPlanMultiAgentEnv_038b3_00000_0_2025-09-06_17-29-40/checkpoint_000023"
        PARTIAL_GP_CHECKPOINT_DIR = os.path.dirname(os.path.dirname(current_dir)) + "/Agents/ray_results/M36T20W300/PPO_LocalSchedulingMultiAgentEnv_6589e_00000_0_2025-08-28_21-53-37/checkpoint_000049"
    else:
        raise ValueError(f"Invalid configuration: M{args.machines}T{args.transbots}")

    # Run comparison
    comparison = ComprehensiveComparison(
        n_machines=args.machines,
        n_transbots=args.transbots,
        factory_seed=args.factory_seed,
        max_instances=args.max_instances,
        num_episodes=args.num_episodes,
        num_workers=args.num_workers,
        partial_global_plan_checkpoint_dir=PARTIAL_GP_CHECKPOINT_DIR,
        no_partial_global_plan_checkpoint_dir=NO_PARTIAL_GP_CHECKPOINT_DIR
    )
    
    results = comparison.run_comparison()
    
    print(f"\nComparison completed! Results saved to: {comparison.results_dir}")


if __name__ == "__main__":
    main()
