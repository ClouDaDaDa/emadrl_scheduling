import os
import re
import time
import pickle
import numpy as np
import pandas as pd  # Add pandas import
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import ray
from concurrent.futures import as_completed
import threading
from dataclasses import dataclass
import copy

from configs import dfjspt_params
from System.SchedulingInstance import SchedulingInstance
from local_realtime_scheduling.result_collector import ResultCollector, create_timestamped_output_dir
from local_realtime_scheduling.result_analyzer import ResultAnalyzer
from local_realtime_scheduling.Environment.LocalSchedulingMultiAgentEnv_v3_4 import LocalSchedulingMultiAgentEnv
from local_realtime_scheduling.BaselineMethods.InitialSchedule.InitialScheduleEnv_v3_3 import InitialScheduleEnv
from local_realtime_scheduling.BaselineMethods.DispatchingRules.machine_agent_heuristics import machine_heuristic
from local_realtime_scheduling.BaselineMethods.DispatchingRules.transbot_agent_heuristics import transbot_heuristic
from local_realtime_scheduling.test_heuristic_combinations import TOP_HEURISTIC_COMBINATIONS
from local_realtime_scheduling.InterfaceWithGlobal.divide_global_schedule_to_local_from_pkl import LocalSchedule, Local_Job_schedule


@dataclass
class TestTask:
    """Represents a single test task to be executed"""
    n_jobs: int
    instance_id: int
    policy_type: str
    episode_id: int
    heuristic_combo: Optional[Dict[str, Any]] = None
    task_id: Optional[str] = None
    
    def __post_init__(self):
        if self.task_id is None:
            self.task_id = f"J{self.n_jobs}I{self.instance_id}_{self.policy_type}_ep{self.episode_id}"


@dataclass 
class TestResult:
    """Represents the result of a single test task"""
    task: TestTask
    makespan: float
    execution_time: float
    total_reward: float
    initial_estimated_makespan: float
    total_ops: int
    success: bool = True
    error_message: Optional[str] = None


class ParallelResultCollector:
    """Thread-safe result collector for parallel execution with CSV-based tracking"""
    
    def __init__(self, output_dir: str, is_global_analysis: bool = True):
        self.output_dir = Path(output_dir)
        self.is_global_analysis = is_global_analysis
        self.results = {}
        self.lock = threading.Lock()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV file path for episode results
        self.csv_dir = self.output_dir / "csv"
        self.csv_dir.mkdir(exist_ok=True)
        self.episode_results_csv = self.csv_dir / "episode_results.csv"
        
        # Load existing results if resuming
        self._load_existing_results()
        
    def _load_existing_results(self):
        """Load existing results from episode_results.csv"""
        if self.episode_results_csv.exists():
            try:
                df = pd.read_csv(self.episode_results_csv)
                print(f"✓ Loaded {len(df)} existing episode results from CSV")
                
                # Convert CSV data back to results structure
                for _, row in df.iterrows():
                    # Parse instance info from instance_filename (e.g., "global_J150I100")
                    instance_filename = row['instance_filename']
                    if instance_filename.startswith('global_J'):
                        # Extract J{n_jobs}I{instance_id}
                        match = re.match(r'global_J(\d+)I(\d+)', instance_filename)
                        if match:
                            n_jobs = int(match.group(1))
                            instance_id = int(match.group(2))
                            instance_key = f"J{n_jobs}I{instance_id}"
                            
                            if instance_key not in self.results:
                                self.results[instance_key] = defaultdict(list)
                            
                            # Add episode data
                            episode_data = {
                                'episode_id': row['episode_id'],
                                'makespan': row['makespan'],
                                'execution_time': row['execution_time'],
                                'total_reward': row['total_reward'],
                                'initial_estimated_makespan': row['estimated_makespan'],
                                'total_ops': row['n_ops'],
                                'success': True,  # If it's in CSV, it was successful
                                'error_message': None,
                                'timestamp': row.get('timestamp', datetime.now().isoformat())
                            }
                            
                            self.results[instance_key][row['policy_name']].append(episode_data)
                            
            except Exception as e:
                print(f"⚠ Could not load episode results from CSV: {e}")
    
    def _save_episode_to_csv(self, result: TestResult):
        """Save a single episode result to CSV immediately"""
        try:
            # Prepare CSV row data
            csv_data = {
                'episode_id': result.task.episode_id,
                'policy_name': result.task.policy_type,
                'instance_filename': f"global_J{result.task.n_jobs}I{result.task.instance_id}",
                'makespan': result.makespan,
                'delta_makespan': result.makespan - result.initial_estimated_makespan if result.initial_estimated_makespan > 0 else 0,
                'estimated_makespan': result.initial_estimated_makespan,
                'delta_estimated_makespan': 0,  # Usually 0 for global instances
                'total_reward': result.total_reward,
                'execution_time': result.execution_time,
                'execution_time_per_step': result.execution_time / result.total_ops if result.total_ops > 0 else 0,
                'is_truncated': False,  # Global instances are not truncated
                'timestamp': datetime.now().isoformat(),
                'n_jobs': result.task.n_jobs,
                'instance_id': result.task.instance_id,
                'window_id': '',  # Global instances don't have window_id
                'n_ops': result.total_ops,
                'is_global_instance': True
            }
            
            # Convert to DataFrame
            new_row_df = pd.DataFrame([csv_data])
            
            # Append to existing CSV or create new one
            if self.episode_results_csv.exists():
                # Append to existing CSV
                new_row_df.to_csv(self.episode_results_csv, mode='a', header=False, index=False)
            else:
                # Create new CSV with header
                new_row_df.to_csv(self.episode_results_csv, index=False)
                
        except Exception as e:
            print(f"⚠ Failed to save episode to CSV: {e}")
        
    def add_result(self, result: TestResult):
        """Thread-safe method to add a test result with immediate CSV saving"""
        with self.lock:
            instance_key = f"J{result.task.n_jobs}I{result.task.instance_id}"
            if instance_key not in self.results:
                self.results[instance_key] = defaultdict(list)
            
            self.results[instance_key][result.task.policy_type].append({
                'episode_id': result.task.episode_id,
                'makespan': result.makespan,
                'execution_time': result.execution_time,
                'total_reward': result.total_reward,
                'initial_estimated_makespan': result.initial_estimated_makespan,
                'total_ops': result.total_ops,
                'success': result.success,
                'error_message': result.error_message,
                'timestamp': datetime.now().isoformat()
            })
            
            # Save to CSV immediately (only if successful)
            if result.success:
                self._save_episode_to_csv(result)
    
    def get_completed_tasks_from_csv(self) -> set:
        """Get set of completed task IDs from CSV data"""
        completed_tasks = set()
        
        if self.episode_results_csv.exists():
            try:
                df = pd.read_csv(self.episode_results_csv)
                for _, row in df.iterrows():
                    # Reconstruct task_id from CSV data
                    n_jobs = row['n_jobs']
                    instance_id = row['instance_id']
                    policy_name = row['policy_name']
                    episode_id = row['episode_id']
                    
                    task_id = f"J{n_jobs}I{instance_id}_{policy_name}_ep{episode_id - 1}"
                    completed_tasks.add(task_id)
                    
            except Exception as e:
                print(f"⚠ Error reading completed tasks from CSV: {e}")
        
        return completed_tasks
    
    def is_task_completed(self, task: TestTask) -> bool:
        """Check if a specific task has already been completed based on CSV data"""
        completed_tasks = self.get_completed_tasks_from_csv()
        return task.task_id in completed_tasks
    
    def get_missing_tasks(self, all_tasks: List[TestTask]) -> List[TestTask]:
        """Get list of tasks that haven't been completed yet based on CSV data"""
        completed_tasks = self.get_completed_tasks_from_csv()
        return [task for task in all_tasks if task.task_id not in completed_tasks]
    
    def get_completion_status(self, all_tasks: List[TestTask]) -> Dict[str, Any]:
        """Get detailed completion status based on CSV data"""
        completed_tasks = self.get_completed_tasks_from_csv()
        completed_count = len([task for task in all_tasks if task.task_id in completed_tasks])
        total_count = len(all_tasks)
        
        # Group by categories for detailed breakdown
        by_job_size = defaultdict(lambda: {'completed': 0, 'total': 0})
        by_policy = defaultdict(lambda: {'completed': 0, 'total': 0})
        by_instance = defaultdict(lambda: {'completed': 0, 'total': 0})
        
        for task in all_tasks:
            is_completed = task.task_id in completed_tasks
            
            by_job_size[task.n_jobs]['total'] += 1
            by_policy[task.policy_type]['total'] += 1
            by_instance[f"J{task.n_jobs}I{task.instance_id}"]['total'] += 1
            
            if is_completed:
                by_job_size[task.n_jobs]['completed'] += 1
                by_policy[task.policy_type]['completed'] += 1
                by_instance[f"J{task.n_jobs}I{task.instance_id}"]['completed'] += 1
        
        return {
            'overall': {
                'completed': completed_count,
                'total': total_count,
                'completion_rate': completed_count / total_count if total_count > 0 else 0.0
            },
            'by_job_size': dict(by_job_size),
            'by_policy': dict(by_policy),
            'by_instance': dict(by_instance)
        }
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of all collected results"""
        with self.lock:
            total_tests = sum(
                sum(len(policy_results) for policy_results in instance_results.values())
                for instance_results in self.results.values()
            )
            successful_tests = sum(
                sum(sum(1 for episode in policy_results if episode['success'])
                    for policy_results in instance_results.values())
                for instance_results in self.results.values()
            )
            
            return {
                'total_instances': len(self.results),
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'success_rate': successful_tests / total_tests if total_tests > 0 else 0.0
            }


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


@ray.remote(num_cpus=1)
def run_single_episode_remote(
    task: TestTask,
    testing_dir: str,
    checkpoint_dir: Optional[str] = None,
    save_snapshots: bool = True,
    detailed_log: bool = True
) -> TestResult:
    """
    Ray remote function to run a single episode of a policy on a global instance
    
    This function is designed to be completely independent and stateless to enable
    safe parallel execution across multiple processes.
    """
    
    try:
        # Get all local schedule files for this global instance
        local_files, total_ops = get_local_schedule_files_for_global_instance(
            task.n_jobs, task.instance_id, testing_dir
        )
        if not local_files:
            return TestResult(
                task=task,
                makespan=0.0,
                execution_time=0.0,
                total_reward=0.0,
                initial_estimated_makespan=0.0,
                total_ops=0,
                success=False,
                error_message=f"No local schedule files found for J{task.n_jobs}I{task.instance_id}"
            )
        
        num_windows = len(local_files)
        
        # Prepare MADRL modules if needed
        machine_rl_module = None
        transbot_rl_module = None
        
        if task.policy_type == 'madrl' and checkpoint_dir:
            import torch
            from ray.rllib.core.rl_module import RLModule
            from ray.rllib.utils.numpy import convert_to_numpy, softmax
            
            machine_rl_module_checkpoint_dir = Path(checkpoint_dir) / "learner_group" / "learner" / "rl_module" / "p_machine"
            transbot_rl_module_checkpoint_dir = Path(checkpoint_dir) / "learner_group" / "learner" / "rl_module" / "p_transbot"
            
            machine_rl_module = RLModule.from_checkpoint(machine_rl_module_checkpoint_dir)
            transbot_rl_module = RLModule.from_checkpoint(transbot_rl_module_checkpoint_dir)
        
        # Environment configuration
        env_config = {
            "n_machines": dfjspt_params.n_machines,
            "n_transbots": dfjspt_params.n_transbots,
            "factory_instance_seed": dfjspt_params.factory_instance_seed,
            "enable_dynamic_agent_filtering": getattr(dfjspt_params, 'enable_dynamic_agent_filtering', False),
        }
        
        episode_start_time = time.time()
        episode_total_reward = 0.0
        initial_estimated_makespan = None
        
        # Create fresh environment for this episode
        if task.policy_type == 'initial':
            scheduling_env = InitialScheduleEnv(env_config)
        else:
            scheduling_env = LocalSchedulingMultiAgentEnv(env_config)
        
        # Process each time window in sequence
        for window_idx in range(num_windows):
            local_file_path = os.path.join(testing_dir, local_files[window_idx])
            
            # CRITICAL FIX: Deep copy the local schedule to avoid race conditions
            with open(local_file_path, "rb") as file:
                original_local_schedule = pickle.load(file)
            
            # Create a deep copy for this specific episode to ensure complete state isolation
            local_schedule = copy.deepcopy(original_local_schedule)
            
            # # ADDITIONAL FIX: Reset all job states to ensure clean slate for each episode
            # if hasattr(local_schedule, 'jobs') and local_schedule.jobs:
            #     for job_id, job in local_schedule.jobs.items():
            #         # Reset job to initial state to avoid state conflicts
            #         job.reset_job()
            #         # Set the job's operations for current time window properly
            #         if hasattr(job, 'p_ops_for_cur_tw') and len(job.p_ops_for_cur_tw) > 0:
            #             # Ensure current_processing_operation matches the expected value
            #             job.current_processing_operation = job.p_ops_for_cur_tw[0]
            
            if detailed_log:
                print(f"    [{task.task_id}] Processing window {window_idx}: {local_files[window_idx]}")
            
            # Prepare reset options
            if window_idx == 0:
                # First window - initialize fresh
                reset_options = {
                    "factory_instance": scheduling_env.factory_instance,
                    "scheduling_instance": SchedulingInstance(
                        seed=52 + task.instance_id,
                        n_jobs=task.n_jobs,
                        n_machines=scheduling_env.num_machines,
                    ),
                    "local_schedule": local_schedule,
                    "current_window": 0,
                    "instance_n_jobs": task.n_jobs,
                    "current_instance_id": task.instance_id,
                    "start_t_for_curr_time_window": 0,
                    "local_result_file": None,
                }
            else:
                # Load snapshot from previous window - use unique episode-based path to avoid conflicts
                snapshot_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + \
                    f"/Environment/instance_snapshots" + \
                    f"/M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}W{dfjspt_params.time_window_size}" \
                    + f"/{task.policy_type.upper()}/snapshot_J{task.n_jobs}I{task.instance_id}" \
                    + f"_{window_idx - 1}_ep{task.episode_id}_pid{os.getpid()}_parallel.pkl"  # Add PID for additional uniqueness
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(snapshot_dir), exist_ok=True)
                
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
                        "instance_n_jobs": task.n_jobs,
                        "current_instance_id": task.instance_id,
                        "start_t_for_curr_time_window": prev_snapshot["start_t_for_curr_time_window"],
                        "local_result_file": None,
                    }
            
            # Reset environment with options
            observations, infos = scheduling_env.reset(options=reset_options)
            
            # Capture initial estimated makespan from first window
            if window_idx == 0:
                initial_estimated_makespan = scheduling_env.initial_estimated_makespan
            
            done = {'__all__': False}
            truncated = {'__all__': False}
            window_rewards = {}
            for agent in scheduling_env.agents:
                window_rewards[agent] = 0.0
            
            # Run the episode for this window
            while (not done['__all__']) and (not truncated['__all__']):
                actions = {}
                
                # Generate actions based on policy type
                if task.policy_type == 'random':
                    # Random policy
                    for agent_id, obs in observations.items():
                        action_mask = obs['action_mask']
                        valid_actions = [i for i, valid in enumerate(action_mask) if valid == 1]
                        if valid_actions:
                            if len(valid_actions) > 1:
                                valid_actions.pop(-1)  # Remove last action if multiple
                            actions[agent_id] = np.random.choice(valid_actions)
                        else:
                            raise Exception(f"No valid actions for agent {agent_id}!")
                
                elif task.policy_type.startswith('heuristic_'):
                    # Specific heuristic policy
                    if task.heuristic_combo is None:
                        raise ValueError(f"heuristic_combo must be provided for heuristic policy {task.policy_type}")
                    
                    for agent_id, obs in observations.items():
                        if agent_id.startswith("machine"):
                            try:
                                actions[agent_id] = machine_heuristic(
                                    obs=obs,
                                    job_rule=task.heuristic_combo['machine_job_rule'],
                                    maint_rule=task.heuristic_combo['machine_maint_rule'],
                                    due_date=initial_estimated_makespan,
                                    periodic_interval=task.heuristic_combo['machine_periodic_interval'],
                                    threshold=task.heuristic_combo['machine_threshold'],
                                )
                            except Exception as e:
                                if detailed_log:
                                    print(f"[{task.task_id}] Machine heuristic error: {e}, falling back to random action")
                                # Fall back to random valid action
                                action_mask = obs['action_mask']
                                valid_actions = [i for i, valid in enumerate(action_mask) if valid == 1]
                                if valid_actions and len(valid_actions) > 1:
                                    valid_actions.pop(-1)  # Remove do-nothing action if possible
                                actions[agent_id] = np.random.choice(valid_actions) if valid_actions else 0
                                
                        elif agent_id.startswith("transbot"):
                            try:
                                actions[agent_id] = transbot_heuristic(
                                    obs=obs,
                                    job_rule=task.heuristic_combo['transbot_job_rule'],
                                    charge_rule=task.heuristic_combo['transbot_charge_rule'],
                                    threshold=task.heuristic_combo['transbot_threshold']
                                )
                            except Exception as e:
                                if detailed_log:
                                    print(f"[{task.task_id}] Transbot heuristic error: {e}, falling back to random action")
                                # Fall back to random valid action
                                action_mask = obs['action_mask']
                                valid_actions = [i for i, valid in enumerate(action_mask) if valid == 1]
                                if valid_actions and len(valid_actions) > 1:
                                    valid_actions.pop(-1)  # Remove do-nothing action if possible
                                actions[agent_id] = np.random.choice(valid_actions) if valid_actions else 0
                
                elif task.policy_type == 'madrl':
                    # MADRL policy
                    import torch
                    from ray.rllib.utils.numpy import convert_to_numpy, softmax
                    
                    for agent_id, obs in observations.items():
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
                            rl_module_out = machine_rl_module.forward_inference(input_dict)
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
                            rl_module_out = transbot_rl_module.forward_inference(input_dict)
                        
                        logits = convert_to_numpy(rl_module_out['action_dist_inputs'])[0]
                        action_prob = softmax(logits)
                        action_prob[action_prob <= 1e-6] = 0.0
                        actions[agent_id] = np.random.choice(len(logits), p=action_prob)
                
                elif task.policy_type == 'initial':
                    # Initial schedule execution
                    for agent_id, obs in observations.items():
                        action_mask = obs['action_mask']
                        valid_actions = [i for i, valid in enumerate(action_mask) if valid == 1]
                        if valid_actions:
                            actions[agent_id] = valid_actions[0]
                        else:
                            raise Exception(f"No valid actions for agent {agent_id}!")
                
                # Step the environment
                observations, rewards, done, truncated, info = scheduling_env.step(actions)
                
                # Save intermediate snapshots with unique names to avoid conflicts
                if (save_snapshots and scheduling_env.remaining_operations == 0 
                    and not scheduling_env.has_saved_instance_snapshot):
                    
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    snapshot_dir = os.path.dirname(current_dir) + \
                        "/Environment/instance_snapshots" + \
                        f"/M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}W{dfjspt_params.time_window_size}" \
                        + f"/{task.policy_type.upper()}/snapshot_J{scheduling_env.instance_n_jobs}I{scheduling_env.current_instance_id}" \
                        + f"_{scheduling_env.current_window}"
                    
                    scheduling_env.has_saved_instance_snapshot = True
                    for job_id in scheduling_env.local_schedule.jobs:
                        this_job = scheduling_env.scheduling_instance.jobs[job_id]
                        this_job.p_ops_for_cur_tw = []
                    
                    # Save with unique episode identifier to avoid conflicts
                    scheduling_env._save_instance_snapshot(
                        final=False,
                        policy_name=task.policy_type.upper(),
                        episode_id=f"{task.episode_id}_pid{os.getpid()}_parallel",  # Add PID for uniqueness
                        instance_snapshot_dir=snapshot_dir,
                    )
                
                # Accumulate rewards
                for agent, reward in rewards.items():
                    window_rewards[agent] += reward
            
            # Save final snapshot for this window with unique episode identifier
            if save_snapshots:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                snapshot_dir = os.path.dirname(current_dir) + \
                    "/Environment/instance_snapshots" + \
                    f"/M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}W{dfjspt_params.time_window_size}" \
                    + f"/{task.policy_type.upper()}/snapshot_J{scheduling_env.instance_n_jobs}I{scheduling_env.current_instance_id}"
                
                scheduling_env._save_instance_snapshot(
                    final=(window_idx == num_windows - 1),
                    policy_name=task.policy_type.upper(),
                    episode_id=f"{task.episode_id}_pid{os.getpid()}_parallel",  # Add PID for uniqueness
                    instance_snapshot_dir=snapshot_dir,
                )
            
            # Add window rewards to episode total
            episode_total_reward += window_rewards.get('machine0', 0.0)
        
        # Calculate final results
        episode_end_time = time.time()
        episode_execution_time = episode_end_time - episode_start_time
        
        if scheduling_env.local_result.actual_local_makespan is not None:
            final_makespan = scheduling_env.local_result.actual_local_makespan
        else:
            final_makespan = scheduling_env.current_time_after_step
        
        if detailed_log:
            print(f"[{task.task_id}] Completed: makespan={final_makespan:.2f}, time={episode_execution_time:.2f}s")
        
        return TestResult(
            task=task,
            makespan=final_makespan,
            execution_time=episode_execution_time,
            total_reward=episode_total_reward,
            initial_estimated_makespan=initial_estimated_makespan or 0.0,
            total_ops=total_ops,
            success=True
        )
        
    except Exception as e:
        import traceback
        error_details = f"{str(e)}\n{traceback.format_exc()}"
        return TestResult(
            task=task,
            makespan=0.0,
            execution_time=0.0,
            total_reward=0.0,
            initial_estimated_makespan=0.0,
            total_ops=0,
            success=False,
            error_message=error_details
        )


def create_test_tasks(
    job_sizes: List[int],
    instance_ids: List[int],
    policies_to_test: List[str],
    num_repeat: int
) -> List[TestTask]:
    """Create all test tasks for parallel execution"""
    
    tasks = []
    
    for n_jobs in job_sizes:
        for instance_id in instance_ids:
            for policy in policies_to_test:
                # Get heuristic combination if it's a heuristic policy
                heuristic_combo = None
                if policy.startswith('heuristic_'):
                    heuristic_combo = next((combo for combo in TOP_HEURISTIC_COMBINATIONS if combo['name'] == policy), None)
                    if heuristic_combo is None:
                        print(f"Warning: Unknown heuristic combination {policy}, skipping...")
                        continue
                
                for episode_id in range(num_repeat):
                    task = TestTask(
                        n_jobs=n_jobs,
                        instance_id=instance_id,
                        policy_type=policy,
                        episode_id=episode_id,
                        heuristic_combo=heuristic_combo
                    )
                    tasks.append(task)
    
    return tasks


def filter_completed_tasks(all_tasks: List[TestTask], output_dir: str) -> Tuple[List[TestTask], int]:
    """Filter out already completed tasks and return pending tasks"""
    # Create a temporary collector just to check completion status
    temp_collector = ParallelResultCollector(output_dir)
    
    pending_tasks = temp_collector.get_missing_tasks(all_tasks)
    completed_count = len(all_tasks) - len(pending_tasks)
    
    return pending_tasks, completed_count


def print_completion_status(all_tasks: List[TestTask], output_dir: str):
    """Print detailed completion status"""
    temp_collector = ParallelResultCollector(output_dir)
    status = temp_collector.get_completion_status(all_tasks)
    
    print(f"\n{'='*60}")
    print(f"COMPLETION STATUS")
    print(f"{'='*60}")
    print(f"Overall: {status['overall']['completed']}/{status['overall']['total']} "
          f"({status['overall']['completion_rate']*100:.1f}%)")
    
    print(f"\nBy Job Size:")
    for job_size, stats in sorted(status['by_job_size'].items()):
        print(f"  J{job_size}: {stats['completed']}/{stats['total']} "
              f"({stats['completed']/stats['total']*100:.1f}%)")
    
    print(f"\nBy Policy:")
    for policy, stats in sorted(status['by_policy'].items()):
        print(f"  {policy}: {stats['completed']}/{stats['total']} "
              f"({stats['completed']/stats['total']*100:.1f}%)")
    
    missing_tasks = temp_collector.get_missing_tasks(all_tasks)
    if missing_tasks:
        print(f"\nMissing Tasks ({len(missing_tasks)}):")
        for task in missing_tasks[:10]:  # Show first 10
            print(f"  {task.task_id}")
        if len(missing_tasks) > 10:
            print(f"  ... and {len(missing_tasks) - 10} more")
    
    print(f"{'='*60}")


def run_parallel_global_instance_comparison(
    output_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    num_repeat: int = 5,
    job_sizes: List[int] = None,
    instance_ids: List[int] = None,
    policies_to_test: List[str] = None,
    max_concurrent_tasks: Optional[int] = None,
    detailed_log: bool = True,
    save_snapshots: bool = True,
    resume_mode: bool = False  # NEW: Enable resume functionality
) -> ResultCollector:
    """
    Run parallel comprehensive comparison across multiple global instances
    
    Args:
        output_dir: Directory to save results
        checkpoint_dir: Path to MADRL checkpoint
        num_repeat: Number of episodes per policy per instance
        job_sizes: List of job sizes to test (e.g., [10, 20, 30])
        instance_ids: List of instance IDs to test
        policies_to_test: List of policies to test
        max_concurrent_tasks: Maximum number of concurrent Ray tasks (None = auto-detect)
        detailed_log: Whether to print detailed logs
        save_snapshots: Whether to save environment snapshots (disabled by default in parallel mode)
        resume_mode: If True, only run missing/failed tests from previous runs
    
    Returns:
        ResultCollector with all results
    """
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        # Use all available CPUs if not specified
        num_cpus = max_concurrent_tasks or os.cpu_count()
        ray.init(num_cpus=num_cpus, local_mode=False)
        print(f"✓ Ray initialized with {num_cpus} CPUs")
    
    # Set defaults
    if job_sizes is None:
        job_sizes = [50, 100, 150, 200]  # Default job sizes for M20T10
    if instance_ids is None:
        instance_ids = list(range(100, 103))  # Default instance IDs
    if policies_to_test is None:
        # Default: test all 5 heuristics, initial schedule, and madrl (if checkpoint provided)
        policies_to_test = [combo['name'] for combo in TOP_HEURISTIC_COMBINATIONS] + ['initial']
        if checkpoint_dir:
            policies_to_test.append('madrl')
    if max_concurrent_tasks is None:
        max_concurrent_tasks = min(os.cpu_count(), 7)  # Use available CPUs, max 7 as mentioned
    
    # Create output directory
    if output_dir is None:
        output_dir = create_timestamped_output_dir(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) +
            "/results_data/global_parallel_comparison"
        )
    
    # Create all test tasks
    all_tasks = create_test_tasks(job_sizes, instance_ids, policies_to_test, num_repeat)
    
    # Handle resume mode
    if resume_mode:
        print(f"\n🔄 RESUME MODE ENABLED")
        print(f"Checking for completed tasks in: {output_dir}")
        
        # Check completion status
        print_completion_status(all_tasks, output_dir)
        
        # Filter out completed tasks
        pending_tasks, completed_count = filter_completed_tasks(all_tasks, output_dir)
        
        if not pending_tasks:
            print(f"✅ All {len(all_tasks)} tasks already completed!")
            # Load existing results and return
            result_collector = ResultCollector(output_dir, is_global_analysis=True)
            parallel_collector = ParallelResultCollector(output_dir, is_global_analysis=True)
            
            # Convert to standard format (existing code)
            for instance_key, instance_results in parallel_collector.results.items():
                for policy_name, episodes in instance_results.items():
                    makespans = []
                    execution_times = []
                    rewards = []
                    estimated_makespan = None
                    total_ops = None
                    is_truncated_list = []
                    
                    for episode in episodes:
                        if episode['success']:
                            makespans.append(episode['makespan'])
                            execution_times.append(episode['execution_time'])
                            rewards.append(episode['total_reward'])
                            is_truncated_list.append(False)
                            
                            if estimated_makespan is None:
                                estimated_makespan = episode['initial_estimated_makespan']
                            if total_ops is None:
                                total_ops = episode['total_ops']
                    
                    if makespans:
                        result_collector.add_policy_results(
                            policy_name=policy_name,
                            instance_filename=f"global_{instance_key}",
                            makespans=makespans,
                            execution_times=execution_times,
                            estimated_makespan=estimated_makespan or 0.0,
                            total_rewards=rewards,
                            is_truncated_list=is_truncated_list,
                            n_ops=total_ops
                        )
            
            return result_collector
        
        print(f"🎯 Resuming with {len(pending_tasks)} pending tasks (skipping {completed_count} completed)")
        tasks_to_run = pending_tasks
        
    else:
        tasks_to_run = all_tasks
    
    total_tasks = len(tasks_to_run)
    
    # Initialize result collectors
    result_collector = ResultCollector(output_dir, is_global_analysis=True)
    parallel_collector = ParallelResultCollector(output_dir, is_global_analysis=True)
    
    # Get testing directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    testing_dir = os.path.dirname(current_dir) + \
        "/InterfaceWithGlobal/local_schedules" + \
        f"/M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}W{dfjspt_params.time_window_size}/testing"
    
    print(f"\n{'='*80}")
    print(f"PARALLEL GLOBAL INSTANCE COMPARISON {'(RESUME MODE)' if resume_mode else ''}")
    print(f"{'='*80}")
    print(f"Job sizes: {job_sizes}")
    print(f"Instance IDs: {instance_ids}")
    print(f"Policies: {', '.join(policies_to_test)}")
    print(f"Episodes per policy: {num_repeat}")
    if resume_mode:
        print(f"Total tasks: {len(all_tasks)} (running {total_tasks} pending)")
    else:
        print(f"Total tasks: {total_tasks}")
    print(f"Max concurrent tasks: {max_concurrent_tasks}")
    print(f"Expected parallelization speedup: ~{min(total_tasks, max_concurrent_tasks)}x")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}")
    
    # Submit all tasks to Ray
    print(f"\n🚀 Submitting {total_tasks} tasks to Ray cluster...")
    start_time = time.time()
    
    # Submit tasks in batches to avoid overwhelming the scheduler
    batch_size = max_concurrent_tasks * 2  # Submit 2x the concurrent limit at a time
    pending_futures = []
    completed_tasks = 0
    failed_tasks = 0
    
    for batch_start in range(0, total_tasks, batch_size):
        batch_end = min(batch_start + batch_size, total_tasks)
        batch_tasks = tasks_to_run[batch_start:batch_end]
        
        # Submit batch
        for task in batch_tasks:
            future = run_single_episode_remote.remote(
                task=task,
                testing_dir=testing_dir,
                checkpoint_dir=checkpoint_dir if task.policy_type == 'madrl' else None,
                save_snapshots=save_snapshots,
                detailed_log=True  # Reduce logging noise in parallel mode
            )
            pending_futures.append(future)
        
        print(f"  Submitted batch {batch_start//batch_size + 1}/{(total_tasks-1)//batch_size + 1}: "
              f"tasks {batch_start}-{batch_end-1}")
        
        # Process completed tasks as they finish
        while len(pending_futures) > max_concurrent_tasks:
            # Wait for at least one task to complete
            ready_futures, pending_futures = ray.wait(pending_futures, num_returns=1, timeout=1.0)
            
            for future in ready_futures:
                try:
                    result = ray.get(future)
                    parallel_collector.add_result(result)  # This now saves immediately
                    
                    if result.success:
                        completed_tasks += 1
                        if detailed_log and completed_tasks % 5 == 0:  # Show progress every 5 tasks
                            elapsed_time = time.time() - start_time
                            avg_time_per_task = elapsed_time / completed_tasks
                            remaining_time = avg_time_per_task * (total_tasks - completed_tasks)
                            print(f"    ✓ Progress: {completed_tasks}/{total_tasks} "
                                  f"({completed_tasks/total_tasks*100:.1f}%) "
                                  f"ETA: {remaining_time/60:.1f}min")
                    else:
                        failed_tasks += 1
                        print(f"    ✗ Task failed: {result.task.task_id} - {result.error_message}")
                        
                except Exception as e:
                    failed_tasks += 1
                    print(f"    ✗ Ray task error: {e}")
    
    # Process remaining tasks
    print(f"\n⏳ Processing remaining {len(pending_futures)} tasks...")
    while pending_futures:
        ready_futures, pending_futures = ray.wait(pending_futures, num_returns=min(10, len(pending_futures)), timeout=5.0)
        
        for future in ready_futures:
            try:
                result = ray.get(future)
                parallel_collector.add_result(result)  # This now saves immediately
                
                if result.success:
                    completed_tasks += 1
                else:
                    failed_tasks += 1
                    print(f"    ✗ Task failed: {result.task.task_id} - {result.error_message}")
                    
            except Exception as e:
                failed_tasks += 1
                print(f"    ✗ Ray task error: {e}")
        
        if ready_futures:
            elapsed_time = time.time() - start_time
            print(f"    ✓ Progress: {completed_tasks}/{total_tasks} "
                  f"({completed_tasks/total_tasks*100:.1f}%) "
                  f"Elapsed: {elapsed_time/60:.1f}min")
    
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"PARALLEL EXECUTION COMPLETED")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Successful tasks: {completed_tasks}/{total_tasks}")
    print(f"Failed tasks: {failed_tasks}")
    print(f"Success rate: {completed_tasks/total_tasks*100:.2f}%")
    if completed_tasks > 0:
        print(f"Average time per task: {total_time/completed_tasks:.2f} seconds")
        expected_serial_time = total_time * max_concurrent_tasks
        print(f"Estimated speedup: ~{expected_serial_time/total_time:.1f}x")
    print(f"{'='*60}")
    
    # Show final completion status
    if resume_mode or failed_tasks > 0:
        print_completion_status(all_tasks, output_dir)
    
    # Convert parallel results to standard ResultCollector format
    print("\n📊 Aggregating results...")
    
    for instance_key, instance_results in parallel_collector.results.items():
        for policy_name, episodes in instance_results.items():
            # Group episodes by their properties
            makespans = []
            execution_times = []
            rewards = []
            estimated_makespan = None
            total_ops = None
            is_truncated_list = []
            
            for episode in episodes:
                if episode['success']:
                    makespans.append(episode['makespan'])
                    execution_times.append(episode['execution_time'])
                    rewards.append(episode['total_reward'])
                    is_truncated_list.append(False)  # Global instances are complete
                    
                    # Use values from first successful episode
                    if estimated_makespan is None:
                        estimated_makespan = episode['initial_estimated_makespan']
                    if total_ops is None:
                        total_ops = episode['total_ops']
            
            if makespans:  # Only add if we have successful results
                result_collector.add_policy_results(
                    policy_name=policy_name,
                    instance_filename=f"global_{instance_key}",
                    makespans=makespans,
                    execution_times=execution_times,
                    estimated_makespan=estimated_makespan or 0.0,
                    total_rewards=rewards,
                    is_truncated_list=is_truncated_list,
                    n_ops=total_ops
                )
    
    # Save all results
    result_collector.save_all_results()
    
    # Save detailed parallel results
    parallel_results_path = Path(output_dir) / "parallel_execution_results.pkl"
    with open(parallel_results_path, 'wb') as f:
        pickle.dump(parallel_collector.results, f)
    print(f"Parallel execution details saved to {parallel_results_path}")
    
    # Print summary
    summary = parallel_collector.get_results_summary()
    print(f"\nExecution Summary:")
    print(f"  Total instances tested: {summary['total_instances']}")
    print(f"  Total episodes executed: {summary['total_tests']}")
    print(f"  Successful episodes: {summary['successful_tests']}")
    print(f"  Overall success rate: {summary['success_rate']*100:.2f}%")
    
    return result_collector 