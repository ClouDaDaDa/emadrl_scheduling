import os
import re
import time
import pickle
import numpy as np
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
from local_realtime_scheduling.result_collector import ResultCollector, create_timestamped_output_dir
from local_realtime_scheduling.result_analyzer import ResultAnalyzer
from local_realtime_scheduling.BaselineMethods.DispatchingRules.machine_agent_heuristics import machine_heuristic
from local_realtime_scheduling.BaselineMethods.DispatchingRules.transbot_agent_heuristics import transbot_heuristic
from local_realtime_scheduling.test_heuristic_combinations import TOP_HEURISTIC_COMBINATIONS


@dataclass
class LocalTestTask:
    """Represents a single local test task to be executed"""
    instance_file: str
    policy_type: str
    episode_id: int
    heuristic_combo: Optional[Dict[str, Any]] = None
    task_id: Optional[str] = None
    
    def __post_init__(self):
        if self.task_id is None:
            # Extract meaningful info from filename for task ID
            match = re.match(r'local_schedule_J(\d+)I(\d+)_(\d+)_ops(\d+)\.pkl', self.instance_file)
            if match:
                n_jobs, instance_id, window_id, n_ops = match.groups()
                self.task_id = f"J{n_jobs}I{instance_id}W{window_id}_{self.policy_type}_ep{self.episode_id}"
            else:
                self.task_id = f"{self.instance_file}_{self.policy_type}_ep{self.episode_id}"


@dataclass 
class LocalTestResult:
    """Represents the result of a single local test task"""
    task: LocalTestTask
    makespan: float
    execution_time: float
    total_reward: float
    initial_estimated_makespan: float
    n_ops: int
    is_truncated: bool
    success: bool = True
    error_message: Optional[str] = None


class ParallelLocalResultCollector:
    """Thread-safe result collector for parallel local execution"""
    
    def __init__(self, output_dir: str, is_global_analysis: bool = False):
        self.output_dir = Path(output_dir)
        self.is_global_analysis = is_global_analysis
        self.results = {}
        self.lock = threading.Lock()
        
    def add_result(self, result: LocalTestResult):
        """Thread-safe method to add a test result"""
        with self.lock:
            instance_key = result.task.instance_file
            if instance_key not in self.results:
                self.results[instance_key] = defaultdict(list)
            
            self.results[instance_key][result.task.policy_type].append({
                'episode_id': result.task.episode_id,
                'makespan': result.makespan,
                'execution_time': result.execution_time,
                'total_reward': result.total_reward,
                'initial_estimated_makespan': result.initial_estimated_makespan,
                'n_ops': result.n_ops,
                'is_truncated': result.is_truncated,
                'success': result.success,
                'error_message': result.error_message
            })
    
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


@ray.remote(num_cpus=1)
def run_single_local_episode_remote(
    task: LocalTestTask,
    testing_dir: str,
    checkpoint_dir: Optional[str] = None,
    detailed_log: bool = False
) -> LocalTestResult:
    """
    Ray remote function to run a single episode of a policy on a local instance
    
    This function is designed to be completely independent and stateless to enable
    safe parallel execution across multiple processes.
    """
    
    try:
        # Get the full path to the local instance file
        local_file_path = os.path.join(testing_dir, task.instance_file)
        
        if not os.path.exists(local_file_path):
            return LocalTestResult(
                task=task,
                makespan=0.0,
                execution_time=0.0,
                total_reward=0.0,
                initial_estimated_makespan=0.0,
                n_ops=0,
                is_truncated=True,
                success=False,
                error_message=f"Local instance file not found: {local_file_path}"
            )
        
        # Extract n_ops from filename for reporting
        n_ops = 0
        ops_match = re.search(r'_ops(\d+)\.pkl$', task.instance_file)
        if ops_match:
            n_ops = int(ops_match.group(1))
        
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
        
        # Create fresh environment for this episode
        from local_realtime_scheduling.Environment.LocalSchedulingMultiAgentEnv_v3_4 import LocalSchedulingMultiAgentEnv
        from local_realtime_scheduling.Agents.generate_training_data import generate_reset_options_for_training
        
        scheduling_env = LocalSchedulingMultiAgentEnv(env_config)
        
        # Generate reset options for the local instance
        env_reset_options = generate_reset_options_for_training(
            local_schedule_filename=task.instance_file,
            for_training=False,
        )
        
        # Reset environment
        observations, infos = scheduling_env.reset(options=env_reset_options)
        
        # Capture initial estimated makespan
        initial_estimated_makespan = scheduling_env.initial_estimated_makespan
        
        done = {'__all__': False}
        truncated = {'__all__': False}
        episode_rewards = {}
        for agent in scheduling_env.agents:
            episode_rewards[agent] = 0.0
        
        # Run the episode
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
            
            # Step the environment
            observations, rewards, done, truncated, info = scheduling_env.step(actions)
            
            # Accumulate rewards
            for agent, reward in rewards.items():
                episode_rewards[agent] += reward
        
        # Calculate final results
        episode_end_time = time.time()
        episode_execution_time = episode_end_time - episode_start_time
        
        # Extract results - for local instances, we use delta makespan
        if scheduling_env.local_result.actual_local_makespan is not None:
            delta_makespan = scheduling_env.local_result.actual_local_makespan - scheduling_env.local_result.time_window_start
            is_truncated = False
        else:
            delta_makespan = scheduling_env.current_time_after_step - scheduling_env.local_result.time_window_start
            is_truncated = True
        
        total_reward = episode_rewards.get('machine0', 0.0)
        
        if detailed_log:
            print(f"[{task.task_id}] Completed: makespan={delta_makespan:.2f}, time={episode_execution_time:.2f}s")
        
        return LocalTestResult(
            task=task,
            makespan=delta_makespan,
            execution_time=episode_execution_time,
            total_reward=total_reward,
            initial_estimated_makespan=initial_estimated_makespan or 0.0,
            n_ops=n_ops,
            is_truncated=is_truncated,
            success=True
        )
        
    except Exception as e:
        import traceback
        error_details = f"{str(e)}\n{traceback.format_exc()}"
        return LocalTestResult(
            task=task,
            makespan=0.0,
            execution_time=0.0,
            total_reward=0.0,
            initial_estimated_makespan=0.0,
            n_ops=0,
            is_truncated=True,
            success=False,
            error_message=error_details
        )


def create_local_test_tasks(
    instance_files: List[str],
    policies_to_test: List[str],
    num_repeat: int
) -> List[LocalTestTask]:
    """Create all test tasks for parallel execution"""
    
    tasks = []
    
    for instance_file in instance_files:
        for policy in policies_to_test:
            # Get heuristic combination if it's a heuristic policy
            heuristic_combo = None
            if policy.startswith('heuristic_'):
                heuristic_combo = next((combo for combo in TOP_HEURISTIC_COMBINATIONS if combo['name'] == policy), None)
                if heuristic_combo is None:
                    print(f"Warning: Unknown heuristic combination {policy}, skipping...")
                    continue
            
            for episode_id in range(num_repeat):
                task = LocalTestTask(
                    instance_file=instance_file,
                    policy_type=policy,
                    episode_id=episode_id,
                    heuristic_combo=heuristic_combo
                )
                tasks.append(task)
    
    return tasks


def run_parallel_local_instance_comparison(
    output_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    num_repeat: int = 5,
    max_instances: Optional[int] = None,
    policies_to_test: List[str] = None,
    max_concurrent_tasks: Optional[int] = None,
    detailed_log: bool = True,
    instance_filter: Optional[Dict[str, Any]] = None
) -> ResultCollector:
    """
    Run parallel comprehensive comparison across multiple local instances
    
    Args:
        output_dir: Directory to save results (if None, uses timestamped directory)
        checkpoint_dir: Path to MADRL checkpoint
        num_repeat: Number of episodes per policy per instance
        max_instances: Maximum number of instances to test (None for all)
        policies_to_test: List of policies to test
        max_concurrent_tasks: Maximum number of concurrent Ray tasks (None = auto-detect)
        detailed_log: Whether to print detailed logs
        instance_filter: Filter for selecting instances (e.g., {'min_jobs': 50, 'max_jobs': 100})
    
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
    if policies_to_test is None:
        # Default: test all 5 heuristics and madrl (if checkpoint provided)
        policies_to_test = [combo['name'] for combo in TOP_HEURISTIC_COMBINATIONS]
        if checkpoint_dir:
            policies_to_test.append('madrl')
    if max_concurrent_tasks is None:
        max_concurrent_tasks = min(os.cpu_count(), 8)  # Reasonable default
    
    # Create output directory
    if output_dir is None:
        output_dir = create_timestamped_output_dir(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) +
            "/results_data/local_parallel_comparison"
        )
    
    # Initialize result collectors
    result_collector = ResultCollector(output_dir, is_global_analysis=False)
    parallel_collector = ParallelLocalResultCollector(output_dir, is_global_analysis=False)
    
    # Get testing directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    testing_dir = os.path.dirname(current_dir) + \
        "/InterfaceWithGlobal/local_schedules" + \
        f"/M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}W{dfjspt_params.time_window_size}/testing"
    
    # Get all .pkl files in the directory
    local_schedule_files = [file_name for file_name in os.listdir(testing_dir) if file_name.endswith(".pkl")]
    
    # Apply instance filter if provided
    if instance_filter:
        filtered_files = []
        for file_name in local_schedule_files:
            pattern = r'local_schedule_J(\d+)I(\d+)_(\d+)_ops(\d+)\.pkl'
            match = re.match(pattern, file_name)
            if match:
                n_jobs = int(match.group(1))
                instance_id = int(match.group(2))
                window_id = int(match.group(3))
                n_ops = int(match.group(4))
                
                # Apply filters
                if 'min_jobs' in instance_filter and n_jobs < instance_filter['min_jobs']:
                    continue
                if 'max_jobs' in instance_filter and n_jobs > instance_filter['max_jobs']:
                    continue
                if 'min_instance_id' in instance_filter and instance_id < instance_filter['min_instance_id']:
                    continue
                if 'max_instance_id' in instance_filter and instance_id > instance_filter['max_instance_id']:
                    continue
                if 'min_ops' in instance_filter and n_ops < instance_filter['min_ops']:
                    continue
                if 'max_ops' in instance_filter and n_ops > instance_filter['max_ops']:
                    continue
                
                filtered_files.append(file_name)
        
        local_schedule_files = filtered_files
    
    # Sort files by the 'ops' number for consistent processing
    reversed_sorted_files = sorted(local_schedule_files, 
                                  key=lambda f: int(re.search(r'ops(\d+)\.pkl', f).group(1)),
                                  reverse=True)
    
    # Limit number of instances if specified
    if max_instances is not None and max_instances < len(reversed_sorted_files):
        reversed_sorted_files = reversed_sorted_files[:max_instances]
    
    # Create all test tasks
    all_tasks = create_local_test_tasks(reversed_sorted_files, policies_to_test, num_repeat)
    total_tasks = len(all_tasks)
    
    print(f"\n{'='*80}")
    print(f"PARALLEL LOCAL INSTANCE COMPARISON")
    print(f"{'='*80}")
    print(f"Total instances to test: {len(reversed_sorted_files)}")
    print(f"Policies: {', '.join(policies_to_test)}")
    print(f"Episodes per policy: {num_repeat}")
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
        batch_tasks = all_tasks[batch_start:batch_end]
        
        # Submit batch
        for task in batch_tasks:
            future = run_single_local_episode_remote.remote(
                task=task,
                testing_dir=testing_dir,
                checkpoint_dir=checkpoint_dir if task.policy_type == 'madrl' else None,
                detailed_log=False  # Reduce logging noise in parallel mode
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
                    parallel_collector.add_result(result)
                    
                    if result.success:
                        completed_tasks += 1
                        if detailed_log and completed_tasks % 10 == 0:
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
                parallel_collector.add_result(result)
                
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
    
    # Convert parallel results to standard ResultCollector format
    print("\n📊 Aggregating results...")
    
    for instance_file, instance_results in parallel_collector.results.items():
        for policy_name, episodes in instance_results.items():
            # Group episodes by their properties
            makespans = []
            execution_times = []
            rewards = []
            estimated_makespan = None
            n_ops = None
            is_truncated_list = []
            
            for episode in episodes:
                if episode['success']:
                    makespans.append(episode['makespan'])
                    execution_times.append(episode['execution_time'])
                    rewards.append(episode['total_reward'])
                    is_truncated_list.append(episode['is_truncated'])
                    
                    # Use values from first successful episode
                    if estimated_makespan is None:
                        estimated_makespan = episode['initial_estimated_makespan']
                    if n_ops is None:
                        n_ops = episode['n_ops']
            
            if makespans:  # Only add if we have successful results
                result_collector.add_policy_results(
                    policy_name=policy_name,
                    instance_filename=instance_file,
                    makespans=makespans,
                    execution_times=execution_times,
                    estimated_makespan=estimated_makespan or 0.0,
                    total_rewards=rewards,
                    is_truncated_list=is_truncated_list,
                    n_ops=n_ops
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