import os
import re
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from configs import dfjspt_params
from local_realtime_scheduling.result_collector import ResultCollector, create_timestamped_output_dir
from local_realtime_scheduling.result_analyzer import ResultAnalyzer
from local_realtime_scheduling.InterfaceWithGlobal.divide_global_schedule_to_local_from_pkl import LocalSchedule, Local_Job_schedule
from local_realtime_scheduling.BaselineMethods.DispatchingRules.machine_agent_heuristics import machine_heuristic
from local_realtime_scheduling.BaselineMethods.DispatchingRules.transbot_agent_heuristics import transbot_heuristic
from local_realtime_scheduling.test_heuristic_combinations import TOP_HEURISTIC_COMBINATIONS




def run_random_policy_with_data_collection(
    local_instance_file: str,
    num_episodes: int = 10,
    detailed_log: bool = False
) -> Tuple[List[float], List[float], float, List[float], List[bool]]:
    """
    Run random policy and return structured results
    Returns: (makespans, execution_times, estimated_makespan, total_rewards, is_truncated_list)
    """
    print(f"Running Random Policy on {local_instance_file}")
    
    from local_realtime_scheduling.Environment.LocalSchedulingMultiAgentEnv_v3_4 import LocalSchedulingMultiAgentEnv
    from local_realtime_scheduling.Agents.generate_training_data import generate_reset_options_for_training
    
    config = {
        "n_machines": dfjspt_params.n_machines,
        "n_transbots": dfjspt_params.n_transbots,
        "factory_instance_seed": dfjspt_params.factory_instance_seed,
        "enable_dynamic_agent_filtering": getattr(dfjspt_params, 'enable_dynamic_agent_filtering', False),
        # "enable_dynamic_agent_filtering": getattr(dfjspt_params, 'enable_dynamic_agent_filtering', True),
        # "no_obs_solution_type": getattr(dfjspt_params, 'no_obs_solution_type', "dummy_agent"),
        # "render_mode": "human",
    }
    scheduling_env = LocalSchedulingMultiAgentEnv(config)
    
    makespans = []
    execution_times = []
    total_rewards = []
    is_truncated_list = []
    estimated_makespan = None

    for episode_id in range(num_episodes):
        start_time = time.time()
        
        env_reset_options = generate_reset_options_for_training(
            local_schedule_filename=local_instance_file,
            for_training=False,
        )

        observations, infos = scheduling_env.reset(options=env_reset_options)
        
        if estimated_makespan is None:
            estimated_makespan = scheduling_env.initial_estimated_makespan

        done = {'__all__': False}
        truncated = {'__all__': False}
        episode_rewards = {}
        for agent in scheduling_env.agents:
            episode_rewards[agent] = 0.0

        while (not done['__all__']) and (not truncated['__all__']):
            actions = {}
            for agent_id, obs in observations.items():
                action_mask = obs['action_mask']
                valid_actions = [i for i, valid in enumerate(action_mask) if valid == 1]
                if valid_actions:
                    if len(valid_actions) > 1:
                        valid_actions.pop(-1)
                    actions[agent_id] = np.random.choice(valid_actions)
                else:
                    raise Exception(f"No valid actions for agent {agent_id}!")

            observations, rewards, done, truncated, info = scheduling_env.step(actions)

            for agent, reward in rewards.items():
                episode_rewards[agent] += reward

        end_time = time.time()
        execution_time = end_time - start_time
        
        # Extract results
        if scheduling_env.local_result.actual_local_makespan is not None:
            delta_makespan = scheduling_env.local_result.actual_local_makespan - scheduling_env.local_result.time_window_start
            is_truncated = False
        else:
            delta_makespan = scheduling_env.current_time_after_step - scheduling_env.local_result.time_window_start
            is_truncated = True

        makespans.append(delta_makespan)
        execution_times.append(execution_time)
        total_rewards.append(episode_rewards.get('machine0', 0.0))
        is_truncated_list.append(is_truncated)

        if detailed_log:
            print(f"  Episode {episode_id+1}: makespan={delta_makespan:.2f}, time={execution_time:.2f}s")

    if detailed_log:
        print(f"  Average makespan: {np.mean(makespans):.2f}")
        print(f"  Average time: {np.mean(execution_times):.2f}s")

    return makespans, execution_times, estimated_makespan, total_rewards, is_truncated_list


def run_specific_heuristic_policy_with_data_collection(
    heuristic_combo: Dict[str, Any],
    local_instance_file: str,
    num_episodes: int = 10,
    detailed_log: bool = False
) -> Tuple[List[float], List[float], float, List[float], List[bool]]:
    """
    Run a specific heuristic combination policy and return structured results
    Returns: (makespans, execution_times, estimated_makespan, total_rewards, is_truncated_list)
    """
    print(f"Running {heuristic_combo['name']} Policy on {local_instance_file}")
    
    from local_realtime_scheduling.Environment.LocalSchedulingMultiAgentEnv_v3_4 import LocalSchedulingMultiAgentEnv
    from local_realtime_scheduling.Agents.generate_training_data import generate_reset_options_for_training
    
    config = {
        "n_machines": dfjspt_params.n_machines,
        "n_transbots": dfjspt_params.n_transbots,
        "factory_instance_seed": dfjspt_params.factory_instance_seed,
        "enable_dynamic_agent_filtering": getattr(dfjspt_params, 'enable_dynamic_agent_filtering', False),
    }
    scheduling_env = LocalSchedulingMultiAgentEnv(config)
    
    makespans = []
    execution_times = []
    total_rewards = []
    is_truncated_list = []
    estimated_makespan = None

    for episode_id in range(num_episodes):
        start_time = time.time()
        
        env_reset_options = generate_reset_options_for_training(
            local_schedule_filename=local_instance_file,
            for_training=False,
        )

        observations, infos = scheduling_env.reset(options=env_reset_options)
        
        if estimated_makespan is None:
            estimated_makespan = scheduling_env.initial_estimated_makespan

        done = {'__all__': False}
        truncated = {'__all__': False}
        episode_rewards = {}
        for agent in scheduling_env.agents:
            episode_rewards[agent] = 0.0

        while (not done['__all__']) and (not truncated['__all__']):
            actions = {}
            for agent_id, obs in observations.items():
                if agent_id.startswith("machine"):
                    try:
                        actions[agent_id] = machine_heuristic(
                            obs=obs,
                            job_rule=heuristic_combo['machine_job_rule'],
                            maint_rule=heuristic_combo['machine_maint_rule'],
                            due_date=estimated_makespan,
                            periodic_interval=heuristic_combo['machine_periodic_interval'],
                            threshold=heuristic_combo['machine_threshold'],
                        )
                    except Exception as e:
                        if detailed_log:
                            print(f"Machine heuristic error: {e}, falling back to random action")
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
                            job_rule=heuristic_combo['transbot_job_rule'],
                            charge_rule=heuristic_combo['transbot_charge_rule'],
                            threshold=heuristic_combo['transbot_threshold']
                        )
                    except Exception as e:
                        if detailed_log:
                            print(f"Transbot heuristic error: {e}, falling back to random action")
                        # Fall back to random valid action
                        action_mask = obs['action_mask']
                        valid_actions = [i for i, valid in enumerate(action_mask) if valid == 1]
                        if valid_actions and len(valid_actions) > 1:
                            valid_actions.pop(-1)  # Remove do-nothing action if possible
                        actions[agent_id] = np.random.choice(valid_actions) if valid_actions else 0

            observations, rewards, done, truncated, info = scheduling_env.step(actions)

            for agent, reward in rewards.items():
                episode_rewards[agent] += reward

        end_time = time.time()
        execution_time = end_time - start_time
        
        # Extract results
        if scheduling_env.local_result.actual_local_makespan is not None:
            delta_makespan = scheduling_env.local_result.actual_local_makespan - scheduling_env.local_result.time_window_start
            is_truncated = False
        else:
            delta_makespan = scheduling_env.current_time_after_step - scheduling_env.local_result.time_window_start
            is_truncated = True

        makespans.append(delta_makespan)
        execution_times.append(execution_time)
        total_rewards.append(episode_rewards.get('machine0', 0.0))
        is_truncated_list.append(is_truncated)

        if detailed_log:
            print(f"  Episode {episode_id+1}: makespan={delta_makespan:.2f}, time={execution_time:.2f}s")

    if detailed_log:
        print(f"  Average makespan: {np.mean(makespans):.2f}")
        print(f"  Average time: {np.mean(execution_times):.2f}s")

    return makespans, execution_times, estimated_makespan, total_rewards, is_truncated_list


def run_madrl_policy_with_data_collection(
    checkpoint_dir: str,
    local_instance_file: str,
    num_episodes: int = 10,
    detailed_log: bool = False
) -> Tuple[List[float], List[float], float, List[float], List[bool]]:
    """
    Run MADRL policy and return structured results
    Returns: (makespans, execution_times, estimated_makespan, total_rewards, is_truncated_list)
    """
    print(f"Running MADRL Policy on {local_instance_file}")
    
    import torch
    from ray.rllib.core.rl_module import RLModule
    from ray.rllib.utils.numpy import convert_to_numpy, softmax
    from local_realtime_scheduling.Environment.LocalSchedulingMultiAgentEnv_v3_4 import LocalSchedulingMultiAgentEnv
    from local_realtime_scheduling.Agents.generate_training_data import generate_reset_options_for_training
    
    # Load trained models
    machine_rl_module_checkpoint_dir = Path(checkpoint_dir) / "learner_group" / "learner" / "rl_module" / "p_machine"
    transbot_rl_module_checkpoint_dir = Path(checkpoint_dir) / "learner_group" / "learner" / "rl_module" / "p_transbot"

    machine_rl_module = RLModule.from_checkpoint(machine_rl_module_checkpoint_dir)
    transbot_rl_module = RLModule.from_checkpoint(transbot_rl_module_checkpoint_dir)
    
    config = {
        "n_machines": dfjspt_params.n_machines,
        "n_transbots": dfjspt_params.n_transbots,
        "factory_instance_seed": dfjspt_params.factory_instance_seed,
        "enable_dynamic_agent_filtering": getattr(dfjspt_params, 'enable_dynamic_agent_filtering', False),
        # "enable_dynamic_agent_filtering": getattr(dfjspt_params, 'enable_dynamic_agent_filtering', True),
        # "no_obs_solution_type": getattr(dfjspt_params, 'no_obs_solution_type', "dummy_agent"),
        # "render_mode": "human",
    }
    scheduling_env = LocalSchedulingMultiAgentEnv(config)
    
    makespans = []
    execution_times = []
    total_rewards = []
    is_truncated_list = []
    estimated_makespan = None

    for episode_id in range(num_episodes):
        start_time = time.time()
        
        env_reset_options = generate_reset_options_for_training(
            local_schedule_filename=local_instance_file,
            for_training=False,
        )

        observations, infos = scheduling_env.reset(options=env_reset_options)
        
        if estimated_makespan is None:
            estimated_makespan = scheduling_env.initial_estimated_makespan

        done = {'__all__': False}
        truncated = {'__all__': False}
        episode_rewards = {}
        for agent in scheduling_env.agents:
            episode_rewards[agent] = 0.0

        while (not done['__all__']) and (not truncated['__all__']):
            actions = {}
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

            observations, rewards, done, truncated, info = scheduling_env.step(actions)

            for agent, reward in rewards.items():
                episode_rewards[agent] += reward

        end_time = time.time()
        execution_time = end_time - start_time
        
        # Extract results
        if scheduling_env.local_result.actual_local_makespan is not None:
            delta_makespan = scheduling_env.local_result.actual_local_makespan - scheduling_env.local_result.time_window_start
            is_truncated = False
        else:
            delta_makespan = scheduling_env.current_time_after_step - scheduling_env.local_result.time_window_start
            is_truncated = True

        makespans.append(delta_makespan)
        execution_times.append(execution_time)
        total_rewards.append(episode_rewards.get('machine0', 0.0))
        is_truncated_list.append(is_truncated)

        if detailed_log:
            print(f"  Episode {episode_id+1}: makespan={delta_makespan:.2f}, time={execution_time:.2f}s")

    if detailed_log:
        print(f"  Average makespan: {np.mean(makespans):.2f}")
        print(f"  Average time: {np.mean(execution_times):.2f}s")

    return makespans, execution_times, estimated_makespan, total_rewards, is_truncated_list


def test_one_instance_with_data_collection(
    file_name: str,
    result_collector: ResultCollector,
    checkpoint_dir: Optional[str] = None,
    num_repeat: int = 5,
    do_render: bool = False,
    do_plot_gantt: bool = False,
    detailed_log: bool = True,
    policies_to_test: List[str] = None
) -> None:
    """
    Test one instance with all specified policies and collect results
    
    Args:
        file_name: Instance file name
        result_collector: ResultCollector instance
        checkpoint_dir: Path to MADRL checkpoint (required if testing MADRL)
        num_repeat: Number of episodes per policy
        do_render: Whether to render the environment
        do_plot_gantt: Whether to plot Gantt charts
        detailed_log: Whether to print detailed logs
        policies_to_test: List of policies to test ['heuristic_X', 'madrl', 'initial']
    """
    
    if policies_to_test is None:
        # Default: test all 5 heuristics, and madrl
        policies_to_test = [combo['name'] for combo in TOP_HEURISTIC_COMBINATIONS] + ['madrl']
    
    print(f"\n{'='*60}")
    print(f"Testing instance: {file_name}")
    print(f"Policies: {', '.join(policies_to_test)}")
    print(f"Episodes per policy: {num_repeat}")
    print(f"{'='*60}")
    
    # Test Random Policy
    if 'random' in policies_to_test:
        try:
            makespans, exec_times, est_makespan, rewards, is_truncated = run_random_policy_with_data_collection(
                file_name, num_repeat, detailed_log
            )
            result_collector.add_policy_results(
                policy_name='random',
                instance_filename=file_name,
                makespans=makespans,
                execution_times=exec_times,
                estimated_makespan=est_makespan,
                total_rewards=rewards,
                is_truncated_list=is_truncated
            )
            print(f"✓ Random policy completed")
        except Exception as e:
            print(f"✗ Random policy failed: {e}")

    # Test each of the top 5 Heuristic Policies
    for heuristic_combo in TOP_HEURISTIC_COMBINATIONS:
        if heuristic_combo['name'] in policies_to_test:
            try:
                makespans, exec_times, est_makespan, rewards, is_truncated = run_specific_heuristic_policy_with_data_collection(
                    heuristic_combo, file_name, num_repeat, detailed_log
                )
                result_collector.add_policy_results(
                    policy_name=heuristic_combo['name'],
                    instance_filename=file_name,
                    makespans=makespans,
                    execution_times=exec_times,
                    estimated_makespan=est_makespan,
                    total_rewards=rewards,
                    is_truncated_list=is_truncated
                )
                print(f"✓ {heuristic_combo['name']} policy completed")
            except Exception as e:
                print(f"✗ {heuristic_combo['name']} policy failed: {e}")

    # Test MADRL Policy
    if 'madrl' in policies_to_test:
        if checkpoint_dir is None:
            print(f"✗ MADRL policy skipped: checkpoint_dir not provided")
        else:
            try:
                makespans, exec_times, est_makespan, rewards, is_truncated = run_madrl_policy_with_data_collection(
                    checkpoint_dir, file_name, num_repeat, detailed_log
                )
                result_collector.add_policy_results(
                    policy_name='madrl',
                    instance_filename=file_name,
                    makespans=makespans,
                    execution_times=exec_times,
                    estimated_makespan=est_makespan,
                    total_rewards=rewards,
                    is_truncated_list=is_truncated
                )
                print(f"✓ MADRL policy completed")
            except Exception as e:
                print(f"✗ MADRL policy failed: {e}")

    # Test Initial Schedule Policy (if enabled)
    if 'initial' in policies_to_test:
        print(f"⚠ Initial schedule policy not yet implemented with data collection")


def run_comprehensive_comparison(
    output_dir: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    num_repeat: int = 5,
    max_instances: Optional[int] = None,
    policies_to_test: List[str] = None,
    detailed_log: bool = True,
    instance_filter: Optional[Dict[str, Any]] = None
) -> ResultCollector:
    """
    Run comprehensive comparison across multiple instances
    
    Args:
        output_dir: Directory to save results (if None, uses timestamped directory)
        checkpoint_dir: Path to MADRL checkpoint
        num_repeat: Number of episodes per policy per instance
        max_instances: Maximum number of instances to test (None for all)
        policies_to_test: List of policies to test
        detailed_log: Whether to print detailed logs
        instance_filter: Filter for selecting instances (e.g., {'min_jobs': 50, 'max_jobs': 100})
        
    Returns:
        ResultCollector with all results
    """

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if output_dir is None:
        output_dir = create_timestamped_output_dir(os.path.dirname(current_dir) + "/results_data/comprehensive_comparison_results")
    
    if policies_to_test is None:
        # Default: test all 5 heuristics, and madrl (if checkpoint provided)
        policies_to_test = [combo['name'] for combo in TOP_HEURISTIC_COMBINATIONS]
        if checkpoint_dir is not None:
            policies_to_test.append('madrl')
    
    # Initialize result collector
    result_collector = ResultCollector(output_dir)
    
    # Get instance files
    local_schedule_dir = current_dir + \
                         "/InterfaceWithGlobal/local_schedules" + \
                         f"/M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}W{dfjspt_params.time_window_size}/testing"
    
    # Get all .pkl files in the directory
    local_schedule_files = [file_name for file_name in os.listdir(local_schedule_dir) if file_name.endswith(".pkl")]
    
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
    
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE POLICY COMPARISON")
    print(f"{'='*80}")
    print(f"Total instances to test: {len(reversed_sorted_files)}")
    print(f"Policies: {', '.join(policies_to_test)}")
    print(f"Episodes per policy: {num_repeat}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}")
    
    # Test each instance
    for i, file_name in enumerate(reversed_sorted_files):
        print(f"\nProgress: {i+1}/{len(reversed_sorted_files)} instances")
        
        test_one_instance_with_data_collection(
            file_name=file_name,
            result_collector=result_collector,
            checkpoint_dir=checkpoint_dir,
            num_repeat=num_repeat,
            do_render=False,
            do_plot_gantt=False,
            detailed_log=detailed_log,
            policies_to_test=policies_to_test
        )
    
    # Save all results
    result_collector.save_all_results()
    
    return result_collector


def quick_test_comparison(
    instance_file: str = None,
    checkpoint_dir: str = None,
    num_repeat: int = 3
) -> ResultCollector:
    """
    Quick test for debugging and development
    """
    if instance_file is None:
        # Use a default test instance that exists in the testing directory
        instance_file = "local_schedule_J10I108_0_ops40.pkl"

    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = create_timestamped_output_dir(current_dir + "/quick_test_results")
    result_collector = ResultCollector(output_dir)
    
    policies = [combo['name'] for combo in TOP_HEURISTIC_COMBINATIONS]
    if checkpoint_dir is not None:
        policies.append('madrl')
    
    test_one_instance_with_data_collection(
        file_name=instance_file,
        result_collector=result_collector,
        checkpoint_dir=checkpoint_dir,
        num_repeat=num_repeat,
        detailed_log=True,
        policies_to_test=policies
    )
    
    result_collector.save_all_results()
    return result_collector


# Example usage and main function
def main():
    """Main function for running comparisons"""
    
    # Configuration
    # CHECKPOINT_DIR = '/Users/dadada/Downloads/ema_rts_data/ray_results/M16T5W300/PPO_LocalSchedulingMultiAgentEnv_a5bcd_00000_0_2025-05-31_22-15-04/checkpoint_000021'
    CHECKPOINT_DIR = '/home/rglab/wwd/ema_rts/local_realtime_scheduling/Agents/ray_results/M50T50W300/PPO_LocalSchedulingMultiAgentEnv_241ff_00000_0_2025-05-30_11-20-50/checkpoint_000057'


    # ========== CONFIGURATION ==========
    # Set this to True to run a new comprehensive comparison
    # Set this to False to load existing results from the specified pickle file
    RUN_NEW_COMPARISON = True
    
    # If loading existing results, specify the path to the pickle file
    EXISTING_RESULTS_PICKLE = "/path/to/your/saved/results.pkl"
    
    # Output directory for new results (only used if RUN_NEW_COMPARISON=True)
    CUSTOM_OUTPUT_DIR = None  # Set to None for auto-generated timestamped directory
    
    # ===================================

    if RUN_NEW_COMPARISON:
        print("Running new comprehensive comparison...")
        # Full comparison
        full_result = run_comprehensive_comparison(
            output_dir=CUSTOM_OUTPUT_DIR,
            checkpoint_dir=CHECKPOINT_DIR,
            num_repeat=5,
            max_instances=20,  # Limit for testing
            instance_filter={
                # 'min_instance_id': 100,
                # 'max_instance_id': 105,
                # 'max_jobs': 50,
            },
            detailed_log=True,
        )
        
        # Save the ResultCollector object specifically for later loading
        collector_pickle_path = full_result.output_dir / "result_collector.pkl"
        print(f"\nSaving ResultCollector for future loading to: {collector_pickle_path}")
        with open(collector_pickle_path, 'wb') as f:
            import pickle
            pickle.dump(full_result, f)
        print(f"✓ ResultCollector saved! You can load it later using:")
        print(f"  EXISTING_RESULTS_PICKLE = '{collector_pickle_path}'")
        print(f"  RUN_NEW_COMPARISON = False")
        
    else:
        print(f"Loading existing results from: {EXISTING_RESULTS_PICKLE}")
        try:
            with open(EXISTING_RESULTS_PICKLE, 'rb') as f:
                import pickle
                full_result = pickle.load(f)
            print("✓ Results loaded successfully!")
            full_result.print_summary()  # Print summary of loaded results
        except FileNotFoundError:
            print(f"✗ Error: File not found: {EXISTING_RESULTS_PICKLE}")
            print("Please check the file path or set RUN_NEW_COMPARISON=True to run a new comparison.")
            return
        except Exception as e:
            print(f"✗ Error loading results: {e}")
            return

    # Create analyzer from collector
    print("\nCreating result analyzer...")
    full_result_analyzer = ResultAnalyzer(result_collector=full_result)

    # Get statistics
    print("Generating basic statistics...")
    basic_stats = full_result_analyzer.get_basic_statistics()
    policy_comparison = full_result_analyzer.compare_policies_overall()

    # Statistical significance testing
    print("Performing statistical significance tests...")
    sig_test = full_result_analyzer.statistical_significance_test('heuristic_spt_periodic')

    # Generate visualizations
    print("Generating visualizations...")
    full_result_analyzer.plot_policy_comparison_boxplot()
    full_result_analyzer.plot_performance_by_instance_size()
    full_result_analyzer.plot_execution_time_comparison()
    full_result_analyzer.plot_execution_time_per_step_comparison()

    # Generate comprehensive report
    print("Generating comprehensive report...")
    full_result_analyzer.generate_comprehensive_report(output_dir=full_result.output_dir / "analysis_results")
    
    print(f"\n{'='*60}")
    print("Analysis completed!")
    if RUN_NEW_COMPARISON:
        print(f"Results and analysis saved to: {full_result.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main() 