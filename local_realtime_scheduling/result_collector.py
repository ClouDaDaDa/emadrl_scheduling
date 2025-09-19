import os
import re
import csv
import json
import pickle
import time
import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import pandas as pd
import numpy as np
from configs import dfjspt_params


@dataclass
class EpisodeResult:
    """Results for a single episode of one policy on one instance"""
    episode_id: int
    policy_name: str
    instance_filename: str
    makespan: float
    delta_makespan: float
    estimated_makespan: float
    delta_estimated_makespan: float
    total_reward: float
    execution_time: float
    execution_time_per_step: float  # Average execution time per decision step (execution_time / delta_makespan)
    is_truncated: bool
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    
    # Instance characteristics extracted from filename
    n_jobs: Optional[int] = None
    instance_id: Optional[int] = None
    window_id: Optional[int] = None
    n_ops: Optional[int] = None
    is_global_instance: bool = False  # Flag to distinguish global vs local instances
    
    def __post_init__(self):
        """Extract instance characteristics from filename"""
        if self.instance_filename:
            self._parse_instance_filename()
    
    def _parse_instance_filename(self):
        """Parse instance filename to extract characteristics"""
        # Pattern for global instances: global_J{n_jobs}I{instance_id}
        global_pattern = r'global_J(\d+)I(\d+)'
        global_match = re.match(global_pattern, self.instance_filename)
        
        if global_match:
            self.is_global_instance = True
            self.n_jobs = int(global_match.group(1))
            self.instance_id = int(global_match.group(2))
            self.window_id = None  # Global instances span multiple windows
            self.n_ops = None  # Not applicable for global instances
        else:
            # Pattern for local instances: local_schedule_J{n_jobs}I{instance_id}_{window_id}_ops{n_ops}.pkl
            local_pattern = r'local_schedule_J(\d+)I(\d+)_(\d+)_ops(\d+)\.pkl'
            local_match = re.match(local_pattern, self.instance_filename)
            if local_match:
                self.is_global_instance = False
                self.n_jobs = int(local_match.group(1))
                self.instance_id = int(local_match.group(2))
                self.window_id = int(local_match.group(3))
                self.n_ops = int(local_match.group(4))


@dataclass
class InstanceResult:
    """Aggregated results for all policies on one instance"""
    instance_filename: str
    n_jobs: Optional[int] = None
    instance_id: Optional[int] = None
    window_id: Optional[int] = None
    n_ops: Optional[int] = None
    is_global_instance: bool = False  # Flag to distinguish global vs local instances
    
    # Policy results - dictionary mapping policy name to list of episode results
    policy_results: Dict[str, List[EpisodeResult]] = field(default_factory=dict)
    
    # Keep backwards compatibility for legacy policy names
    @property
    def random_results(self) -> List[EpisodeResult]:
        return self.policy_results.get('random', [])
    
    @property
    def heuristic_results(self) -> List[EpisodeResult]:
        return self.policy_results.get('heuristic', [])
    
    @property
    def madrl_results(self) -> List[EpisodeResult]:
        return self.policy_results.get('madrl', [])
    
    @property
    def initial_results(self) -> List[EpisodeResult]:
        return self.policy_results.get('initial', [])

    def __post_init__(self):
        """Extract instance characteristics from filename"""
        if self.instance_filename:
            self._parse_instance_filename()
    
    def _parse_instance_filename(self):
        """Parse instance filename to extract characteristics"""
        # Pattern for global instances: global_J{n_jobs}I{instance_id}
        global_pattern = r'global_J(\d+)I(\d+)'
        global_match = re.match(global_pattern, self.instance_filename)
        
        if global_match:
            self.is_global_instance = True
            self.n_jobs = int(global_match.group(1))
            self.instance_id = int(global_match.group(2))
            self.window_id = None  # Global instances span multiple windows
            self.n_ops = None  # Not applicable for global instances
        else:
            # Pattern for local instances: local_schedule_J{n_jobs}I{instance_id}_{window_id}_ops{n_ops}.pkl
            local_pattern = r'local_schedule_J(\d+)I(\d+)_(\d+)_ops(\d+)\.pkl'
            local_match = re.match(local_pattern, self.instance_filename)
            if local_match:
                self.is_global_instance = False
                self.n_jobs = int(local_match.group(1))
                self.instance_id = int(local_match.group(2))
                self.window_id = int(local_match.group(3))
                self.n_ops = int(local_match.group(4))
    
    def get_policy_statistics(self, policy_name: str) -> Dict[str, float]:
        """Get statistics for a specific policy"""
        results = self.policy_results.get(policy_name, [])
        if not results:
            return {}
        
        # For global instances, use actual makespans; for local instances, use delta makespans
        if self.is_global_instance:
            makespans = [r.makespan for r in results]  # Use actual global makespans
        else:
            makespans = [r.delta_makespan for r in results]  # Use delta makespans for local instances
            
        execution_times = [r.execution_time for r in results]
        execution_times_per_step = [r.execution_time_per_step for r in results]
        
        return {
            f"{policy_name}_mean_makespan": np.mean(makespans),
            f"{policy_name}_min_makespan": np.min(makespans),
            f"{policy_name}_max_makespan": np.max(makespans),
            f"{policy_name}_std_makespan": np.std(makespans),
            f"{policy_name}_mean_time": np.mean(execution_times),
            f"{policy_name}_std_time": np.std(execution_times),
            f"{policy_name}_mean_time_per_step": np.mean(execution_times_per_step),
            f"{policy_name}_std_time_per_step": np.std(execution_times_per_step),
            f"{policy_name}_min_time_per_step": np.min(execution_times_per_step),
            f"{policy_name}_max_time_per_step": np.max(execution_times_per_step),
            f"{policy_name}_num_episodes": len(results),
            f"{policy_name}_truncated_rate": sum(r.is_truncated for r in results) / len(results),
        }
    
    def get_all_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics for all policies"""
        stats = {
            'instance_filename': self.instance_filename,
            'n_jobs': self.n_jobs,
            'instance_id': self.instance_id,
            'window_id': self.window_id,
            'n_ops': self.n_ops,
            'is_global_instance': self.is_global_instance,
        }
        
        # Add policy-specific statistics for all policies
        for policy_name in self.policy_results.keys():
            policy_stats = self.get_policy_statistics(policy_name)
            stats.update(policy_stats)
        
        # Add comparative statistics for standard policies if they exist
        random_results = self.policy_results.get('random', [])
        heuristic_results = self.policy_results.get('heuristic', [])
        madrl_results = self.policy_results.get('madrl', [])
        
        # Choose the appropriate makespan values based on instance type
        if self.is_global_instance:
            # For global instances, use actual makespans
            if random_results and heuristic_results:
                random_mean = np.mean([r.makespan for r in random_results])
                heuristic_mean = np.mean([r.makespan for r in heuristic_results])
                stats['heuristic_vs_random_improvement'] = (random_mean - heuristic_mean) / random_mean * 100
            
            if heuristic_results and madrl_results:
                heuristic_mean = np.mean([r.makespan for r in heuristic_results])
                madrl_mean = np.mean([r.makespan for r in madrl_results])
                stats['madrl_vs_heuristic_improvement'] = (heuristic_mean - madrl_mean) / heuristic_mean * 100
            
            if random_results and madrl_results:
                random_mean = np.mean([r.makespan for r in random_results])
                madrl_mean = np.mean([r.makespan for r in madrl_results])
                stats['madrl_vs_random_improvement'] = (random_mean - madrl_mean) / random_mean * 100
        else:
            # For local instances, use delta makespans
            if random_results and heuristic_results:
                random_mean = np.mean([r.delta_makespan for r in random_results])
                heuristic_mean = np.mean([r.delta_makespan for r in heuristic_results])
                stats['heuristic_vs_random_improvement'] = (random_mean - heuristic_mean) / random_mean * 100
            
            if heuristic_results and madrl_results:
                heuristic_mean = np.mean([r.delta_makespan for r in heuristic_results])
                madrl_mean = np.mean([r.delta_makespan for r in madrl_results])
                stats['madrl_vs_heuristic_improvement'] = (heuristic_mean - madrl_mean) / heuristic_mean * 100
            
            if random_results and madrl_results:
                random_mean = np.mean([r.delta_makespan for r in random_results])
                madrl_mean = np.mean([r.delta_makespan for r in madrl_results])
                stats['madrl_vs_random_improvement'] = (random_mean - madrl_mean) / random_mean * 100
        
        return stats


class ResultCollector:
    """Main class for collecting, storing, and analyzing experiment results"""
    
    def __init__(self, output_dir: str = "experiment_results", is_global_analysis: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories
        self.csv_dir = self.output_dir / "csv"
        self.json_dir = self.output_dir / "json"
        self.pickle_dir = self.output_dir / "pickle"
        self.plots_dir = self.output_dir / "plots"
        
        for dir_path in [self.csv_dir, self.json_dir, self.pickle_dir, self.plots_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        # Results storage
        self.instance_results: Dict[str, InstanceResult] = {}
        self.is_global_analysis = is_global_analysis  # Flag for global vs local analysis
        self.experiment_metadata = {
            'start_time': datetime.datetime.now().isoformat(),
            'analysis_type': 'global' if is_global_analysis else 'local',
            'config': {
                'n_machines': dfjspt_params.n_machines,
                'n_transbots': dfjspt_params.n_transbots,
                'time_window_size': dfjspt_params.time_window_size,
                'factory_instance_seed': dfjspt_params.factory_instance_seed,
            }
        }
    
    def add_episode_result(self, 
                          policy_name: str,
                          instance_filename: str,
                          episode_id: int,
                          makespan: float,
                          delta_makespan: float,
                          estimated_makespan: float,
                          total_reward: float,
                          execution_time: float,
                          execution_time_per_step: Optional[float] = None,
                          is_truncated: bool = False,
                          n_ops: Optional[int] = None) -> None:
        """Add a single episode result"""
        
        delta_estimated_makespan = estimated_makespan - (makespan - delta_makespan)
        
        # Calculate execution_time_per_step if not provided
        if execution_time_per_step is None:
            execution_time_per_step = execution_time / delta_makespan if delta_makespan > 0 else 0.0
        
        episode_result = EpisodeResult(
            episode_id=episode_id,
            policy_name=policy_name,
            instance_filename=instance_filename,
            makespan=makespan,
            delta_makespan=delta_makespan,
            estimated_makespan=estimated_makespan,
            delta_estimated_makespan=delta_estimated_makespan,
            total_reward=total_reward,
            execution_time=execution_time,
            execution_time_per_step=execution_time_per_step,
            is_truncated=is_truncated
        )
        
        # Override n_ops if explicitly provided (for global instances)
        if n_ops is not None:
            episode_result.n_ops = n_ops
        
        # Get or create instance result
        if instance_filename not in self.instance_results:
            self.instance_results[instance_filename] = InstanceResult(instance_filename)
            # Set n_ops for the instance result if provided
            if n_ops is not None:
                self.instance_results[instance_filename].n_ops = n_ops
        
        instance_result = self.instance_results[instance_filename]
        
        # Add to appropriate policy results using the dynamic dictionary
        if policy_name not in instance_result.policy_results:
            instance_result.policy_results[policy_name] = []
        
        instance_result.policy_results[policy_name].append(episode_result)
    
    def add_policy_results(self,
                          policy_name: str,
                          instance_filename: str,
                          makespans: List[float],
                          execution_times: List[float],
                          estimated_makespan: float,
                          total_rewards: List[float] = None,
                          is_truncated_list: List[bool] = None,
                          n_ops: Optional[int] = None) -> None:
        """Add results for all episodes of a policy on one instance"""
        
        if total_rewards is None:
            total_rewards = [0.0] * len(makespans)
        if is_truncated_list is None:
            is_truncated_list = [False] * len(makespans)
        
        # Detect if this is a global instance by checking the filename pattern
        is_global = instance_filename.startswith('global_')
        
        for i, (makespan_value, exec_time, total_reward, is_truncated) in enumerate(
            zip(makespans, execution_times, total_rewards, is_truncated_list)):
            
            if is_global:
                # For global instances: makespan_value is already the final global makespan
                final_makespan = makespan_value
                delta_makespan = makespan_value - estimated_makespan  # Improvement over estimated
                # Calculate execution time per step based on delta makespan
                execution_time_per_step = exec_time / delta_makespan if delta_makespan > 0 else 0.0
            else:
                # For local instances: makespan_value is delta_makespan
                delta_makespan = makespan_value
                final_makespan = delta_makespan + estimated_makespan  # Absolute makespan
                # Calculate execution time per step based on delta makespan
                execution_time_per_step = exec_time / delta_makespan if delta_makespan > 0 else 0.0
            
            self.add_episode_result(
                policy_name=policy_name,
                instance_filename=instance_filename,
                episode_id=i + 1,
                makespan=final_makespan,
                delta_makespan=delta_makespan,
                estimated_makespan=estimated_makespan,
                total_reward=total_reward,
                execution_time=exec_time,
                execution_time_per_step=execution_time_per_step,
                is_truncated=is_truncated,
                n_ops=n_ops  # Pass n_ops for global instances
            )
    
    def save_episode_results_csv(self) -> None:
        """Save all episode results to CSV"""
        all_episodes = []
        
        for instance_result in self.instance_results.values():
            for policy_name, results in instance_result.policy_results.items():
                for result in results:
                    all_episodes.append(asdict(result))
        
        if all_episodes:
            df = pd.DataFrame(all_episodes)
            csv_path = self.csv_dir / "episode_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"Episode results saved to {csv_path}")
    
    def save_instance_summary_csv(self) -> None:
        """Save instance-level summary statistics to CSV"""
        summaries = []
        
        for instance_result in self.instance_results.values():
            summary = instance_result.get_all_statistics()
            summaries.append(summary)
        
        if summaries:
            df = pd.DataFrame(summaries)
            csv_path = self.csv_dir / "instance_summary.csv"
            df.to_csv(csv_path, index=False)
            print(f"Instance summary saved to {csv_path}")
    
    def save_policy_comparison_csv(self) -> None:
        """Save policy comparison results to CSV"""
        comparisons = []
        
        for instance_result in self.instance_results.values():
            comp = {
                'instance_filename': instance_result.instance_filename,
                'n_jobs': instance_result.n_jobs,
                'instance_id': instance_result.instance_id,
                'window_id': instance_result.window_id,
                'n_ops': instance_result.n_ops,
                'is_global_instance': instance_result.is_global_instance,
            }
            
            # Get best result for each policy
            for policy_name, results in instance_result.policy_results.items():
                if results:
                    # Use appropriate makespan values based on instance type
                    if instance_result.is_global_instance:
                        makespans = [r.makespan for r in results]  # Use actual makespans for global instances
                    else:
                        makespans = [r.delta_makespan for r in results]  # Use delta makespans for local instances
                        
                    execution_times_per_step = [r.execution_time_per_step for r in results]
                    # Clean policy name for column names (replace special characters)
                    clean_policy_name = policy_name.replace('-', '_').replace(' ', '_')
                    comp[f"{clean_policy_name}_best"] = min(makespans)
                    comp[f"{clean_policy_name}_mean"] = np.mean(makespans)
                    comp[f"{clean_policy_name}_worst"] = max(makespans)
                    comp[f"{clean_policy_name}_std"] = np.std(makespans)
                    comp[f"{clean_policy_name}_mean_time_per_step"] = np.mean(execution_times_per_step)
                    comp[f"{clean_policy_name}_std_time_per_step"] = np.std(execution_times_per_step)
            
            comparisons.append(comp)
        
        if comparisons:
            df = pd.DataFrame(comparisons)
            csv_path = self.csv_dir / "policy_comparison.csv"
            df.to_csv(csv_path, index=False)
            print(f"Policy comparison saved to {csv_path}")
    
    def save_json_results(self) -> None:
        """Save detailed results to JSON"""
        results_dict = {}
        
        for filename, instance_result in self.instance_results.items():
            results_dict[filename] = asdict(instance_result)
        
        # Add metadata
        output_data = {
            'metadata': self.experiment_metadata,
            'results': results_dict
        }
        
        json_path = self.json_dir / "detailed_results.json"
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"Detailed results saved to {json_path}")
    
    def save_pickle_results(self) -> None:
        """Save results as pickle for Python analysis"""
        pickle_path = self.pickle_dir / "results.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump({
                'metadata': self.experiment_metadata,
                'instance_results': self.instance_results
            }, f)
        print(f"Pickle results saved to {pickle_path}")
    
    def save_all_results(self) -> None:
        """Save results in all formats"""
        print(f"\nSaving experiment results to {self.output_dir}")
        self.save_episode_results_csv()
        self.save_instance_summary_csv()
        self.save_policy_comparison_csv()
        self.save_json_results()
        self.save_pickle_results()
        
        # Update metadata with end time
        self.experiment_metadata['end_time'] = datetime.datetime.now().isoformat()
        self.experiment_metadata['num_instances'] = len(self.instance_results)
        
        # Save metadata separately
        metadata_path = self.json_dir / "experiment_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.experiment_metadata, f, indent=2, default=str)
        
        print(f"Results saved successfully!")
        self.print_summary()
    
    def print_summary(self) -> None:
        """Print a summary of the experiment results"""
        print(f"\n{'='*60}")
        if self.is_global_analysis:
            print(f"GLOBAL INSTANCE EXPERIMENT SUMMARY")
        else:
            print(f"EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Total instances tested: {len(self.instance_results)}")
        
        # Overall statistics
        all_policies = set()
        for instance_result in self.instance_results.values():
            all_policies.update(instance_result.policy_results.keys())
        
        print(f"Policies tested: {', '.join(sorted(all_policies))}")
        
        # Configuration
        config = self.experiment_metadata['config']
        print(f"Configuration:")
        print(f"  - Machines: {config['n_machines']}")
        print(f"  - Transbots: {config['n_transbots']}")
        print(f"  - Time window size: {config['time_window_size']}")
        
        # Performance overview
        if len(self.instance_results) > 0:
            if self.is_global_analysis:
                print(f"\nGlobal Instance Performance (actual makespan):")
                
                for policy in sorted(all_policies):
                    all_makespans = []
                    all_execution_times = []
                    all_execution_times_per_step = []
                    for instance_result in self.instance_results.values():
                        results = instance_result.policy_results.get(policy, [])
                        if results:
                            # For global analysis, use actual makespans
                            makespans = [r.makespan for r in results]
                            execution_times = [r.execution_time for r in results]
                            execution_times_per_step = [r.execution_time_per_step for r in results]
                            all_makespans.extend(makespans)
                            all_execution_times.extend(execution_times)
                            all_execution_times_per_step.extend(execution_times_per_step)
                    
                    if all_makespans:
                        mean_makespan = np.mean(all_makespans)
                        min_makespan = np.min(all_makespans)
                        max_makespan = np.max(all_makespans)
                        std_makespan = np.std(all_makespans)
                        mean_exec_time = np.mean(all_execution_times)
                        mean_time_per_step = np.mean(all_execution_times_per_step)
                        std_time_per_step = np.std(all_execution_times_per_step)
                        print(f"  {policy.upper():<30}: mean={mean_makespan:>7.1f}, min={min_makespan:>7.1f}, max={max_makespan:>7.1f}, std={std_makespan:>6.1f}")
                        print(f"  {' ':<30}  exec_time={mean_exec_time:>6.2f}s, time/step={mean_time_per_step:>7.5f}±{std_time_per_step:<7.5f}s")
            else:
                print(f"\nPerformance Overview (mean delta makespan):")
                
                for policy in sorted(all_policies):
                    all_makespans = []
                    all_execution_times_per_step = []
                    for instance_result in self.instance_results.values():
                        results = instance_result.policy_results.get(policy, [])
                        if results:
                            makespans = [r.delta_makespan for r in results]
                            execution_times_per_step = [r.execution_time_per_step for r in results]
                            all_makespans.extend(makespans)
                            all_execution_times_per_step.extend(execution_times_per_step)
                    
                    if all_makespans:
                        mean_makespan = np.mean(all_makespans)
                        std_makespan = np.std(all_makespans)
                        mean_time_per_step = np.mean(all_execution_times_per_step)
                        std_time_per_step = np.std(all_execution_times_per_step)
                        print(f"  - {policy.capitalize()}: {mean_makespan:.2f} ± {std_makespan:.2f} (time/step: {mean_time_per_step:.6f} ± {std_time_per_step:.6f}s)")
        
        print(f"{'='*60}")
    
    def load_results(self, pickle_path: str) -> None:
        """Load previously saved results"""
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
            self.experiment_metadata = data['metadata']
            self.instance_results = data['instance_results']
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get all episode results as a pandas DataFrame"""
        all_episodes = []
        
        for instance_result in self.instance_results.values():
            for policy_name, results in instance_result.policy_results.items():
                for result in results:
                    all_episodes.append(asdict(result))
        
        return pd.DataFrame(all_episodes)
    
    def get_summary_dataframe(self) -> pd.DataFrame:
        """Get instance summary statistics as a pandas DataFrame"""
        summaries = []
        
        for instance_result in self.instance_results.values():
            summary = instance_result.get_all_statistics()
            summaries.append(summary)
        
        return pd.DataFrame(summaries)


def create_timestamped_output_dir(base_dir: str = "experiment_results") -> str:
    """Create a timestamped output directory"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_str = f"M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}W{dfjspt_params.time_window_size}"
    output_dir = f"{base_dir}/{config_str}_{timestamp}"
    # Ensure base directory exists
    Path(base_dir).mkdir(exist_ok=True, parents=True)
    return output_dir 