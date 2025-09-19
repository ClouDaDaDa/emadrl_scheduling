#!/usr/bin/env python3
"""
Parallel test script for M36T20 local instance comparison.
This script demonstrates how to efficiently utilize multiple CPUs for testing local instances
with M36T20 configuration (36 machines, 20 transbots).

Key Features:
- Parallel execution across multiple logical CPUs
- Configurable test parameters
- Efficient resource utilization
- Progress monitoring and error handling
- Comprehensive result analysis
"""

import os
import sys
import ray
import time

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Configuration for M36T20 setup - update this path as needed
CHECKPOINT_DIR = '/Users/dadada/Downloads/ema_rts_data/ray_results/M36T20W300/PPO_LocalSchedulingMultiAgentEnv_bbc71_00000_0_2025-05-31_22-44-19/checkpoint_000025'

from local_realtime_scheduling.test_local_parallel.compare_local_instances_parallel import (
    run_parallel_local_instance_comparison
)
from local_realtime_scheduling.result_analyzer import ResultAnalyzer
from local_realtime_scheduling.test_heuristic_combinations import TOP_HEURISTIC_COMBINATIONS

NUM_CPUS = 7


def test_parallel_performance():
    """Test to demonstrate parallel performance gains"""
    print("\n" + "=" * 80)
    print("PARALLEL PERFORMANCE DEMONSTRATION - M36T20")
    print("=" * 80)

    # Small test to show parallelization benefits
    test_policies = ['heuristic_MONR_PERIODIC_NEAREST_NEVER', 'heuristic_EDD_PERIODIC_NEAREST_NEVER']  # 2 policies
    test_episodes = 3  # 3 episodes each
    test_max_instances = 6  # 6 instances

    # Total: 6*2*3 = 36 tests
    expected_tests = test_max_instances * len(test_policies) * test_episodes
    print(f"Running {expected_tests} tests in parallel...")

    start_time = time.time()

    # Initialize Ray for parallel execution
    if not ray.is_initialized():
        ray.init(num_cpus=12, local_mode=False)  # Use 12 CPUs for demo
        print("✓ Ray initialized with 12 CPUs for demonstration")

    try:
        result_collector = run_parallel_local_instance_comparison(
            checkpoint_dir=None,  # Skip MADRL for performance demo
            num_repeat=test_episodes,
            max_instances=test_max_instances,
            policies_to_test=test_policies,
            max_concurrent_tasks=12,
            detailed_log=True,
            instance_filter=None
        )

        total_time = time.time() - start_time

        print(f"\n{'=' * 60}")
        print(f"PERFORMANCE RESULTS")
        print(f"{'=' * 60}")
        print(f"Total tests: {expected_tests}")
        print(f"Total time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
        print(f"Average time per test: {total_time / expected_tests:.2f} seconds")
        print(f"Estimated serial time: ~{total_time * 12:.2f} seconds ({total_time * 12 / 60:.2f} minutes)")
        print(f"Speedup achieved: ~{12:.1f}x")
        print(f"{'=' * 60}")

        # Show some results
        print("\nSample Results:")
        basic_stats = ResultAnalyzer(result_collector=result_collector).get_basic_statistics()
        print(f"  Successful instances: {basic_stats['total_instances']}")
        print(f"  Successful episodes: {basic_stats['total_episodes']}")

    except Exception as e:
        print(f"Error in parallel execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()


def run_full_M36T20_comparison():
    """Run the full M36T20 comparison with all policies and instances"""
    print("\n" + "=" * 80)
    print("FULL M36T20 PARALLEL COMPARISON")
    print("=" * 80)

    # Full test configuration
    max_instances = 20  # Test 20 instances
    policies_to_test = [combo['name'] for combo in TOP_HEURISTIC_COMBINATIONS]  # 5 heuristic policies
    if CHECKPOINT_DIR and os.path.exists(CHECKPOINT_DIR):
        policies_to_test.append('madrl')  # Add MADRL if checkpoint available
    else:
        print(f"CHECKPOINT_DIR {CHECKPOINT_DIR} doesn't exist, skipping MADRL!")

    num_episodes = 5  # 5 episodes each

    # Calculate total tests: 20*6*5 = 600 (or 20*5*5 = 500 without MADRL)
    expected_tests = max_instances * len(policies_to_test) * num_episodes

    print(f"Configuration:")
    print(f"  Max instances: {max_instances}")
    print(f"  Policies: {len(policies_to_test)} ({', '.join(policies_to_test)})")
    print(f"  Episodes per policy: {num_episodes}")
    print(f"  Total tests: {expected_tests}")
    print(f"  Expected serial time: ~{expected_tests * 1.5 / 60:.1f} minutes (assuming 1.5 min per test)")
    print(f"  Expected parallel time with {NUM_CPUS} CPUs: ~{expected_tests * 1.5 / NUM_CPUS / 60:.1f} minutes")

    # Initialize Ray for maximum parallel execution
    if not ray.is_initialized():
        ray.init(num_cpus=NUM_CPUS, local_mode=False)  # Use NUM_CPUS logical CPUs
        print(f"✓ Ray initialized with {NUM_CPUS} CPUs")

    start_time = time.time()

    try:
        result_collector = run_parallel_local_instance_comparison(
            checkpoint_dir=CHECKPOINT_DIR if 'madrl' in policies_to_test else None,
            num_repeat=num_episodes,
            max_instances=max_instances,
            policies_to_test=policies_to_test,
            max_concurrent_tasks=NUM_CPUS,  # Use all available CPUs
            instance_filter={'max_jobs': 100},
            detailed_log=True,
        )

        total_time = time.time() - start_time

        print(f"\n{'=' * 80}")
        print(f"FULL COMPARISON COMPLETED!")
        print(f"{'=' * 80}")
        print(f"Total execution time: {total_time / 60:.2f} minutes")
        print(f"Estimated speedup: ~{NUM_CPUS:.0f}x compared to serial execution")
        print(f"Results saved to: {result_collector.output_dir}")

        # Save the ResultCollector object for future loading
        collector_pickle_path = result_collector.output_dir / "result_collector.pkl"
        print(f"\nSaving ResultCollector for future loading to: {collector_pickle_path}")
        with open(collector_pickle_path, 'wb') as f:
            import pickle
            pickle.dump(result_collector, f)
        print(f"✓ ResultCollector saved! You can load it later using:")
        print(f"  python test_local_comparison.py --load-results '{collector_pickle_path}'")

        # Generate comprehensive analysis
        print("\n📊 Generating comprehensive analysis...")
        analyzer = ResultAnalyzer(result_collector=result_collector)

        # Basic statistics
        basic_stats = analyzer.get_basic_statistics()
        print(f"\nExecution Summary:")
        print(f"  Total instances: {basic_stats['total_instances']}")
        print(f"  Total episodes: {basic_stats['total_episodes']}")
        print(f"  Average makespan: {basic_stats.get('avg_makespan', 'N/A')}")

        # Policy comparison
        policy_comparison = analyzer.compare_policies_overall()
        print(f"\nPolicy Performance (Mean Makespan):")
        for policy, stats in sorted(policy_comparison.items(), key=lambda x: x[1].get('mean_makespan', float('inf'))):
            if 'mean_makespan' in stats:
                print(f"  {policy.upper():<40}: {stats['mean_makespan']:>8.2f} ± {stats['std_makespan']:>6.2f}")

        # Generate visualizations and reports
        print(f"\n📈 Generating visualizations and reports...")
        analyzer.plot_policy_comparison_boxplot()
        analyzer.plot_performance_by_instance_size()
        analyzer.plot_execution_time_comparison()

        # Comprehensive report
        analyzer.generate_comprehensive_report(
            output_dir=result_collector.output_dir / "analysis_results"
        )

        print(f"\n🎉 Analysis complete! Check results in: {result_collector.output_dir}")

    except Exception as e:
        print(f"❌ Error in full comparison: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()


def run_focused_comparison():
    """Run a focused comparison on specific instance types"""
    print("\n" + "=" * 80)
    print("FOCUSED M36T20 COMPARISON")
    print("=" * 80)

    # Focused test configuration
    max_instances = 15  # Test 15 instances
    policies_to_test = [
        'heuristic_MONR_PERIODIC_NEAREST_NEVER',  # Best performing heuristic
        'heuristic_EDD_PERIODIC_NEAREST_NEVER',  # Good alternative
        'heuristic_SPT_PERIODIC_NEAREST_NEVER',  # Shortest processing time
    ]

    # Add MADRL if available
    if CHECKPOINT_DIR and os.path.exists(CHECKPOINT_DIR):
        policies_to_test.append('madrl')
        print("✓ MADRL checkpoint found, including MADRL policy")
    else:
        print("⚠ MADRL checkpoint not found, skipping MADRL policy")

    num_episodes = 5  # 5 episodes each

    # Focus on medium-sized instances
    instance_filter = {
        'min_jobs': 20,
        'max_jobs': 80,
        'min_ops': 50,
        'max_ops': 200
    }

    expected_tests = max_instances * len(policies_to_test) * num_episodes

    print(f"Focused Configuration:")
    print(f"  Max instances: {max_instances}")
    print(f"  Instance filter: {instance_filter}")
    print(f"  Policies: {policies_to_test}")
    print(f"  Episodes per policy: {num_episodes}")
    print(f"  Total tests: {expected_tests}")

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=NUM_CPUS, local_mode=False)
        print(f"✓ Ray initialized with {NUM_CPUS} CPUs")

    start_time = time.time()

    try:
        result_collector = run_parallel_local_instance_comparison(
            checkpoint_dir=CHECKPOINT_DIR if 'madrl' in policies_to_test else None,
            num_repeat=num_episodes,
            max_instances=max_instances,
            policies_to_test=policies_to_test,
            max_concurrent_tasks=NUM_CPUS,
            detailed_log=True,
            instance_filter=instance_filter
        )

        total_time = time.time() - start_time

        print(f"\n{'=' * 60}")
        print(f"FOCUSED COMPARISON COMPLETED!")
        print(f"{'=' * 60}")
        print(f"Execution time: {total_time / 60:.2f} minutes")
        print(f"Average time per test: {total_time / expected_tests:.2f} seconds")
        print(f"Results directory: {result_collector.output_dir}")

        # Detailed analysis
        analyzer = ResultAnalyzer(result_collector=result_collector)

        # Policy comparison
        policy_comparison = analyzer.compare_policies_overall()
        print(f"\nDetailed Results Summary:")
        for policy, stats in policy_comparison.items():
            if 'mean_makespan' in stats:
                print(f"  {policy}:")
                print(f"    Mean Makespan: {stats['mean_makespan']:.2f} ± {stats['std_makespan']:.2f}")
                print(f"    Episodes: {stats['num_episodes']}")
                print(f"    Success Rate: {stats.get('success_rate', 'N/A')}")

        # Generate focused analysis
        analyzer.plot_policy_comparison_boxplot()
        analyzer.generate_comprehensive_report(
            output_dir=result_collector.output_dir / "focused_analysis"
        )

    except Exception as e:
        print(f"❌ Error in focused comparison: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()


def main():
    """Main function with multiple test options"""
    import argparse

    parser = argparse.ArgumentParser(description='Parallel M36T20 Local Instance Testing')
    parser.add_argument('--mode', choices=['demo', 'full', 'focused'],
                        default='full',
                        help='Test mode: demo (quick), full (complete), or focused (medium instances)')
    parser.add_argument('--cpus', type=int, default=NUM_CPUS,
                        help=f'Number of CPUs to use (default: {NUM_CPUS})')

    args = parser.parse_args()

    # # Update global NUM_CPUS if specified
    # global NUM_CPUS
    # NUM_CPUS = args.cpus

    print("🚀 Parallel M36T20 Local Instance Testing")
    print(f"Available CPUs: {os.cpu_count()}")
    print(f"Ray will use: {args.cpus} CPUs")
    print(f"Checkpoint: {CHECKPOINT_DIR}")
    print(f"Checkpoint exists: {os.path.exists(CHECKPOINT_DIR) if CHECKPOINT_DIR else False}")

    if args.mode == 'demo':
        print("\n🎯 Running performance demonstration...")
        test_parallel_performance()

    elif args.mode == 'full':
        print("\n🎯 Running full M36T20 comparison...")
        print("⚠ This will run 600+ tests and may take 15-45 minutes even with parallelization")
        response = input("Continue? (y/n): ")
        if response.lower() == 'y':
            run_full_M36T20_comparison()
        else:
            print("Full test cancelled.")

    elif args.mode == 'focused':
        print("\n🎯 Running focused comparison on medium-sized instances...")
        run_focused_comparison()

    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()