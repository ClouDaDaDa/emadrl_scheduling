#!/usr/bin/env python3
"""
Parallel test script for M50T20 global instance comparison.
This script demonstrates how to efficiently utilize multiple CPUs for testing.

Key Features:
- Parallel execution across 49 logical CPUs
- 4*3*7*5=420 total tests parallelized
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

# Configuration for M36T20 setup
CHECKPOINT_DIR = '/Users/dadada/Downloads/ema_rts_data/ray_results/M36T20W300/PPO_LocalSchedulingMultiAgentEnv_bbc71_00000_0_2025-05-31_22-44-19/checkpoint_000025'

from local_realtime_scheduling.test_global_parallel.compare_global_instances_parallel import (
    run_parallel_global_instance_comparison
)
from local_realtime_scheduling.result_analyzer import ResultAnalyzer
from local_realtime_scheduling.test_heuristic_combinations import TOP_HEURISTIC_COMBINATIONS

NUM_CPUS = 7


def test_parallel_performance():
    """Test to demonstrate parallel performance gains"""
    print("\n" + "=" * 80)
    print("PARALLEL PERFORMANCE DEMONSTRATION")
    print("=" * 80)

    # Small test to show parallelization benefits
    test_job_sizes = [50, 100]  # 2 job sizes
    test_instance_ids = [100, 101]  # 2 instances each
    test_policies = ['heuristic_MONR_PERIODIC_NEAREST_NEVER', 'heuristic_EDD_PERIODIC_NEAREST_NEVER']  # 2 policies
    test_episodes = 3  # 3 episodes each

    # Total: 2*2*2*3 = 24 tests
    expected_tests = len(test_job_sizes) * len(test_instance_ids) * len(test_policies) * test_episodes
    print(f"Running {expected_tests} tests in parallel...")

    start_time = time.time()

    # Initialize Ray for parallel execution
    if not ray.is_initialized():
        ray.init(num_cpus=8, local_mode=False)  # Use 8 CPUs for demo
        print("✓ Ray initialized with 8 CPUs for demonstration")

    try:
        result_collector = run_parallel_global_instance_comparison(
            checkpoint_dir=None,  # Skip MADRL for performance demo
            num_repeat=test_episodes,
            job_sizes=test_job_sizes,
            instance_ids=test_instance_ids,
            policies_to_test=test_policies,
            max_concurrent_tasks=8,
            detailed_log=True,
            save_snapshots=False  # Disabled for speed
        )

        total_time = time.time() - start_time

        print(f"\n{'=' * 60}")
        print(f"PERFORMANCE RESULTS")
        print(f"{'=' * 60}")
        print(f"Total tests: {expected_tests}")
        print(f"Total time: {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
        print(f"Average time per test: {total_time / expected_tests:.2f} seconds")
        print(f"Estimated serial time: ~{total_time * 8:.2f} seconds ({total_time * 8 / 60:.2f} minutes)")
        print(f"Speedup achieved: ~{8:.1f}x")
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


def run_full_M50T20_comparison():
    """Run the full M50T20 comparison with all policies and instances"""
    print("\n" + "=" * 80)
    print("FULL M50T20 PARALLEL COMPARISON")
    print("=" * 80)

    # Full test configuration as mentioned by user
    job_sizes = [50, 100, 150, 200]  # 4 job sizes
    instance_ids = [100, 101, 102]  # 3 instances each
    policies_to_test = [combo['name'] for combo in TOP_HEURISTIC_COMBINATIONS] + ['initial']  # 5 + 1 policies
    if CHECKPOINT_DIR and os.path.exists(CHECKPOINT_DIR):
        policies_to_test.append('madrl')  # Add MADRL if checkpoint available
    else:
        print(f"CHECKPOINT_DIR {CHECKPOINT_DIR} doesn't exist!")
    num_episodes = 5  # 5 episodes each

    # Calculate total tests: 4*3*7*5 = 420
    expected_tests = len(job_sizes) * len(instance_ids) * len(policies_to_test) * num_episodes

    print(f"Configuration:")
    print(f"  Job sizes: {job_sizes}")
    print(f"  Instance IDs: {instance_ids}")
    print(f"  Policies: {len(policies_to_test)} ({', '.join(policies_to_test)})")
    print(f"  Episodes per policy: {num_episodes}")
    print(f"  Total tests: {expected_tests}")
    print(f"  Expected serial time: ~{expected_tests * 2 / 60:.1f} minutes (assuming 2 min per test)")
    print(f"  Expected parallel time with {NUM_CPUS} CPUs: ~{expected_tests * 2 / NUM_CPUS / 60:.1f} minutes")

    # Initialize Ray for maximum parallel execution
    if not ray.is_initialized():
        ray.init(num_cpus=NUM_CPUS, local_mode=False)  # Use NUM_CPUS logical CPUs
        print(f"✓ Ray initialized with {NUM_CPUS} CPUs")

    start_time = time.time()

    try:
        result_collector = run_parallel_global_instance_comparison(
            checkpoint_dir=CHECKPOINT_DIR if 'madrl' in policies_to_test else None,
            num_repeat=num_episodes,
            job_sizes=job_sizes,
            instance_ids=instance_ids,
            policies_to_test=policies_to_test,
            max_concurrent_tasks=NUM_CPUS,  # Use all available CPUs
            detailed_log=True,
            save_snapshots=True
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
        print(f"  python test_global_comparison.py --load-results '{collector_pickle_path}'")

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


def run_custom_parallel_test():
    """Run a custom parallel test with user-specified parameters"""
    print("\n" + "=" * 80)
    print("CUSTOM PARALLEL TEST CONFIGURATION")
    print("=" * 80)

    # Customizable parameters - modify these as needed
    job_sizes = [100, 150]  # Test 2 job sizes
    instance_ids = [100, 101]  # Test 2 instances each
    policies_to_test = [
        'heuristic_MONR_PERIODIC_NEAREST_NEVER',  # Best heuristic
        'heuristic_EDD_PERIODIC_NEAREST_NEVER',  # Another good heuristic
        'initial'  # Initial schedule baseline
    ]
    num_episodes = 3  # 3 episodes each
    max_cpus = 16  # Use 16 CPUs

    # Add MADRL if checkpoint is available
    if CHECKPOINT_DIR and os.path.exists(CHECKPOINT_DIR):
        policies_to_test.append('madrl')
        print("✓ MADRL checkpoint found, including MADRL policy")
    else:
        print("⚠ MADRL checkpoint not found, skipping MADRL policy")

    expected_tests = len(job_sizes) * len(instance_ids) * len(policies_to_test) * num_episodes

    print(f"\nCustom Test Configuration:")
    print(f"  Job sizes: {job_sizes}")
    print(f"  Instance IDs: {instance_ids}")
    print(f"  Policies: {policies_to_test}")
    print(f"  Episodes per policy: {num_episodes}")
    print(f"  Max CPUs: {max_cpus}")
    print(f"  Total tests: {expected_tests}")

    # Confirm before proceeding
    response = input(f"\nProceed with {expected_tests} parallel tests? (y/n): ")
    if response.lower() != 'y':
        print("Test cancelled.")
        return

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=max_cpus, local_mode=False)
        print(f"✓ Ray initialized with {max_cpus} CPUs")

    start_time = time.time()

    try:
        result_collector = run_parallel_global_instance_comparison(
            checkpoint_dir=CHECKPOINT_DIR if 'madrl' in policies_to_test else None,
            num_repeat=num_episodes,
            job_sizes=job_sizes,
            instance_ids=instance_ids,
            policies_to_test=policies_to_test,
            max_concurrent_tasks=max_cpus,
            detailed_log=True,
            save_snapshots=False
        )

        total_time = time.time() - start_time

        print(f"\n{'=' * 60}")
        print(f"CUSTOM TEST COMPLETED!")
        print(f"{'=' * 60}")
        print(f"Execution time: {total_time / 60:.2f} minutes")
        print(f"Average time per test: {total_time / expected_tests:.2f} seconds")
        print(f"Results directory: {result_collector.output_dir}")

        # Quick analysis
        analyzer = ResultAnalyzer(result_collector=result_collector)
        policy_comparison = analyzer.compare_policies_overall()

        print(f"\nQuick Results Summary:")
        for policy, stats in policy_comparison.items():
            if 'mean_makespan' in stats:
                print(f"  {policy}: {stats['mean_makespan']:.2f} ± {stats['std_makespan']:.2f}")

    except Exception as e:
        print(f"❌ Error in custom test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ray.shutdown()


def main():
    """Main function with multiple test options"""
    import argparse

    parser = argparse.ArgumentParser(description='Parallel M50T20 Global Instance Testing')
    parser.add_argument('--mode', choices=['demo', 'full', 'custom'],
                        default='full',
                        help='Test mode: demo (quick), full (complete), or custom (configurable)')
    parser.add_argument('--cpus', type=int, default=NUM_CPUS,
                        help=f'Number of CPUs to use (default: {NUM_CPUS})')

    args = parser.parse_args()

    # # Update NUM_CPUS if specified
    # global NUM_CPUS
    # NUM_CPUS = args.cpus

    print("🚀 Parallel M50T20 Global Instance Testing")
    print(f"Available CPUs: {os.cpu_count()}")
    print(f"Ray will use: {args.cpus} CPUs")
    print(f"Checkpoint: {CHECKPOINT_DIR}")
    print(f"Checkpoint exists: {os.path.exists(CHECKPOINT_DIR) if CHECKPOINT_DIR else False}")

    if args.mode == 'demo':
        print("\n🎯 Running performance demonstration...")
        test_parallel_performance()

    elif args.mode == 'full':
        print("\n🎯 Running full M50T20 comparison...")
        print("⚠ This will run 420+ tests and may take 30-60 minutes even with parallelization")
        response = input("Continue? (y/n): ")
        if response.lower() == 'y':
            run_full_M50T20_comparison()
        else:
            print("Full test cancelled.")

    elif args.mode == 'custom':
        print("\n🎯 Running custom configuration...")
        run_custom_parallel_test()

    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()