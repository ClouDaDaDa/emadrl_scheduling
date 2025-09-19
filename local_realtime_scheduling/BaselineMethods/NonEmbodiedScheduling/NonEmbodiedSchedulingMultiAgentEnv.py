"""
Non-Embodied Multi-Agent Scheduling Environment

This environment implements a disembodied scheduling approach as a baseline comparison
to the embodied scheduling environment. Key differences:
1. Event-driven time progression instead of step-by-step simulation
2. Abstract resource allocation
3. Global/aggregate observations instead of local views
4. Fixed processing/transport times without reliability/battery/congestion effects
5. Logical feasibility checks only (no spatial/energy constraints)
"""

import numpy as np
import random
import pickle
import os
from typing import Dict, List, Optional, Tuple, Any
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)

from System.FactoryInstance import FactoryInstance
from configs import dfjspt_params
from local_realtime_scheduling.Environment.ExecutionResult import LocalResult, Local_Job_result, Operation_result
from local_realtime_scheduling.InterfaceWithGlobal.divide_global_schedule_to_local_from_pkl import LocalSchedule, Local_Job_schedule

# Logging level control
# Set to "DEBUG" to enable all debug output, "INFO" for minimal output
LOG_LEVEL = "INFO"  # Default to INFO level

def _add_realistic_time_variation(nominal_time: float) -> float:
    """
    Add realistic time variation to nominal processing/transport times.
    
    Args:
        nominal_time: The standard/nominal time
        
    Returns:
        Actual time with realistic variation (minimum 3.0)
    """
    # Add 0-15% random increase to nominal time
    variation_factor = 1.0 + random.uniform(0.0, 0.15)
    varied_time = nominal_time * variation_factor
    
    # Add ±5 time units of noise
    noise = random.uniform(-5.0, 5.0)
    actual_time = varied_time + noise
    
    # Ensure minimum time of 3.0
    return max(3.0, actual_time)

def debug_print(message: str):
    """Print debug message only if LOG_LEVEL is set to DEBUG."""
    if LOG_LEVEL == "DEBUG":
        print(f"DEBUG: {message}")

def info_print(message: str):
    """Print info message if LOG_LEVEL is INFO or DEBUG."""
    if LOG_LEVEL in ["INFO", "DEBUG"]:
        print(message)

class NonEmbodiedSchedulingMultiAgentEnv(MultiAgentEnv):
    """
    A Non-Embodied Multi-agent Environment for Production Scheduling.
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, config):
        """
        Initialize the non-embodied scheduling environment.
        
        Args:
            config: Configuration dictionary containing:
                - n_machines: Number of machines
                - n_transbots: Number of transbots
                - factory_instance_seed: Seed for factory instance
        """
        super().__init__()

        # Initialize parameters
        self.num_machines = config["n_machines"]
        self.num_transbots = config["n_transbots"]
        
        # Create simplified factory instance (only for basic layout info)
        self.factory_instance = FactoryInstance(
            seed=config["factory_instance_seed"],
            n_machines=self.num_machines,
            n_transbots=self.num_transbots,
        )
        
        # Get transport time matrix (fixed values, no A* or congestion)
        self.transport_time_matrix = self.factory_instance.factory_graph.unload_transport_time_matrix
        
        # Environment state
        self.current_time = 0.0
        self.event_step = 0
        self.max_event_steps = 1000  # Maximum number of decision events
        
        # Decision stage tracking for two-stage process
        self.decision_stage = "machine"  # "machine" or "transbot"
        self.pending_machine_assignments = {}  # machine_id -> job_id (tentative assignments)
        self.pending_transbot_assignments = {}  # transbot_id -> job_id (tentative assignments)
        
        # Job and operation tracking
        self.num_jobs = 0
        self.total_operations_in_window = 0
        self.remaining_operations = 0
        
        # Machine states (simplified)
        self.machine_states = {}  # machine_id -> {'status': idle/busy, 'current_job': job_id, 'finish_time': float}
        self.machine_job_assignments = {}  # machine_id -> job_id (current assignment)
        
        # Transbot states (simplified - no spatial movement, battery, etc.)
        self.transbot_states = {}  # transbot_id -> {'status': idle/busy, 'current_job': job_id, 'finish_time': float}
        self.transbot_job_assignments = {}  # transbot_id -> job_id (current assignment)
        
        # Job states (simplified)
        self.job_states = {}  # job_id -> {'current_op': int, 'status': ready/processing/completed, 'assigned_machine': int, 'finish_time': float}
        
        # Define agent IDs (machines and transbots, but transbots have simplified actions)
        self.machine_agents = [f"machine{i}" for i in range(self.num_machines)]
        self.transbot_agents = [f"transbot{i}" for i in range(self.num_transbots)]
        self.agents = self.possible_agents = self.machine_agents + self.transbot_agents
        
        # Fixed observation dimensions
        self.max_jobs_in_obs = 20  # Fixed maximum jobs in observation
        self.global_features_dim = 10  # Global statistics features
        self.job_features_dim = 8  # Per-job features
        
        # Define observation and action spaces
        self.observation_spaces = {}
        self.action_spaces = {}
        
        # Machine agents observation and action spaces
        for machine_agent_id in self.machine_agents:
            self.observation_spaces[machine_agent_id] = spaces.Dict({
                "action_mask": spaces.Box(
                    0, 1, shape=(self.max_jobs_in_obs + 1,), dtype=np.int32  # +1 for "do nothing"
                ),
                "observation": spaces.Dict({
                    "global_features": spaces.Box(
                        low=-float('inf'), high=float('inf'),
                        shape=(self.global_features_dim,), dtype=np.float32
                    ),
                    "job_features": spaces.Box(
                        low=-float('inf'), high=float('inf'),
                        shape=(self.max_jobs_in_obs, self.job_features_dim), dtype=np.float32
                    ),
                })
            })
            
            # Action: select a job from the list or do nothing
            self.action_spaces[machine_agent_id] = spaces.Discrete(self.max_jobs_in_obs + 1)
        
        # Transbot agents observation and action spaces (simplified)
        for transbot_agent_id in self.transbot_agents:
            self.observation_spaces[transbot_agent_id] = spaces.Dict({
                "action_mask": spaces.Box(
                    0, 1, shape=(self.max_jobs_in_obs + 1,), dtype=np.int32  # +1 for "do nothing"
                ),
                "observation": spaces.Dict({
                    "global_features": spaces.Box(
                        low=-float('inf'), high=float('inf'),
                        shape=(self.global_features_dim,), dtype=np.float32
                    ),
                    "job_features": spaces.Box(
                        low=-float('inf'), high=float('inf'),
                        shape=(self.max_jobs_in_obs, self.job_features_dim), dtype=np.float32
                    ),
                })
            })
            
            # Action: select a job to transport or do nothing (no charging in non-embodied)
            self.action_spaces[transbot_agent_id] = spaces.Discrete(self.max_jobs_in_obs + 1)
        
        # Results tracking
        self.local_result = None
        self.local_result_file = None
        self.initial_estimated_makespan = None
        self.time_upper_bound = None
        
        # Episode tracking
        self.terminateds = set()
        self.truncateds = set()
        self.rewards = {}
        self.resetted = False
        
        # Initialize time tracking attributes for callback compatibility
        self.current_time_before_step = 0.0
        self.current_time_after_step = 0.0
        
        # # Print transport time matrix for debugging
        # self._print_transport_time_matrix()

    def _print_transport_time_matrix(self):
        """Print the transport time matrix for debugging purposes."""
        print("Transport time matrix:")
        print(self.transport_time_matrix)
        print(f"Min transport time: {np.min(self.transport_time_matrix[self.transport_time_matrix > 0])}")
        print(f"Max transport time: {np.max(self.transport_time_matrix)}")

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.
        
        Args:
            seed: Random seed
            options: Options dictionary containing:
                - factory_instance: FactoryInstance
                - scheduling_instance: SchedulingInstance  
                - local_schedule: LocalSchedule
                - start_t_for_curr_time_window: float
                - local_result_file: str (optional)
        
        Returns:
            Tuple of (observations, infos)
        """
        super().reset(seed=seed)
        
        if options is None:
            raise ValueError("Options must be provided with scheduling instances!")
        
        # Reset state
        self.resetted = True
        self.terminateds.clear()
        self.truncateds.clear()
        self.rewards = {agent_id: 0.0 for agent_id in self.agents}
        
        # Load problem instance
        self.factory_instance = options["factory_instance"]
        self.scheduling_instance = options["scheduling_instance"]
        self.local_schedule = options["local_schedule"]
        self.current_time = options["start_t_for_curr_time_window"]
        self.local_result_file = options.get("local_result_file", None)
        
        # Initialize result tracking
        self.local_result = LocalResult()
        self.local_result.time_window_start = self.current_time
        
        # Get time bounds
        self.initial_estimated_makespan = self.local_schedule.local_makespan + (
            self.current_time - self.local_schedule.time_window_start
        )
        self.time_upper_bound = dfjspt_params.episode_time_upper_bound
        
        # Initialize job, machine and transbot states
        self._initialize_states()
        
        # Reset counters and decision stage
        self.max_event_steps = max(self.max_event_steps, self.total_operations_in_window * 10)
        self.event_step = 0
        self.decision_stage = "machine"
        self.pending_machine_assignments = {}
        self.pending_transbot_assignments = {}
        
        # Normalize rewards by total operations to keep scale stable across instances
        self.reward_scale = 1.0 / max(1, self.total_operations_in_window)
        
        # Initialize time tracking for callback compatibility
        self.current_time_before_step = self.current_time
        self.current_time_after_step = self.current_time
        
        # Generate initial observations for machine agents (first stage)
        observations = self._get_observations_for_stage("machine")
        infos = self._get_infos()
        
        return observations, infos

    def _initialize_states(self):
        """Initialize machine and job states for the episode."""
        # Initialize machine states
        self.machine_states = {}
        self.machine_job_assignments = {}
        
        for i in range(self.num_machines):
            self.machine_states[i] = {
                'status': 'idle',
                'current_job': None,
                'finish_time': self.current_time,
                'total_processing_time': 0.0,
                'completed_jobs': 0
            }
            self.machine_job_assignments[i] = None
        
        # Initialize transbot states
        self.transbot_states = {}
        self.transbot_job_assignments = {}
        
        for i in range(self.num_transbots):
            self.transbot_states[i] = {
                'status': 'idle',
                'current_job': None,
                'finish_time': self.current_time,
                'total_transport_time': 0.0,
                'completed_transports': 0
            }
            self.transbot_job_assignments[i] = None
        
        # Initialize job states
        self.job_states = {}
        self.num_jobs = len(self.local_schedule.jobs)
        self.total_operations_in_window = 0
        
        for job_id, local_schedule_job in self.local_schedule.jobs.items():
            scheduling_job = self.scheduling_instance.jobs[job_id]
            
            # Count operations in current window
            ops_in_window = len([op_id for op_id in local_schedule_job.operations 
                               if local_schedule_job.operations[op_id].is_current_window])
            self.total_operations_in_window += ops_in_window
            
            # Find the first operation that needs to be processed in the current window
            # This is the key fix: we need to find the actual operation index in the current window
            first_op_in_window = None
            for op_id in sorted(local_schedule_job.operations.keys()):
                if local_schedule_job.operations[op_id].is_current_window:
                    first_op_in_window = op_id
                    break
            
            # Determine the correct current operation to process
            if first_op_in_window is None:
                # No operations in current window, job is completed
                current_op = len(scheduling_job.operations_matrix)
                initial_status = 'completed'
            else:
                # Use the first operation in the current window as the current operation
                current_op = first_op_in_window
                
                # Check if this operation is actually the one that should be processed
                # by comparing with the scheduling_job's current_processing_operation
                if (scheduling_job.current_processing_operation < len(scheduling_job.operations_matrix) and
                    scheduling_job.current_processing_operation <= current_op):
                    # The scheduling_job's current_processing_operation is valid and <= our window's first op
                    current_op = scheduling_job.current_processing_operation
                
                # Determine status based on whether this operation is ready to be processed
                if current_op < len(scheduling_job.operations_matrix):
                    initial_status = 'ready'
                else:
                    initial_status = 'completed'
            
            # Store the corrected current operation in job_states
            self.job_states[job_id] = {
                'current_op': current_op,
                'total_ops': len(scheduling_job.operations_matrix),
                'status': initial_status,
                'assigned_machine': None,
                'start_time': None,
                'finish_time': None,
                'processing_times': [],  # List of (machine_id, processing_time) for current op
                'transport_times': [],   # List of transport times to each machine
                'progress': 0.0  # Completion progress [0, 1]
            }
            
            # Calculate processing times for current operation (fixed, no reliability effects)
            if current_op < len(scheduling_job.operations_matrix):
                for machine_id in range(self.num_machines):
                    processing_time = scheduling_job.operations_matrix[current_op, machine_id]
                    self.job_states[job_id]['processing_times'].append((machine_id, processing_time))
                    
                    # Calculate transport time from current job location to target machine
                    # For non-embodied version, assume job starts at warehouse (location 0)
                    # or use the machine where the previous operation was completed
                    if current_op == 0:
                        # First operation: transport from warehouse (location 0) to target machine
                        from_location = 0
                    else:
                        # Subsequent operations: transport from previous machine to target machine
                        # For now, assume previous operation was on machine 0 (will be updated dynamically)
                        from_location = 0
                    
                    transport_time = self.transport_time_matrix[from_location, machine_id]
                    
                    self.job_states[job_id]['transport_times'].append(transport_time)
            
            # Create job result tracking
            job_result = Local_Job_result(job_id=job_id)
            self.local_result.add_job_result(job_result)
            
            # Add operation results for current window operations
            for operation_id in local_schedule_job.operations:
                if local_schedule_job.operations[operation_id].is_current_window:
                    job_result.add_operation_result(Operation_result(
                        job_id=job_id,
                        operation_id=operation_id,
                    ))
        
        # Count only operations that are ready to be processed
        self.remaining_operations = self.total_operations_in_window
        
        # Debug: Print initialization summary
        ready_jobs = [jid for jid, js in self.job_states.items() if js['status'] == 'ready']
        waiting_jobs = [jid for jid, js in self.job_states.items() if js['status'] == 'waiting']
        completed_jobs = [jid for jid, js in self.job_states.items() if js['status'] == 'completed']
        debug_print(f"Initialization complete. Total operations: {self.total_operations_in_window}")
        debug_print(f"Ready jobs: {ready_jobs}")
        debug_print(f"Waiting jobs: {waiting_jobs}")
        debug_print(f"Completed jobs: {completed_jobs}")
        debug_print(f"Job states: {[(jid, js['status'], js['current_op'], js['total_ops']) for jid, js in self.job_states.items()]}")
        debug_print(f"Remaining operations initialized to: {self.remaining_operations}")
        debug_print(f"Max event steps set to: {self.max_event_steps}")

    def step(self, action_dict):
        """
        Execute one step of the two-stage decision process.
        
        Stage 1 (machine): Machine agents make decisions, jobs are assigned to machines
        Stage 2 (transbot): Transbot agents make decisions, transport tasks are assigned
        Then: Event-driven simulation advances time to next completion
        
        Args:
            action_dict: Dictionary of agent_id -> action mappings
            
        Returns:
            Tuple of (observations, rewards, terminated, truncated, infos)
        """
        observations, terminated, truncated, infos = {}, {}, {}, {}
        
        # Initialize rewards
        step_rewards = {agent_id: 0.0 for agent_id in self.agents}

        # if self.event_step > 500:
        #     print(self.remaining_operations)
        
        if self.decision_stage == "machine":
            # Stage 1: Process machine decisions and assign jobs to machines
            self._process_machine_actions(action_dict, step_rewards)
            
            # Resolve conflicts and finalize machine assignments
            self._finalize_machine_assignments(step_rewards)
            
            # Switch to transbot stage
            self.decision_stage = "transbot"
            
            # Generate observations for transbot agents only
            observations = self._get_observations_for_stage("transbot")
            
            # Set termination flags for all agents (not terminated/truncated yet)
            for agent_id in self.agents:
                terminated[agent_id] = False
                truncated[agent_id] = False
            
            # Set global termination flags
            terminated["__all__"] = False
            truncated["__all__"] = False
            
        elif self.decision_stage == "transbot":
            # Stage 2: Process transbot decisions and assign transport tasks
            self._process_transbot_actions(action_dict, step_rewards)
            
            # Resolve conflicts and finalize transbot assignments
            self._finalize_transbot_assignments(step_rewards)
            
            # Execute event-driven simulation: advance time to next completion
            completion_occurred = self._advance_to_next_event(step_rewards)
            
            # Validate scheduling constraints after each simulation step
            self._validate_scheduling_constraints()
            
            # Update time tracking for callback compatibility
            self.current_time_before_step = self.current_time_after_step
            self.current_time_after_step = self.current_time
            
            self.event_step += 1
            
            # Switch back to machine stage for next decision cycle
            self.decision_stage = "machine"
            
            # Check termination conditions
            is_terminated = self._check_terminated()
            is_truncated = self._check_truncated() and not is_terminated
            
            # Generate observations for next step
            if not (is_terminated or is_truncated):
                observations = self._get_observations_for_stage("machine")
            else:
                # Provide dummy observations for terminated/truncated episode
                observations = {agent_id: self._get_dummy_observation() for agent_id in self.agents}
                
                # Add completion bonus or penalty (improved values)
                if is_terminated:
                    # Base completion bonus
                    for agent_id in self.agents:
                        step_rewards[agent_id] += 5.0 * self.reward_scale  # Completion bonus
                    
                    # Efficiency bonus based on makespan improvement
                    actual_makespan = self.current_time - self.local_result.time_window_start
                    estimated_makespan = self.initial_estimated_makespan - self.local_result.time_window_start
                    if estimated_makespan > 0:
                        efficiency_ratio = max(0, (estimated_makespan - actual_makespan) / estimated_makespan)
                        efficiency_bonus = min(2.0, efficiency_ratio * 5.0)  # Max 2.0 bonus
                        for agent_id in self.agents:
                            step_rewards[agent_id] += efficiency_bonus * self.reward_scale
                    
                    self._log_episode_completion()
                    # Ensure actual_local_makespan is set for callback compatibility
                    if not hasattr(self.local_result, 'actual_local_makespan') or self.local_result.actual_local_makespan is None:
                        self.local_result.actual_local_makespan = self.current_time
                elif is_truncated:  # Only apply truncation penalty if not terminated
                    for agent_id in self.agents:
                        step_rewards[agent_id] -= 2.0 * self.reward_scale  # Reduced truncation penalty
            
            # Set termination flags for all agents
            for agent_id in self.agents:
                terminated[agent_id] = is_terminated
                truncated[agent_id] = is_truncated
                if is_terminated:
                    self.terminateds.add(agent_id)
                if is_truncated:
                    self.truncateds.add(agent_id)
            
            # Set global termination flags (termination takes priority)
            terminated["__all__"] = is_terminated
            truncated["__all__"] = is_truncated and not is_terminated  # Only truncate if not terminated
        
        # Update cumulative rewards
        for agent_id in self.agents:
            self.rewards[agent_id] += step_rewards[agent_id]
        
        infos = self._get_infos()
        
        # Return per-step rewards (not cumulative)
        return observations, step_rewards, terminated, truncated, infos

    def _get_observations_for_stage(self, stage: str) -> Dict[str, Any]:
        """Generate observations for agents in the specified stage only."""
        observations = {}
        available_jobs = self._get_available_jobs(stage)
        
        # Calculate global features (same for all agents)
        global_features = self._calculate_global_features()
        
        # Calculate job features (same for all agents in non-embodied version)
        job_features = self._calculate_job_features(available_jobs)
        
        if stage == "machine":
            # Only provide observations to machine agents
            for agent_id in self.machine_agents:
                machine_id = int(agent_id.lstrip("machine"))
                
                # Create action mask (logical feasibility only)
                action_mask = self._create_machine_action_mask(machine_id, available_jobs)
                
                observations[agent_id] = {
                    "action_mask": action_mask,
                    "observation": {
                        "global_features": global_features,
                        "job_features": job_features,
                    }
                }
                
        elif stage == "transbot":
            # Only provide observations to transbot agents
            for agent_id in self.transbot_agents:
                transbot_id = int(agent_id.lstrip("transbot"))
                
                # Create action mask for transbot
                action_mask = self._create_transbot_action_mask(transbot_id, available_jobs)
                
                observations[agent_id] = {
                    "action_mask": action_mask,
                    "observation": {
                        "global_features": global_features,
                        "job_features": job_features,
                    }
                }
        
        return observations

    def _finalize_machine_assignments(self, step_rewards):
        """Resolve conflicts and finalize machine assignments."""
        # Track which jobs have been assigned to avoid conflicts
        assigned_jobs = set()
        
        debug_print(f"Finalizing machine assignments. Pending: {self.pending_machine_assignments}")
        
        # For now, use simple conflict resolution: first-come-first-served
        # In a more sophisticated implementation, this could use auction mechanisms
        for machine_id, job_id in self.pending_machine_assignments.items():
            # Check if machine is available and job hasn't been assigned to another machine
            if (self.machine_job_assignments[machine_id] is None) and job_id not in assigned_jobs:
                # Make the assignment
                debug_print(f"Assigning job {job_id} to machine {machine_id}")
                self._assign_job_to_machine(job_id, machine_id)
                assigned_jobs.add(job_id)
            else:
                debug_print(f"Skipping assignment of job {job_id} to machine {machine_id}")
        
        debug_print(f"Finalized machine assignments: {assigned_jobs}")
        
        # Clear pending assignments
        self.pending_machine_assignments = {}

    def _finalize_transbot_assignments(self, step_rewards):
        """Resolve conflicts and finalize transbot assignments."""
        # Track which jobs have been assigned to avoid conflicts
        assigned_jobs = set()
        
        debug_print(f"Finalizing transbot assignments. Pending: {self.pending_transbot_assignments}")
        
        # For now, use simple conflict resolution: first-come-first-served
        for transbot_id, job_id in self.pending_transbot_assignments.items():
            # Check if transbot is available and job hasn't been assigned to another transbot
            if (self.transbot_job_assignments[transbot_id] is None) and job_id not in assigned_jobs:
                # Make the assignment
                debug_print(f"Assigning job {job_id} to transbot {transbot_id}")
                self._assign_job_to_transbot(job_id, transbot_id)
                assigned_jobs.add(job_id)
            else:
                debug_print(f"Skipping assignment of job {job_id} to transbot {transbot_id}")
        
        debug_print(f"Finalized transbot assignments: {assigned_jobs}")
        
        # Handle jobs that don't need transport (already at target machine)
        # These jobs can start processing immediately
        for job_id, job_state in self.job_states.items():
            if (job_state['status'] == 'assigned' and 
                job_state['assigned_machine'] is not None and
                (not hasattr(job_state, 'assigned_transbot') or job_state.get('assigned_transbot') is None)):
                
                # Check if transport is needed based on operation_timing
                current_op = job_state['current_op']
                needs_transport = True
                if ('operation_timing' in job_state and 
                    current_op in job_state['operation_timing'] and
                    not job_state['operation_timing'][current_op].get('needs_transport', True)):
                    needs_transport = False
                
                if not needs_transport:
                    # No transport needed, start processing immediately
                    debug_print(f"Job {job_id} doesn't need transport, starting processing immediately")
                    self._start_processing_without_transport(job_id)
                else:
                    # Find an available transbot for this job
                    for transbot_id, transbot_state in self.transbot_states.items():
                        if (transbot_state['status'] == 'idle' and 
                            (transbot_id not in self.transbot_job_assignments or 
                             self.transbot_job_assignments[transbot_id] is None)):
                            
                            debug_print(f"Auto-assigning transbot {transbot_id} to job {job_id}")
                            self._assign_job_to_transbot(job_id, transbot_id)
                            break
        
        # Clear pending assignments
        self.pending_transbot_assignments = {}

    def _process_machine_actions(self, action_dict, step_rewards):
        """
        Process machine assignment actions and store tentative assignments.
        
        Args:
            action_dict: Dictionary of machine actions
            step_rewards: Dictionary to accumulate step rewards
        """
        available_jobs = self._get_available_jobs("machine")
        
        # Clear previous tentative assignments
        self.pending_machine_assignments = {}
        
        # Process each machine's action
        for agent_id, action in action_dict.items():
            if not agent_id.startswith("machine"):
                continue
                
            machine_id = int(agent_id.lstrip("machine"))
            machine_state = self.machine_states[machine_id]
            
            # Skip if machine is still busy
            if machine_state['status'] == 'busy':
                continue
                
            # Handle "do nothing" action
            if action >= self.max_jobs_in_obs:
                continue
                
            # Get job from available jobs list
            if action >= len(available_jobs):
                continue  # Invalid action, skip
                
            job_id = available_jobs[action]
            job_state = self.job_states.get(job_id)
            
            if job_state is None or job_state['status'] != 'ready':
                continue  # Job not available
                
            # Check if job can be processed by this machine (logical feasibility only)
            if not self._can_machine_process_job(machine_id, job_id):
                step_rewards[agent_id] -= 0.1 * self.reward_scale  # Small fixed penalty for invalid action
                continue
                
            # Check if job is already assigned to another machine
            if job_state['assigned_machine'] is not None:
                step_rewards[agent_id] -= 0.1 * self.reward_scale  # Small fixed penalty for conflict
                continue
                
            # Store tentative assignment (will be resolved in _finalize_machine_assignments)
            debug_print(f"Machine {machine_id} tentatively assigned to job {job_id}")
            self.pending_machine_assignments[machine_id] = job_id
    
    def _process_transbot_actions(self, action_dict, step_rewards):
        """
        Process transbot assignment actions and store tentative assignments.
        
        Args:
            action_dict: Dictionary of transbot actions
            step_rewards: Dictionary to accumulate step rewards
        """
        available_jobs = self._get_available_jobs("transbot")
        
        # Clear previous tentative assignments
        self.pending_transbot_assignments = {}
        
        # Process each transbot's action
        for agent_id, action in action_dict.items():
            if not agent_id.startswith("transbot"):
                continue
                
            transbot_id = int(agent_id.lstrip("transbot"))
            transbot_state = self.transbot_states[transbot_id]
            
            # Skip if transbot is still busy
            if transbot_state['status'] == 'busy':
                continue
                
            # Handle "do nothing" action
            if action >= self.max_jobs_in_obs:
                continue
                
            # Get job from available jobs list
            if action >= len(available_jobs):
                continue  # Invalid action, skip
                
            job_id = available_jobs[action]
            job_state = self.job_states.get(job_id)
            
            if job_state is None or job_state['status'] != 'assigned':
                continue  # Job not available
                
            # Check if job can be transported by this transbot (logical feasibility only)
            if not self._can_transbot_transport_job(transbot_id, job_id):
                step_rewards[agent_id] -= 0.1 * self.reward_scale  # Small fixed penalty for invalid action
                continue
                
            # Check if job is already assigned to another transbot
            if hasattr(job_state, 'assigned_transbot') and job_state.get('assigned_transbot') is not None:
                step_rewards[agent_id] -= 0.1 * self.reward_scale  # Small fixed penalty for conflict
                continue
            
            # Check timing constraint: job must be assigned to a machine first
            if job_state['assigned_machine'] is None:
                step_rewards[agent_id] -= 0.1 * self.reward_scale  # Small fixed penalty for invalid action
                continue
                
            # Store tentative assignment (will be resolved in _finalize_transbot_assignments)
            debug_print(f"Transbot {transbot_id} tentatively assigned to job {job_id}")
            self.pending_transbot_assignments[transbot_id] = job_id

    def _get_available_jobs(self, stage: str = "machine") -> List[int]:
        """Get list of jobs available for assignment, sorted by priority."""
        available_jobs = []
        
        if stage == "machine":
            # For machine stage: jobs that are ready and not assigned to any machine
            for job_id, job_state in self.job_states.items():
                if (job_state['status'] == 'ready' and 
                    job_state['assigned_machine'] is None and
                    job_state['current_op'] < job_state['total_ops']):
                    available_jobs.append(job_id)
                    debug_print(f"Job {job_id} available for machine assignment (status: {job_state['status']}, current_op: {job_state['current_op']}, total_ops: {job_state['total_ops']})")
                else:
                    debug_print(f"Job {job_id} NOT available for machine assignment (status: {job_state['status']}, assigned_machine: {job_state['assigned_machine']}, current_op: {job_state['current_op']}, total_ops: {job_state['total_ops']})")
        elif stage == "transbot":
            # For transbot stage: jobs that are assigned to machines but not yet transported
            for job_id, job_state in self.job_states.items():
                if (job_state['status'] == 'assigned' and 
                    job_state['assigned_machine'] is not None and
                    (not hasattr(job_state, 'assigned_transbot') or job_state.get('assigned_transbot') is None) and
                    job_state['current_op'] < job_state['total_ops']):  # Only incomplete jobs
                    available_jobs.append(job_id)
        
        # Debug: Print available jobs for current stage
        debug_print(f"Stage {stage}, Available jobs: {available_jobs}")
        debug_print(f"Job states: {[(jid, js['status'], js.get('assigned_machine'), js['current_op']) for jid, js in self.job_states.items()]}")
        
        # Sort by simple priority: shortest total processing time first
        def job_priority(job_id):
            job_state = self.job_states[job_id]
            current_op = job_state['current_op']
            total_ops = job_state['total_ops']
            
            # Check if current operation is valid and has processing times
            if current_op < total_ops and job_state['processing_times']:
                # Find minimum processing time > 0
                valid_processing_times = [pt for mid, pt in job_state['processing_times'] if pt > 0]
                if valid_processing_times:
                    min_processing_time = min(valid_processing_times)
                    min_transport_time = min(job_state['transport_times']) if job_state['transport_times'] else 1.0
                    return min_processing_time + min_transport_time
            
            # Return high priority for completed jobs or jobs without valid processing times
            return float('inf')
        
        available_jobs.sort(key=job_priority)
        
        # Limit to maximum observation size
        return available_jobs[:self.max_jobs_in_obs]

    def _can_machine_process_job(self, machine_id: int, job_id: int) -> bool:
        """Check if machine can process job (logical feasibility only)."""
        job_state = self.job_states[job_id]
        
        # Check if machine can handle current operation
        for mid, processing_time in job_state['processing_times']:
            if mid == machine_id and processing_time > 0:
                result = True
                debug_print(f"_can_machine_process_job({machine_id}, {job_id}) = {result} (processing_time: {processing_time})")
                return result
        
        result = False
        debug_print(f"_can_machine_process_job({machine_id}, {job_id}) = {result}")
        debug_print(f"  - processing_times: {job_state['processing_times']}")
        return result

    def _assign_job_to_machine(self, job_id: int, machine_id: int):
        """Assign job to machine (tentative assignment, processing starts after transport)."""
        job_state = self.job_states[job_id]
        machine_state = self.machine_states[machine_id]
        
        # Get processing time for this machine
        processing_time = 0
        for mid, pt in job_state['processing_times']:
            if mid == machine_id:
                processing_time = pt
                break
        
        # Apply realistic time variation to processing time
        actual_processing_time = _add_realistic_time_variation(processing_time)
        
        # Update states - job is assigned to machine but not yet processing
        job_state['assigned_machine'] = machine_id
        job_state['status'] = 'assigned'  # Changed from 'processing' to 'assigned'
        job_state['processing_time'] = actual_processing_time
        
        debug_print(f"Assigned job {job_id} to machine {machine_id}, status: {job_state['status']}")
        
        # Store operation-level timing for Gantt chart with improved structure
        current_op = job_state['current_op']
        if 'operation_timing' not in job_state:
            job_state['operation_timing'] = {}
        
        # Check if transport is needed (compare with previous operation's machine)
        needs_transport = True
        if current_op > 0 and 'operation_timing' in job_state and (current_op - 1) in job_state['operation_timing']:
            prev_machine = job_state['operation_timing'][current_op - 1].get('machine_id')
            if prev_machine == machine_id:
                needs_transport = False  # Same machine, no transport needed
        
        # Initialize operation timing with improved structure
        job_state['operation_timing'][current_op] = {
            'transbot_id': None,  # Will be set when transbot is assigned
            'transport_start_time': None,  # Will be set when transport starts
            'transport_finish_time': None,  # Will be set when transport starts
            'machine_id': machine_id,
            'processing_start_time': None,  # Will be set when processing starts
            'processing_finish_time': None,  # Will be set when processing starts
            'needs_transport': needs_transport  # Flag indicating if transport is required
        }
        
        # Machine is not yet busy (will become busy when processing starts)
        # machine_state remains 'idle' until processing actually starts
        
        self.machine_job_assignments[machine_id] = job_id
    
    def _assign_job_to_transbot(self, job_id: int, transbot_id: int):
        """Assign job to transbot for transport and start transport process."""
        job_state = self.job_states[job_id]
        transbot_state = self.transbot_states[transbot_id]
        
        # Calculate transport time: unload travel + loaded travel
        # For non-embodied version, use simplified calculation
        machine_id = job_state['assigned_machine']
        
        # Determine the current location of the job
        current_op = job_state['current_op']
        if current_op == 0:
            # First operation: job starts at warehouse (location 0)
            from_location = 0
        else:
            # Subsequent operations: transport from previous machine to target machine
            if ('operation_timing' in job_state and 
                (current_op - 1) in job_state['operation_timing']):
                prev_machine = job_state['operation_timing'][current_op - 1]['machine_id']
                from_location = prev_machine
            else:
                # Fallback: assume previous operation was on machine 0
                from_location = 0
        
        # Get transport time from transport time matrix
        nominal_transport_time = self.transport_time_matrix[from_location, machine_id]
        
        # Apply realistic time variation to transport time
        transport_time = _add_realistic_time_variation(nominal_transport_time)
        
        debug_print(f"Transport calculation for job {job_id}: from location {from_location} to machine {machine_id}, nominal: {nominal_transport_time}, actual: {transport_time}")
        
        # CRITICAL FIX: Handle zero transport time case
        if nominal_transport_time <= 0.1:  # Small threshold to handle floating point precision
            # No transport needed, start processing immediately
            debug_print(f"Transport time negligible ({nominal_transport_time}) for job {job_id}, starting processing immediately")
            self._start_processing_without_transport(job_id)
            return
        
        # Update states - start transport
        job_state['assigned_transbot'] = transbot_id
        job_state['transport_start_time'] = self.current_time
        job_state['transport_finish_time'] = self.current_time + transport_time
        job_state['status'] = 'transporting'  # Job is being transported
        
        debug_print(f"Assigned job {job_id} to transbot {transbot_id}, status: {job_state['status']}, transport_time: {transport_time}")
        
        # Update operation timing with transport information
        current_op = job_state['current_op']
        if 'operation_timing' in job_state and current_op in job_state['operation_timing']:
            op_timing = job_state['operation_timing'][current_op]
            op_timing['transbot_id'] = transbot_id
            op_timing['transport_start_time'] = self.current_time
            op_timing['transport_finish_time'] = self.current_time + transport_time
        
        transbot_state['status'] = 'busy'
        transbot_state['current_job'] = job_id
        transbot_state['finish_time'] = self.current_time + transport_time
        transbot_state['total_transport_time'] += transport_time
        
        self.transbot_job_assignments[transbot_id] = job_id

    def _start_processing_without_transport(self, job_id: int):
        """Start processing a job that doesn't need transport (already at target machine)."""
        job_state = self.job_states[job_id]
        machine_id = job_state['assigned_machine']
        
        if machine_id is None or machine_id not in self.machine_states:
            print(f"ERROR: Job {job_id} assigned to invalid machine {machine_id}")
            return
        
        machine_state = self.machine_states[machine_id]
        
        # Check if machine is available
        if machine_state['status'] != 'idle':
            print(f"ERROR: Machine {machine_id} is not idle for job {job_id}")
            return
        
        # Get processing time for this machine
        processing_time = job_state.get('processing_time', 0)
        
        # Update job state - now processing
        job_state['status'] = 'processing'
        job_state['start_time'] = self.current_time
        job_state['finish_time'] = self.current_time + processing_time
        
        debug_print(f"Started processing job {job_id} on machine {machine_id} without transport, processing_time: {processing_time}")
        
        # Update operation timing with actual start and finish times
        current_op = job_state['current_op']
        if 'operation_timing' in job_state and current_op in job_state['operation_timing']:
            op_timing = job_state['operation_timing'][current_op]
            op_timing['processing_start_time'] = self.current_time
            op_timing['processing_finish_time'] = self.current_time + processing_time
            # Set transport timing to same as processing start (no transport time)
            op_timing['transport_start_time'] = self.current_time
            op_timing['transport_finish_time'] = self.current_time
            # Mark as no transport needed
            op_timing['needs_transport'] = False
            
            # Calculate actual transport time (should be 0 for no transport)
            op_timing['actual_transport_time'] = 0.0
        
        # Update machine state - now busy
        machine_state['status'] = 'busy'
        machine_state['current_job'] = job_id
        machine_state['finish_time'] = self.current_time + processing_time
        machine_state['total_processing_time'] += processing_time
        
        # Update machine job assignments
        self.machine_job_assignments[machine_id] = job_id

    def _advance_to_next_event(self, step_rewards) -> bool:
        """
        Advance time to the next completion event (event-driven progression).
        
        Args:
            step_rewards: Dictionary to accumulate rewards
            
        Returns:
            bool: True if any completion occurred
        """
        # Find next completion time
        next_completion_time = float('inf')
        completing_machines = []
        completing_transbots = []
        
        # Debug: Print current state
        debug_print(f"Advancing to next event. Current time: {self.current_time}")
        busy_machines = [mid for mid, state in self.machine_states.items() if state['status'] == 'busy']
        busy_transbots = [tid for tid, state in self.transbot_states.items() if state['status'] == 'busy']
        debug_print(f"Busy machines: {busy_machines}, Busy transbots: {busy_transbots}")
        
        # Check machine completion times
        for machine_id, machine_state in self.machine_states.items():
            if machine_state['status'] == 'busy' and machine_state['finish_time'] > self.current_time:
                debug_print(f"Machine {machine_id} will complete at {machine_state['finish_time']}")
                if machine_state['finish_time'] < next_completion_time:
                    next_completion_time = machine_state['finish_time']
                    completing_machines = [machine_id]
                    completing_transbots = []
                elif machine_state['finish_time'] == next_completion_time:
                    completing_machines.append(machine_id)
        
        # Check transbot completion times
        for transbot_id, transbot_state in self.transbot_states.items():
            debug_print(f"Transbot {transbot_id}: status={transbot_state['status']}, "
                  f"finish_time={transbot_state['finish_time']}, current_time={self.current_time}")
            if transbot_state['status'] == 'busy' and transbot_state['finish_time'] > self.current_time:
                debug_print(f"Transbot {transbot_id} will complete at {transbot_state['finish_time']}")
                if transbot_state['finish_time'] < next_completion_time:
                    next_completion_time = transbot_state['finish_time']
                    completing_machines = []
                    completing_transbots = [transbot_id]
                elif transbot_state['finish_time'] == next_completion_time:
                    completing_transbots.append(transbot_id)
        
        # If no completions, add small idle penalty and return
        if next_completion_time == float('inf'):
            debug_print("No completion events found, adding idle penalty")
            debug_print(f"Current time: {self.current_time}, No busy machines or transbots")
            idle_penalty = 0.05 * self.reward_scale  # Small fixed penalty for idle time
            for agent_id in self.agents:
                step_rewards[agent_id] -= idle_penalty
            return False
        
        # Advance time to next completion
        time_advance = next_completion_time - self.current_time
        self.current_time = next_completion_time
        
        debug_print(f"Advanced time from {self.current_time - time_advance} to {self.current_time}")
        debug_print(f"Processing completions: machines={completing_machines}, transbots={completing_transbots}")
        
        # Add small time penalty (removed large cumulative penalty)
        # Note: Time penalty removed to avoid cumulative negative rewards
        # The main reward signal comes from completion rewards
        
        # Process completions
        completion_occurred = False
        for machine_id in completing_machines:
            debug_print(f"Completing job on machine {machine_id}")
            if self._complete_job_on_machine(machine_id, step_rewards):
                completion_occurred = True
        
        for transbot_id in completing_transbots:
            debug_print(f"Completing transport on transbot {transbot_id}")
            if self._complete_transport_on_transbot(transbot_id, step_rewards):
                completion_occurred = True
        
        debug_print(f"Completion occurred: {completion_occurred}")
        return completion_occurred

    def _complete_job_on_machine(self, machine_id: int, step_rewards) -> bool:
        """
        Complete job processing on a machine.
        
        Args:
            machine_id: ID of the completing machine
            step_rewards: Dictionary to accumulate rewards
            
        Returns:
            bool: True if completion occurred
        """
        machine_state = self.machine_states[machine_id]
        job_id = machine_state['current_job']
        
        if job_id is None:
            return False
            
        job_state = self.job_states[job_id]
        
        # Calculate actual duration for reward
        actual_duration = self.current_time - job_state['start_time']
        
        # Get nominal times for reward calculation
        processing_time = 0
        transport_time = 0
        for mid, pt in job_state['processing_times']:
            if mid == machine_id:
                processing_time = pt
                break
        if machine_id < len(job_state['transport_times']):
            transport_time = job_state['transport_times'][machine_id]
        
        nominal_duration = processing_time + transport_time
        max_duration = dfjspt_params.max_prcs_time + dfjspt_params.max_tspt_time
        
        # Give completion reward (improved: fixed value, scale-independent)
        completion_reward = 2.0  # Fixed reward for completing an operation
        agent_id = f"machine{machine_id}"
        step_rewards[agent_id] += completion_reward * self.reward_scale
        
        # Record in results BEFORE updating job state
        completed_op = job_state['current_op']  # The operation that just completed
        if job_id in self.local_result.jobs and completed_op in self.local_result.jobs[job_id].operations:
            op_result = self.local_result.jobs[job_id].operations[completed_op]
            op_result.assigned_machine = machine_id
            op_result.actual_start_processing_time = job_state['start_time']
            op_result.actual_finish_processing_time = self.current_time
        
        # Update operation timing with completion information
        if 'operation_timing' in job_state and completed_op in job_state['operation_timing']:
            op_timing = job_state['operation_timing'][completed_op]
            # Calculate actual processing time
            if op_timing['processing_start_time'] is not None:
                op_timing['actual_processing_time'] = self.current_time - op_timing['processing_start_time']
            # Calculate actual transport time if transport occurred
            if op_timing['transport_start_time'] is not None and op_timing['transport_finish_time'] is not None:
                op_timing['actual_transport_time'] = op_timing['transport_finish_time'] - op_timing['transport_start_time']
        
        # Update job operation progress
        job_state['current_op'] += 1
        
        debug_print(f"Job {job_id} operation {job_state['current_op']-1} completed, advancing to operation {job_state['current_op']}")
        
        # Check if job is completely finished
        if job_state['current_op'] >= job_state['total_ops']:
            job_state['status'] = 'completed'
            job_state['progress'] = 1.0
            debug_print(f"Job {job_id} completed all operations")
        else:
            # Job has more operations, check if next operation is in current window
            current_op = job_state['current_op']
            scheduling_job = None
            
            # Find the corresponding scheduling job to get operation matrix
            for sj_id, sj in enumerate(self.scheduling_instance.jobs):
                if sj.job_id == job_id:
                    scheduling_job = sj
                    break
            
            if scheduling_job and current_op < len(scheduling_job.operations_matrix):
                # Check if next operation is in current window
                local_schedule_job = self.local_schedule.jobs[job_id]
                is_next_op_in_window = False
                for op_id in local_schedule_job.operations:
                    if (local_schedule_job.operations[op_id].is_current_window and 
                        op_id == current_op):
                        is_next_op_in_window = True
                        break
                
                if is_next_op_in_window:
                    # Next operation is in current window, make job ready
                    job_state['status'] = 'ready'
                    job_state['assigned_machine'] = None
                    if hasattr(job_state, 'assigned_transbot'):
                        job_state['assigned_transbot'] = None
                    
                    debug_print(f"Job {job_id} next operation {current_op} is in current window, status set to 'ready'")
                    
                    # Update processing times for the new operation
                    job_state['processing_times'] = []
                    job_state['transport_times'] = []
                    
                    for machine_id_inner in range(self.num_machines):
                        processing_time = scheduling_job.operations_matrix[current_op, machine_id_inner]
                        job_state['processing_times'].append((machine_id_inner, processing_time))
                        
                        # Add transport time (simplified)
                        transport_time = 1.0  # Default transport time
                        job_state['transport_times'].append(transport_time)
                else:
                    # Next operation is not in current window, mark as waiting
                    job_state['status'] = 'waiting'
                    job_state['assigned_machine'] = None
                    if hasattr(job_state, 'assigned_transbot'):
                        job_state['assigned_transbot'] = None
                    
                    debug_print(f"Job {job_id} next operation {current_op} is NOT in current window, status set to 'waiting'")
        
        machine_state['status'] = 'idle'
        machine_state['current_job'] = None
        machine_state['completed_jobs'] += 1
        self.machine_job_assignments[machine_id] = None
        
        # Update operation tracking
        self.remaining_operations -= 1

        # Debug: Print completion info
        debug_print(f"Completed job {job_id} on machine {machine_id}, remaining_operations: {self.remaining_operations}")

        return True
    
    def _complete_transport_on_transbot(self, transbot_id: int, step_rewards) -> bool:
        """
        Complete job transport on a transbot and start processing on assigned machine.
        
        Args:
            transbot_id: ID of the completing transbot
            step_rewards: Dictionary to accumulate rewards
            
        Returns:
            bool: True if completion occurred
        """
        transbot_state = self.transbot_states[transbot_id]
        job_id = transbot_state['current_job']
        
        if job_id is None:
            return False
            
        job_state = self.job_states[job_id]
        machine_id = job_state['assigned_machine']
        
        # Calculate actual duration for reward
        actual_duration = self.current_time - job_state.get('transport_start_time', self.current_time)
        
        # Get nominal transport time for reward calculation
        transport_time = job_state.get('transport_times', [1.0])[0] if job_state.get('transport_times') else 1.0
        max_duration = dfjspt_params.max_tspt_time
        
        # Give completion reward (improved: fixed value, scale-independent)
        completion_reward = 1.0  # Fixed reward for completing a transport
        agent_id = f"transbot{transbot_id}"
        step_rewards[agent_id] += completion_reward * self.reward_scale
        
        # Update transbot states - transport completed
        transbot_state['status'] = 'idle'
        transbot_state['current_job'] = None
        transbot_state['completed_transports'] += 1
        self.transbot_job_assignments[transbot_id] = None
        
        # Debug: Print transport completion info
        debug_print(f"Completed transport of job {job_id} on transbot {transbot_id}")
        
        # Clear transbot assignment from job
        job_state['assigned_transbot'] = None
        
        # Start processing on the assigned machine
        if machine_id is not None and machine_id in self.machine_states:
            machine_state = self.machine_states[machine_id]
            
            # Check if machine is available
            if machine_state['status'] == 'idle':
                # Start processing immediately after transport
                processing_time = job_state.get('processing_time', 0)
                
                # Update job state - now processing
                job_state['status'] = 'processing'
                job_state['start_time'] = self.current_time
                job_state['finish_time'] = self.current_time + processing_time
                
                debug_print(f"Started processing job {job_id} on machine {machine_id}, processing_time: {processing_time}")
                
                # Update operation timing with actual start and finish times
                current_op = job_state['current_op']
                if 'operation_timing' in job_state and current_op in job_state['operation_timing']:
                    job_state['operation_timing'][current_op]['processing_start_time'] = self.current_time
                    job_state['operation_timing'][current_op]['processing_finish_time'] = self.current_time + processing_time
                
                # Update machine state - now busy
                machine_state['status'] = 'busy'
                machine_state['current_job'] = job_id
                machine_state['finish_time'] = self.current_time + processing_time
                machine_state['total_processing_time'] += processing_time
                
                # Update machine job assignments
                self.machine_job_assignments[machine_id] = job_id
            else:
                # Machine is busy, job waits (this shouldn't happen in non-embodied version)
                job_state['status'] = 'waiting'
                debug_print(f"Machine {machine_id} is busy, job {job_id} set to 'waiting'")
        
        return True

    def _get_observations(self) -> Dict[str, Any]:
        """Generate global observations for all agents."""
        observations = {}
        available_jobs = self._get_available_jobs("machine")  # Default to machine stage
        
        # Calculate global features (same for all agents)
        global_features = self._calculate_global_features()
        
        # Calculate job features (same for all agents in non-embodied version)
        job_features = self._calculate_job_features(available_jobs)
        
        for agent_id in self.agents:
            if agent_id.startswith("machine"):
                machine_id = int(agent_id.lstrip("machine"))
                
                # Create action mask (logical feasibility only)
                action_mask = self._create_machine_action_mask(machine_id, available_jobs)
                
                observations[agent_id] = {
                    "action_mask": action_mask,
                    "observation": {
                        "global_features": global_features,
                        "job_features": job_features,
                    }
                }
            
            elif agent_id.startswith("transbot"):
                transbot_id = int(agent_id.lstrip("transbot"))
                
                # Create action mask for transbot
                action_mask = self._create_transbot_action_mask(transbot_id, available_jobs)
                
                observations[agent_id] = {
                    "action_mask": action_mask,
                    "observation": {
                        "global_features": global_features,
                        "job_features": job_features,
                    }
                }
        
        return observations

    def _calculate_global_features(self) -> np.ndarray:
        """Calculate global system features."""
        features = np.zeros(self.global_features_dim, dtype=np.float32)
        
        # Time and progress features
        features[0] = self.current_time
        features[1] = self.event_step
        features[2] = self.remaining_operations
        features[3] = self.total_operations_in_window
        features[4] = self.remaining_operations / max(self.total_operations_in_window, 1)
        
        # Machine utilization features
        busy_machines = sum(1 for state in self.machine_states.values() if state['status'] == 'busy')
        features[5] = busy_machines / self.num_machines
        
        # Job status features
        ready_jobs = sum(1 for state in self.job_states.values() if state['status'] == 'ready')
        processing_jobs = sum(1 for state in self.job_states.values() if state['status'] == 'processing')
        completed_jobs = sum(1 for state in self.job_states.values() if state['status'] == 'completed')
        
        features[6] = ready_jobs
        features[7] = processing_jobs
        features[8] = completed_jobs
        
        # Time remaining feature
        features[9] = max(0, self.initial_estimated_makespan - self.current_time)
        
        return features

    def _calculate_job_features(self, available_jobs: List[int]) -> np.ndarray:
        """Calculate job features matrix."""
        job_features = np.full((self.max_jobs_in_obs, self.job_features_dim), -1, dtype=np.float32)
        
        for i, job_id in enumerate(available_jobs[:self.max_jobs_in_obs]):
            job_state = self.job_states[job_id]
            
            # Basic job info
            job_features[i, 0] = job_id
            job_features[i, 1] = job_state['current_op']
            job_features[i, 2] = job_state['progress']
            
            # Processing time features (min, max, avg)
            valid_times = [pt for mid, pt in job_state['processing_times'] if pt > 0]
            if valid_times:
                job_features[i, 3] = min(valid_times)
                job_features[i, 4] = max(valid_times)
                job_features[i, 5] = sum(valid_times) / len(valid_times)
            
            # Transport time features (min, avg)
            if job_state['transport_times']:
                job_features[i, 6] = min(job_state['transport_times'])
                job_features[i, 7] = sum(job_state['transport_times']) / len(job_state['transport_times'])
        
        return job_features

    def _create_machine_action_mask(self, machine_id: int, available_jobs: List[int]) -> np.ndarray:
        """Create action mask for a machine (logical feasibility only)."""
        action_mask = np.zeros(self.max_jobs_in_obs + 1, dtype=np.int32)
        
        # Always allow "do nothing" action
        action_mask[-1] = 1
        
        # Check if machine is idle
        if self.machine_states[machine_id]['status'] != 'idle':
            return action_mask
        
        # Check each available job for logical feasibility
        for i, job_id in enumerate(available_jobs[:self.max_jobs_in_obs]):
            if self._can_machine_process_job(machine_id, job_id):
                job_state = self.job_states[job_id]
                if job_state['status'] == 'ready' and job_state['assigned_machine'] is None:
                    action_mask[i] = 1
        
        return action_mask
    
    def _create_transbot_action_mask(self, transbot_id: int, available_jobs: List[int]) -> np.ndarray:
        """Create action mask for a transbot (logical feasibility only)."""
        action_mask = np.zeros(self.max_jobs_in_obs + 1, dtype=np.int32)
        
        # Always allow "do nothing" action
        action_mask[-1] = 1
        
        # Check if transbot is idle
        if self.transbot_states[transbot_id]['status'] != 'idle':
            return action_mask
        
        # Check each available job for transport feasibility
        for i, job_id in enumerate(available_jobs[:self.max_jobs_in_obs]):
            if self._can_transbot_transport_job(transbot_id, job_id):
                action_mask[i] = 1
        
        return action_mask
    
    def _can_transbot_transport_job(self, transbot_id: int, job_id: int) -> bool:
        """Check if transbot can transport job (logical feasibility only)."""
        job_state = self.job_states[job_id]
        
        # Job must be assigned to a machine and ready for transport
        # and not already assigned to another transbot
        basic_feasibility = (job_state['assigned_machine'] is not None and 
                           job_state['status'] == 'assigned' and
                           (not hasattr(job_state, 'assigned_transbot') or job_state.get('assigned_transbot') is None))
        
        # Additionally check if transport is actually needed
        needs_transport = True
        current_op = job_state['current_op']
        if ('operation_timing' in job_state and 
            current_op in job_state['operation_timing']):
            needs_transport = job_state['operation_timing'][current_op].get('needs_transport', True)
        
        result = basic_feasibility and needs_transport
        
        debug_print(f"_can_transbot_transport_job({transbot_id}, {job_id}) = {result}")
        debug_print(f"  - assigned_machine: {job_state['assigned_machine']}")
        debug_print(f"  - status: {job_state['status']}")
        debug_print(f"  - assigned_transbot: {job_state.get('assigned_transbot')}")
        debug_print(f"  - needs_transport: {needs_transport}")
        
        return result

    def _get_dummy_observation(self) -> Dict[str, Any]:
        """Get dummy observation for terminated episodes."""
        return {
            "action_mask": np.zeros(self.max_jobs_in_obs + 1, dtype=np.int32),
            "observation": {
                "global_features": np.zeros(self.global_features_dim, dtype=np.float32),
                "job_features": np.full((self.max_jobs_in_obs, self.job_features_dim), -1, dtype=np.float32),
            }
        }

    def _validate_scheduling_constraints(self) -> bool:
        """Validate that all scheduling constraints are properly enforced."""
        constraints_violated = []
        
        # 1. Precedence constraints: operations must be processed in order
        for job_id, job_state in self.job_states.items():
            if job_state['status'] in ['processing', 'completed']:
                # Check that current_op is not ahead of what should be processed
                if job_state['current_op'] > job_state['total_ops']:
                    constraints_violated.append(f"Job {job_id}: current_op {job_state['current_op']} > total_ops {job_state['total_ops']}")
        
        # 2. Resource conflict constraints: each machine can only handle one job at a time
        for machine_id, machine_state in self.machine_states.items():
            if machine_state['status'] == 'busy':
                job_id = machine_state['current_job']
                if job_id is not None:
                    job_state = self.job_states.get(job_id)
                    if job_state is None or job_state['assigned_machine'] != machine_id:
                        constraints_violated.append(f"Machine {machine_id}: inconsistent job assignment")
        
        # 3. Resource conflict constraints: each transbot can only handle one job at a time
        for transbot_id, transbot_state in self.transbot_states.items():
            if transbot_state['status'] == 'busy':
                job_id = transbot_state['current_job']
                if job_id is not None:
                    job_state = self.job_states.get(job_id)
                    if job_state is None or job_state.get('assigned_transbot') != transbot_id:
                        constraints_violated.append(f"Transbot {transbot_id}: inconsistent job assignment")
        
        # 4. Timing constraints: transport must happen before processing
        for job_id, job_state in self.job_states.items():
            if job_state['status'] == 'processing':
                # If job is processing, it should have been transported first
                if (hasattr(job_state, 'assigned_transbot') and 
                    job_state.get('assigned_transbot') is not None):
                    constraints_violated.append(f"Job {job_id}: processing while still assigned to transbot")
        
        # 5. Feasibility constraints: jobs can only be processed on compatible machines
        for job_id, job_state in self.job_states.items():
            if job_state['assigned_machine'] is not None:
                machine_id = job_state['assigned_machine']
                if not self._can_machine_process_job(machine_id, job_id):
                    constraints_violated.append(f"Job {job_id}: assigned to incompatible machine {machine_id}")
        
        if constraints_violated:
            print("WARNING: Scheduling constraints violated:")
            for violation in constraints_violated:
                print(f"  - {violation}")
            return False
        
        return True

    def _check_terminated(self) -> bool:
        """Check if episode should terminate (all jobs completed)."""
        return self.remaining_operations <= 0

    def _check_truncated(self) -> bool:
        """Check if episode should be truncated (time/step limits)."""
        # Check step limit (always available)
        steps_exceeded = self.event_step >= self.max_event_steps
        
        # Check time limit (only if local_result and time_upper_bound are available)
        time_exceeded = False
        if (self.local_result is not None and 
            hasattr(self, 'time_upper_bound') and 
            self.time_upper_bound is not None):
            time_exceeded = (self.current_time - self.local_result.time_window_start) >= self.time_upper_bound
        
        return time_exceeded or steps_exceeded

    def _get_infos(self) -> Dict[str, Any]:
        """Get info dictionary."""
        return {
            "current_time": self.current_time,
            "event_step": self.event_step,
            "remaining_operations": self.remaining_operations,
            "total_operations": self.total_operations_in_window
        }

    def _log_episode_completion(self):
        """Log episode completion information."""
        makespan = self.current_time
        delta_makespan = makespan - self.local_result.time_window_start
        
        print(f"Non-embodied episode completed in {self.event_step} decision events and {delta_makespan:.2f} time units.")
        print(f"Final makespan: {makespan:.2f}")
        
        # Always set actual_local_makespan for callback compatibility
        self.local_result.actual_local_makespan = makespan
        
        # Save results if file specified
        if self.local_result_file:
            os.makedirs(os.path.dirname(self.local_result_file), exist_ok=True)
            with open(self.local_result_file, "wb") as f:
                pickle.dump(self.local_result, f)

    def render(self, mode="human"):
        """Render the environment (simplified for non-embodied version)."""
        if mode == "human":
            print(f"Time: {self.current_time:.2f}, Event Step: {self.event_step}")
            print(f"Remaining Operations: {self.remaining_operations}/{self.total_operations_in_window}")
            
            # Show machine states
            for machine_id, state in self.machine_states.items():
                status = state['status']
                job = state.get('current_job', 'None')
                print(f"  Machine {machine_id}: {status}, Job: {job}")

    def plot_jobs_gantt(self, plot_end_time, plot_start_time=None, save_fig_dir=None):
        """
        Plot a Gantt chart with operations colored by their assigned resources (machines and transbots).
        Shows both transport and processing operations for each job, integrated in the same chart.
        
        Args:
            plot_end_time: End time for the plot
            plot_start_time: Start time for the plot (defaults to time_window_start)
            save_fig_dir (str, optional): If provided, saves the plot to this directory.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        # Generate unique colors for machines and transbots using a single colormap
        machine_ids = list(range(self.num_machines))
        transbot_ids = list(range(self.num_transbots))
        
        # Use a single colormap for all resources (machines + transbots)
        all_resource_ids = machine_ids + transbot_ids
        num_resources = len(all_resource_ids)
        
        # Use a colormap that provides distinct colors
        if num_resources <= 20:
            colormap = plt.colormaps["tab20"]
        elif num_resources <= 40:
            colormap = plt.colormaps["tab20b"]
        else:
            colormap = plt.cm.get_cmap("nipy_spectral", num_resources)
        
        # Create color maps for machines and transbots
        machine_color_map = {machine: colormap(i / num_resources) for i, machine in enumerate(machine_ids)}
        transbot_color_map = {transbot: colormap((i + len(machine_ids)) / num_resources) for i, transbot in enumerate(transbot_ids)}

        # Find time window for the Gantt chart
        min_time = self.local_result.time_window_start
        max_time = plot_end_time
        # Add small buffer to avoid matplotlib warning about identical xlims
        if max_time <= min_time:
            max_time = min_time + 1.0
        time_window = (min_time, max_time)

        # Create the Gantt chart figure and axis
        fig, ax = plt.subplots(figsize=(14, 10))

        yticks = []
        yticklabels = []

        # Plot each job with transport and processing operations integrated
        for i, job_id in enumerate(sorted(self.job_states.keys())):
            yticks.append(i)
            yticklabels.append(f"Job {job_id}")

            job_state = self.job_states[job_id]
            
            # Plot each operation for the current job
            for op_id in range(job_state['total_ops']):
                # CRITICAL FIX: Only plot operations that are actually completed
                # Check if this operation was completed AND has valid timing data
                if (op_id < job_state['current_op'] and 
                    'operation_timing' in job_state and 
                    op_id in job_state['operation_timing']):
                    # Use operation-level timing if available
                    if 'operation_timing' in job_state and op_id in job_state['operation_timing']:
                        op_timing = job_state['operation_timing'][op_id]
                        machine_id = op_timing['machine_id']
                        transbot_id = op_timing.get('transbot_id')
                        
                        # Plot transport operation (if transport occurred)
                        if (op_timing.get('needs_transport', True) and 
                            transbot_id is not None and
                            op_timing['transport_start_time'] is not None and 
                            op_timing['transport_finish_time'] is not None):
                            
                            transport_start = op_timing['transport_start_time']
                            transport_end = op_timing['transport_finish_time']
                            
                            # Determine color based on transbot
                            color = transbot_color_map.get(transbot_id, 'gray')
                            
                            # Rectangle height for transport (same as processing)
                            rect_height = 0.3
                            
                            # Plot transport rectangle (positioned at the same y-coordinate as the job)
                            ax.add_patch(
                                Rectangle(
                                    (transport_start, i - rect_height / 2),
                                    transport_end - transport_start,  # width (duration)
                                    rect_height,  # height
                                    facecolor=color,
                                    edgecolor="white",
                                    linewidth=1,
                                    alpha=0.8,
                                )
                            )
                        
                        # Plot processing operation
                        if (op_timing['processing_start_time'] is not None and 
                            op_timing['processing_finish_time'] is not None):
                            
                            processing_start = op_timing['processing_start_time']
                            processing_end = op_timing['processing_finish_time']
                            
                            # Determine color based on machine
                            color = machine_color_map.get(machine_id, 'gray')
                            
                            # Rectangle height for processing (same as transport)
                            rect_height = 0.3
                            
                            # Plot processing rectangle (positioned at the same y-coordinate as the job)
                            ax.add_patch(
                                Rectangle(
                                    (processing_start, i - rect_height / 2),
                                    processing_end - processing_start,  # width (duration)
                                    rect_height,  # height
                                    facecolor=color,
                                    edgecolor="white",
                                    linewidth=1,
                                    alpha=0.8,
                                )
                            )
                    else:
                        # Fallback: Estimate timing based on job progress
                        if ('start_time' in job_state and 'finish_time' in job_state and
                            job_state['start_time'] is not None and job_state['finish_time'] is not None):
                            # Use job-level timing and distribute operations evenly
                            job_start = job_state['start_time']
                            job_end = job_state['finish_time']
                            job_duration = job_end - job_start
                            
                            # Estimate operation timing
                            op_duration = job_duration / job_state['total_ops']
                            start_time = job_start + op_id * op_duration
                            end_time = start_time + op_duration
                            
                            # CRITICAL FIX: Use deterministic machine assignment to avoid duplicate plotting
                            # Each operation should be assigned to exactly one machine
                            machine_id = op_id % self.num_machines
                            
                            # Determine color based on machine
                            color = machine_color_map.get(machine_id, 'gray')

                            # Rectangle height
                            rect_height = 0.3

                            # Plot a rectangle for the operation (processing only in fallback)
                            ax.add_patch(
                                Rectangle(
                                    (start_time, i - rect_height / 2),
                                    end_time - start_time,  # width (duration)
                                    rect_height,  # height
                                    facecolor=color,
                                    edgecolor="white",
                                    linewidth=1,
                                    alpha=0.8,
                                )
                            )

        # Configure axes
        ax.set_xlim(time_window[0], time_window[1])
        ax.set_ylim(-0.5, len(self.job_states) - 0.5)
        ax.set_xlabel("Time")
        ax.set_ylabel("Jobs")
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_title("Non-Embodied Scheduling - Jobs Gantt Chart (Transport + Processing)")

        # Create a unified legend for all resources (machines and transbots)
        all_legend_patches = []
        all_legend_labels = []
        
        # Add machine patches and labels
        for machine, color in machine_color_map.items():
            all_legend_patches.append(Rectangle((0, 0), 1, 1, color=color, alpha=0.8))
            all_legend_labels.append(f"Machine {machine}")
        
        # Add transbot patches and labels
        for transbot, color in transbot_color_map.items():
            all_legend_patches.append(Rectangle((0, 0), 1, 1, color=color, alpha=0.8))
            all_legend_labels.append(f"Transbot {transbot}")
        
        # Create unified legend
        ax.legend(all_legend_patches, all_legend_labels, 
                  title="Resources", loc="upper right", bbox_to_anchor=(1.01, 1), 
                  borderaxespad=0., ncol=2)

        plt.tight_layout()

        # Save or display the plot
        if save_fig_dir is not None:
            plt.savefig(save_fig_dir + "_jobs_gantt.png", dpi=300, bbox_inches='tight')

        plt.show()

    def plot_machines_gantt(self, plot_end_time, plot_start_time=None, save_fig_dir=None):
        """
        Plot a Gantt chart with operations colored by jobs.
        
        Args:
            plot_end_time: End time for the plot
            plot_start_time: Start time for the plot (defaults to time_window_start)
            save_fig_dir (str, optional): If provided, saves the plot to this directory.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        # Generate unique colors for jobs using a colormap
        job_ids = list(self.job_states.keys())
        num_jobs = len(job_ids)
        colormap = plt.colormaps["tab20"] if num_jobs <= 20 else plt.cm.get_cmap("nipy_spectral", num_jobs)
        job_color_map = {job: colormap(i / num_jobs) for i, job in enumerate(sorted(job_ids))}

        # Find time window for the Gantt chart
        min_time = self.local_result.time_window_start
        max_time = plot_end_time
        # Add small buffer to avoid matplotlib warning about identical xlims
        if max_time <= min_time:
            max_time = min_time + 1.0
        time_window = (min_time, max_time)

        # Create the Gantt chart figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))

        yticks = []
        yticklabels = []

        # Plot each machine
        for i, machine_id in enumerate(range(self.num_machines)):
            yticks.append(i)
            yticklabels.append(f"Machine {machine_id}")

            # Plot each operation for the current machine
            for job_id, job_state in self.job_states.items():
                for op_id in range(job_state['total_ops']):
                    # CRITICAL FIX: Only plot operations that are actually completed
                    # Check if this operation was completed AND has valid timing data
                    if (op_id < job_state['current_op'] and 
                        'operation_timing' in job_state and 
                        op_id in job_state['operation_timing']):
                        # Use operation-level timing if available
                        if 'operation_timing' in job_state and op_id in job_state['operation_timing']:
                            op_timing = job_state['operation_timing'][op_id]
                            if op_timing['machine_id'] == machine_id:
                                # Use processing timing for the operation rectangle
                                if op_timing['processing_start_time'] is not None and op_timing['processing_finish_time'] is not None:
                                    start_time = op_timing['processing_start_time']
                                    end_time = op_timing['processing_finish_time']
                                    
                                    # Determine color based on job
                                    color = job_color_map.get(job_id, 'gray')

                                    # Rectangle height
                                    rect_height = 0.3

                                    # Plot a rectangle for the operation
                                    ax.add_patch(
                                        Rectangle(
                                            (start_time, i - rect_height / 2),
                                            end_time - start_time,  # width (duration)
                                            rect_height,  # height
                                            facecolor=color,
                                            edgecolor="white",
                                            linewidth=1,
                                        )
                                    )
                        else:
                            # Fallback: Estimate timing and machine assignment
                            if ('start_time' in job_state and 'finish_time' in job_state and
                                job_state['start_time'] is not None and job_state['finish_time'] is not None):
                                # Use job-level timing and distribute operations evenly
                                job_start = job_state['start_time']
                                job_end = job_state['finish_time']
                                job_duration = job_end - job_start
                                
                                # Estimate operation timing
                                op_duration = job_duration / job_state['total_ops']
                                start_time = job_start + op_id * op_duration
                                end_time = start_time + op_duration
                                
                                # CRITICAL FIX: Ensure each operation is assigned to exactly one machine
                                # Use a deterministic assignment to avoid duplicate plotting
                                assigned_machine = op_id % self.num_machines
                                
                                # Only plot if this operation is assigned to the current machine
                                if assigned_machine == machine_id:
                                    # Determine color based on job
                                    color = job_color_map.get(job_id, 'gray')

                                    # Rectangle height
                                    rect_height = 0.3

                                    # Plot a rectangle for the operation
                                    ax.add_patch(
                                        Rectangle(
                                            (start_time, i - rect_height / 2),
                                            end_time - start_time,  # width (duration)
                                            rect_height,  # height
                                            facecolor=color,
                                            edgecolor="white",
                                            linewidth=1,
                                        )
                                    )

        # Configure axes
        ax.set_xlim(time_window[0], time_window[1])
        ax.set_ylim(-0.5, self.num_machines - 0.5)
        ax.set_xlabel("Time")
        ax.set_ylabel("Machines")
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_title("Non-Embodied Scheduling - Machines Gantt Chart")

        # Create a legend for the jobs
        legend_patches = [Rectangle((0, 0), 1, 1, color=color) for job, color in job_color_map.items()]
        ax.legend(legend_patches, [f"Job {j}" for j in job_color_map.keys()], 
                  title="Jobs", loc="upper right", bbox_to_anchor=(1.01, 1), 
                  borderaxespad=0., ncol=3)

        plt.tight_layout()

        # Save or display the plot
        if save_fig_dir is not None:
            plt.savefig(save_fig_dir + "_machines_gantt.png", dpi=300, bbox_inches='tight')

        plt.show()

    def plot_transbots_gantt(self, plot_end_time, plot_start_time=None, save_fig_dir=None):
        """
        Plot a Gantt chart for transbots (simplified for non-embodied version).
        
        Args:
            plot_end_time: End time for the plot
            plot_start_time: Start time for the plot (defaults to time_window_start)
            save_fig_dir (str, optional): If provided, saves the plot to this directory.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        # Generate unique colors for jobs using a colormap
        job_ids = list(self.job_states.keys())
        num_jobs = len(job_ids)
        colormap = plt.colormaps["tab20"] if num_jobs <= 20 else plt.cm.get_cmap("nipy_spectral", num_jobs)
        job_color_map = {job: colormap(i / num_jobs) for i, job in enumerate(sorted(job_ids))}

        # Find time window for the Gantt chart
        min_time = self.local_result.time_window_start
        max_time = plot_end_time
        # Add small buffer to avoid matplotlib warning about identical xlims
        if max_time <= min_time:
            max_time = min_time + 1.0
        time_window = (min_time, max_time)

        # Create the Gantt chart figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))

        yticks = []
        yticklabels = []

        # Plot each transbot
        for i, transbot_id in enumerate(range(self.num_transbots)):
            yticks.append(i)
            yticklabels.append(f"Transbot {transbot_id}")

            # Plot each transport operation for the current transbot
            for job_id, job_state in self.job_states.items():
                # Check if this job was transported by this transbot using operation_timing
                if 'operation_timing' in job_state:
                    for op_id, op_timing in job_state['operation_timing'].items():
                        if (op_timing.get('transbot_id') == transbot_id and
                            op_timing.get('transport_start_time') is not None and
                            op_timing.get('transport_finish_time') is not None and
                            op_timing.get('needs_transport', True)):  # Only show actual transport operations
                            
                            start_time = op_timing['transport_start_time']
                            end_time = op_timing['transport_finish_time']
                            
                            # Determine color based on job
                            color = job_color_map.get(job_id, 'gray')

                            # Rectangle height
                            rect_height = 0.3

                            # Plot a rectangle for the transport operation
                            ax.add_patch(
                                Rectangle(
                                    (start_time, i - rect_height / 2),
                                    end_time - start_time,  # width (duration)
                                    rect_height,  # height
                                    facecolor=color,
                                    edgecolor="white",
                                    linewidth=1,
                                )
                            )
                else:
                    # Fallback: For non-embodied environment, estimate transport timing
                    # since we don't have detailed transport records
                    if ('start_time' in job_state and 'finish_time' in job_state and 
                        job_state['start_time'] is not None and job_state['finish_time'] is not None):
                        # Estimate transport timing based on job timing
                        job_start = job_state['start_time']
                        job_end = job_state['finish_time']
                        job_duration = job_end - job_start
                        
                        # Assume transport happens before processing
                        transport_duration = job_duration * 0.1  # 10% of job duration
                        start_time = job_start
                        end_time = start_time + transport_duration
                        
                        # Assign to transbot based on job ID for visualization
                        if job_id % self.num_transbots == transbot_id:
                            # Determine color based on job
                            color = job_color_map.get(job_id, 'gray')

                            # Rectangle height
                            rect_height = 0.3

                            # Plot a rectangle for the transport operation
                            ax.add_patch(
                                Rectangle(
                                    (start_time, i - rect_height / 2),
                                    end_time - start_time,  # width (duration)
                                    rect_height,  # height
                                    facecolor=color,
                                    edgecolor="white",
                                    linewidth=1,
                                )
                            )

        # Configure axes
        ax.set_xlim(time_window[0], time_window[1])
        ax.set_ylim(-0.5, self.num_transbots - 0.5)
        ax.set_xlabel("Time")
        ax.set_ylabel("Transbots")
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_title("Non-Embodied Scheduling - Transbots Gantt Chart")

        # Create a legend for the jobs
        legend_patches = [Rectangle((0, 0), 1, 1, color=color) for job, color in job_color_map.items()]
        ax.legend(legend_patches, [f"Job {j}" for j in job_color_map.keys()], 
                  title="Jobs", loc="upper right", bbox_to_anchor=(1.01, 1), 
                  borderaxespad=0., ncol=3)

        plt.tight_layout()

        # Save or display the plot
        if save_fig_dir is not None:
            plt.savefig(save_fig_dir + "_transbots_gantt.png", dpi=300, bbox_inches='tight')

        plt.show()


# Demo methods for running complete episodes
def run_one_local_instance(
        local_instance_file,
        num_episodes=10,
        do_plot_gantt=True,
        detailed_log=True,
):
    """
    Run multiple episodes on a single local instance.
    
    Args:
        local_instance_file: Path to the local instance file
        num_episodes: Number of episodes to run
        do_plot_gantt: Whether to plot Gantt charts
        detailed_log: Whether to print detailed logs
    """
    print(f"Starting non-embodied instance {local_instance_file}:\n")

    config = {
        "n_machines": dfjspt_params.n_machines,
        "n_transbots": dfjspt_params.n_transbots,
        "factory_instance_seed": dfjspt_params.factory_instance_seed,
    }
    scheduling_env = NonEmbodiedSchedulingMultiAgentEnv(config)

    makespans = []

    for episode_id in range(num_episodes):
        makespans = run_one_episode(episode_id=episode_id + 1,
                                    scheduling_env=scheduling_env,
                                    makespans=makespans,
                                    do_plot_gantt=do_plot_gantt,
                                    local_instance_file=local_instance_file,
                                    detailed_log=detailed_log,
                                    )

    print(f"\nMin makespan across {num_episodes} episodes is {np.min(makespans)}.")
    print(f"Average makespan across {num_episodes} episodes is {np.average(makespans)}.")


def run_one_episode(
        episode_id,
        scheduling_env,
        makespans,
        do_plot_gantt=True,
        local_instance_file=None,
        detailed_log=True,
):
    """
    Run a single episode on the non-embodied scheduling environment.
    
    Args:
        episode_id: ID of the episode
        scheduling_env: The scheduling environment instance
        makespans: List to accumulate makespans
        do_plot_gantt: Whether to plot Gantt charts
        local_instance_file: Path to the local instance file
        detailed_log: Whether to print detailed logs
    """
    from local_realtime_scheduling.Agents.generate_training_data import generate_reset_options_for_training
    print(f"\nStarting non-embodied episode {episode_id}:")
    decision_count = 0

    env_reset_options = generate_reset_options_for_training(
        local_schedule_filename=local_instance_file,
        for_training=False,
    )

    observations, infos = scheduling_env.reset(options=env_reset_options)
    
    decision_count += 1
    done = {'__all__': False}
    truncated = {'__all__': False}
    total_rewards = {}
    for agent in scheduling_env.agents:
        total_rewards[agent] = 0.0

    while (not done.get('__all__', False)) and (not truncated.get('__all__', False)):
        actions = {}
        for agent_id, obs in observations.items():
            action_mask = obs['action_mask']
            valid_actions = [i for i, valid in enumerate(action_mask) if valid == 1]
            if valid_actions:
                if len(valid_actions) == 1:
                    actions[agent_id] = valid_actions[0]
                else:
                    actions[agent_id] = np.random.choice(valid_actions)
            else:
                raise Exception(f"No valid actions for agent {agent_id}!")

        observations, rewards, done, truncated, info = scheduling_env.step(actions)

        # print(f"Remaining operation: {scheduling_env.remaining_operations}")
        
        decision_count += 1

        for agent, reward in rewards.items():
            total_rewards[agent] += reward
    
    if do_plot_gantt:
        scheduling_env.plot_jobs_gantt(
            plot_end_time=scheduling_env.current_time,
            save_fig_dir=None,
        )
        scheduling_env.plot_machines_gantt(
            plot_end_time=scheduling_env.current_time,
            save_fig_dir=None,
        )
        scheduling_env.plot_transbots_gantt(
            plot_end_time=scheduling_env.current_time,
            save_fig_dir=None,
        )

    if detailed_log:
        print(f"Episode {episode_id} completed in {decision_count} decision events and {scheduling_env.current_time - scheduling_env.local_result.time_window_start:.2f} time units.")
        print(f"Final makespan: {scheduling_env.current_time:.2f}")
        
        if scheduling_env.local_result.actual_local_makespan is not None:
            print(f"Actual makespan = {scheduling_env.local_result.actual_local_makespan}")
            print(f"Actual delta makespan = {scheduling_env.local_result.actual_local_makespan - scheduling_env.local_result.time_window_start}")
            makespans.append(
                scheduling_env.local_result.actual_local_makespan - scheduling_env.local_result.time_window_start)
        else:
            print(f"Truncated makespan = {scheduling_env.current_time}")
            print(f"Truncated delta makespan = {scheduling_env.current_time - scheduling_env.local_result.time_window_start}")
            makespans.append(scheduling_env.current_time - scheduling_env.local_result.time_window_start)
        
        print(f"Estimated makespan = {scheduling_env.initial_estimated_makespan}")
        print(f"Estimated delta makespan = {scheduling_env.initial_estimated_makespan - scheduling_env.local_result.time_window_start}")
        
        # Print rewards for machine agents (representative of overall performance)
        machine_rewards = [total_rewards[agent] for agent in scheduling_env.agents if agent.startswith('machine')]
        if machine_rewards:
            avg_machine_reward = np.average(machine_rewards)
            print(f"Average machine reward for episode {episode_id}: {avg_machine_reward:.4f}")
        # Print rewards for transbot agents (representative of overall performance)
        transbot_rewards = [total_rewards[agent] for agent in scheduling_env.agents if agent.startswith('transbot')]
        if transbot_rewards:
            avg_transbot_reward = np.average(transbot_rewards)
            print(f"Average transbot reward for episode {episode_id}: {avg_transbot_reward:.4f}")

        print(f"Min makespan up to now is {np.min(makespans)}.")
        print(f"Average makespan up to now is {np.average(makespans)}.")
    else:
        if scheduling_env.local_result.actual_local_makespan is not None:
            makespans.append(
                scheduling_env.local_result.actual_local_makespan - scheduling_env.local_result.time_window_start)
        else:
            makespans.append(scheduling_env.current_time - scheduling_env.local_result.time_window_start)

    return makespans


# Example usage
if __name__ == "__main__":
    import time
    from pathlib import Path
    
    # To enable debug output, change LOG_LEVEL to "DEBUG" at the top of this file
    # LOG_LEVEL = "DEBUG"  # Uncomment this line to see all debug messages
    
    print("Non-Embodied Scheduling Demo")
    print("="*50)

    n_machines = 36
    n_transbots = 20
    n_jobs = 60
    instance_id = 104
    window_id = 2
    n_ops = 170
    real_instance_file = Path(__file__).parent.parent.parent / "InterfaceWithGlobal" / "local_schedules" / f"M{n_machines}T{n_transbots}W300" / "testing" / f"local_schedule_J{n_jobs}I{instance_id}_{window_id}_ops{n_ops}.pkl"

    if os.path.exists(real_instance_file):
        start_time = time.time()
        run_one_local_instance(
            local_instance_file=real_instance_file.name,
            do_plot_gantt=True,
            num_episodes=1
        )
        print(f"Total running time is {time.time() - start_time:.2f} seconds")
    else:
        print(f"Real instance file not found: {real_instance_file}")
        print("Skipping real instance demo.")
    
    print("\n" + "="*50)
    print("Demo completed!")
