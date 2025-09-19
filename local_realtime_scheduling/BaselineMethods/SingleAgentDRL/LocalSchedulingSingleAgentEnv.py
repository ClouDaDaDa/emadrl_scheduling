import os
import sys
import pickle
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as pch
import matplotlib.cm as cm
from matplotlib import animation
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import logging
logging.basicConfig(level=logging.INFO)

# Get the absolute path of the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the root directory by going up two levels
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)
from System.SchedulingInstance import SchedulingInstance
from System.FactoryInstance import FactoryInstance
from configs import dfjspt_params
from local_realtime_scheduling.Environment.ExecutionResult import LocalResult, Local_Job_result, Operation_result
from local_realtime_scheduling.Environment.path_planning import a_star_search
from local_realtime_scheduling.InterfaceWithGlobal.divide_global_schedule_to_local_from_pkl import LocalSchedule, Local_Job_schedule

MAX_PRCS_TIME = dfjspt_params.max_prcs_time
MAX_TSPT_TIME = dfjspt_params.max_tspt_time


class LocalSchedulingSingleAgentEnv(gym.Env):
    """
    A Single-agent Environment for Integrated Production, Transportation and Maintenance Real-time Scheduling.
    The single agent controls all machines and transbots centrally.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config):
        """
        :param config: including:
        n_machines
        n_transbots
        factory_instance_seed
        """
        super().__init__()

        # Initialize parameters
        self.num_machines = config["n_machines"]
        self.num_transbots = config["n_transbots"]
        
        self.factory_instance = FactoryInstance(
            seed=config["factory_instance_seed"],
            n_machines=self.num_machines,
            n_transbots=self.num_transbots,
        )

        # Maintenance parameters
        self.num_maintenance_methods = 4
        self.maintenance_costs = np.array([20.0, 50.0, 80.0, 100.0], dtype=np.float32)
        self.maintenance_cost_for_machines = np.zeros(self.num_machines, dtype=np.float32)
        self.maintenance_counts_for_machines = np.zeros(self.num_machines, dtype=np.int32)
        self.storage_unit_cost = 1.0
        self.tardiness_unit_cost = 1.0
        self.total_cost = 0.0

        self.initial_estimated_makespan = None
        self.time_upper_bound = None
        self.total_n_ops_for_curr_tw = None  # total_num_operations_for_current_time_window
        self.remaining_operations = None

        self.current_time_before_step = 0.0
        self.current_time_after_step = 0.0
        self.reward_this_step = 0.0

        self.decision_stage = 0  # 0 for machines and 1 for transbots
        self.terminated = False
        self.truncated = False
        self.resetted = False

        # Local view parameters (same as multi-agent)
        self.n_jobs_handled_by_machine = 20
        self.jobs_handled_by_machines = -np.ones((self.num_machines, self.n_jobs_handled_by_machine), dtype=np.int32)
        self.n_neighbor_machines = min(5, self.num_machines - 1)
        self.neighbor_machines = -np.ones((self.num_machines, self.n_neighbor_machines), dtype=np.int32)
        
        self.num_machine_actions = self.n_jobs_handled_by_machine + 5
        self.n_job_features_for_machine = 7
        self.n_neighbor_machines_features = 6
        self.n_machine_features = 8

        self.n_jobs_handled_by_transbot = 20
        self.jobs_handled_by_transbots = -np.ones((self.num_transbots, self.n_jobs_handled_by_transbot), dtype=np.int32)
        self.n_neighbor_transbots = min(5, self.num_transbots - 1)
        self.neighbor_transbots = -np.ones((self.num_transbots, self.n_neighbor_transbots), dtype=np.int32)
        
        self.num_transbot_actions = self.n_jobs_handled_by_transbot + 2
        self.n_job_features_for_transbot = 8
        self.n_neighbor_transbots_features = 8
        self.n_transbot_features = 11

        # Define observation and action spaces
        self._define_spaces()

        # Rendering settings
        self.render_mode = config.get("render_mode", None)
        self.fig, self.ax = None, None
        if self.render_mode in ["human", "rgb_array"]:
            self._initialize_rendering()

    def _define_spaces(self):
        """Define observation and action spaces for the single agent."""
        # Observation space: concatenated observations of all machines and transbots
        # Plus global features
        
        # Global features
        global_features_dim = 2  # time progress, remaining operations
        
        # Machine observations (including neighbor machines)
        machine_obs_dim = (
            self.num_machines * (
                self.n_job_features_for_machine * self.n_jobs_handled_by_machine +
                self.n_neighbor_machines_features * self.n_neighbor_machines +
                self.n_machine_features
            )
        )
        
        # Transbot observations (including neighbor transbots)
        transbot_obs_dim = (
            self.num_transbots * (
                self.n_job_features_for_transbot * self.n_jobs_handled_by_transbot +
                self.n_neighbor_transbots_features * self.n_neighbor_transbots +
                self.n_transbot_features
            )
        )
        
        # Total observation dimension
        total_obs_dim = global_features_dim + machine_obs_dim + transbot_obs_dim
        
        # Action mask dimension: total actions for all machines and transbots
        total_action_mask_dim = (
            self.num_machines * self.num_machine_actions + 
            self.num_transbots * self.num_transbot_actions
        )
        
        self.observation_space = spaces.Dict({
            "action_mask": spaces.Box(
                0, 1, 
                shape=(total_action_mask_dim,), 
                dtype=np.int32
            ),
            "observation": spaces.Box(
                low=-float('inf'), 
                high=float('inf'),
                shape=(total_obs_dim,),
                dtype=np.float32
            ),
        })
        
        # Action space: MultiDiscrete where each dimension corresponds to an agent
        # The value represents the action that agent takes
        self.action_space = spaces.MultiDiscrete(
            [self.num_machine_actions] * self.num_machines + 
            [self.num_transbot_actions] * self.num_transbots
        )

    def _get_global_observation(self):
        """Get the global observation combining all agent observations."""
        obs_list = []
        
        # Global features
        global_features = np.array([
            self.initial_estimated_makespan - self.current_time_after_step,  # Time progress
            self.remaining_operations,  # Remaining operations
        ], dtype=np.float32)
        obs_list.append(global_features)

        # Machine observations
        if self.decision_stage == 0:
            for machine_id in range(self.num_machines):
                machine_obs = self._get_machine_obs(machine_id)
                if "observation" in machine_obs:
                    obs_dict = machine_obs["observation"]
                    # Flatten and concatenate machine features
                    job_features_flat = obs_dict["job_features"].flatten()
                    neighbor_machines_features_flat = obs_dict["neighbor_machines_features"].flatten()
                    machine_features = obs_dict["machine_features"]
                    obs_list.extend([job_features_flat, neighbor_machines_features_flat, machine_features])
                else:
                    # If no observation (filtered out), use zeros
                    dummy_job_features = np.zeros(
                        self.n_job_features_for_machine * self.n_jobs_handled_by_machine,
                        dtype=np.float32
                    )
                    dummy_neighbor_features = np.zeros(
                        self.n_neighbor_machines_features * self.n_neighbor_machines,
                        dtype=np.float32
                    )
                    dummy_machine_features = np.zeros(self.n_machine_features, dtype=np.float32)
                    obs_list.extend([dummy_job_features, dummy_neighbor_features, dummy_machine_features])
        else:
            # During transbot stage, include dummy machine observations
            for _ in range(self.num_machines):
                dummy_job_features = np.zeros(
                    self.n_job_features_for_machine * self.n_jobs_handled_by_machine,
                    dtype=np.float32
                )
                dummy_neighbor_features = np.zeros(
                    self.n_neighbor_machines_features * self.n_neighbor_machines,
                    dtype=np.float32
                )
                dummy_machine_features = np.zeros(self.n_machine_features, dtype=np.float32)
                obs_list.extend([dummy_job_features, dummy_neighbor_features, dummy_machine_features])
        
        # Transbot observations
        if self.decision_stage == 1:
            for transbot_id in range(self.num_transbots):
                transbot_obs = self._get_transbot_obs(transbot_id)
                if "observation" in transbot_obs:
                    obs_dict = transbot_obs["observation"]
                    # Flatten and concatenate transbot features
                    job_features_flat = obs_dict["job_features"].flatten()
                    neighbor_transbots_features_flat = obs_dict["neighbor_transbots_features"].flatten()
                    transbot_features = obs_dict["transbot_features"]
                    obs_list.extend([job_features_flat, neighbor_transbots_features_flat, transbot_features])
                else:
                    # If no observation (filtered out), use zeros
                    dummy_job_features = np.zeros(
                        self.n_job_features_for_transbot * self.n_jobs_handled_by_transbot,
                        dtype=np.float32
                    )
                    dummy_neighbor_features = np.zeros(
                        self.n_neighbor_transbots_features * self.n_neighbor_transbots,
                        dtype=np.float32
                    )
                    dummy_transbot_features = np.zeros(self.n_transbot_features, dtype=np.float32)
                    obs_list.extend([dummy_job_features, dummy_neighbor_features, dummy_transbot_features])
        else:
            # During machine stage, include dummy transbot observations
            for _ in range(self.num_transbots):
                dummy_job_features = np.zeros(
                    self.n_job_features_for_transbot * self.n_jobs_handled_by_transbot,
                    dtype=np.float32
                )
                dummy_neighbor_features = np.zeros(
                    self.n_neighbor_transbots_features * self.n_neighbor_transbots,
                    dtype=np.float32
                )
                dummy_transbot_features = np.zeros(self.n_transbot_features, dtype=np.float32)
                obs_list.extend([dummy_job_features, dummy_neighbor_features, dummy_transbot_features])
        
        # Concatenate all observations
        global_obs = np.concatenate(obs_list)
        return global_obs

    def _get_global_action_mask(self):
        """Get the global action mask for all agents in both decision stages."""
        # Total action mask dimension: machines + transbots
        total_mask_dim = (
            self.num_machines * self.num_machine_actions + 
            self.num_transbots * self.num_transbot_actions
        )
        global_mask = np.zeros(total_mask_dim, dtype=np.int32)
        
        # Machine masks (first part of the global mask)
        machine_mask_start = 0
        for machine_id in range(self.num_machines):
            mask_start = machine_mask_start + machine_id * self.num_machine_actions
            mask_end = mask_start + self.num_machine_actions
            
            if self.decision_stage == 0:
                # Machine decision stage - get real mask
                machine_obs = self._get_machine_obs(machine_id)
                global_mask[mask_start:mask_end] = machine_obs["action_mask"]
            else:
                # Transbot decision stage - set dummy mask (all do-nothing)
                global_mask[mask_start:mask_end] = 0
                global_mask[mask_end - 1] = 1  # Only do-nothing action is valid
        
        # Transbot masks (second part of the global mask)
        transbot_mask_start = self.num_machines * self.num_machine_actions
        for transbot_id in range(self.num_transbots):
            mask_start = transbot_mask_start + transbot_id * self.num_transbot_actions
            mask_end = mask_start + self.num_transbot_actions
            
            if self.decision_stage == 1:
                # Transbot decision stage - get real mask
                transbot_obs = self._get_transbot_obs(transbot_id)
                global_mask[mask_start:mask_end] = transbot_obs["action_mask"]
            else:
                # Machine decision stage - set dummy mask (all do-nothing)
                global_mask[mask_start:mask_end] = 0
                global_mask[mask_end - 1] = 1  # Only do-nothing action is valid
        
        return global_mask

    def _update_job_index_queue_for_machines(self, machine_index):
        """Update job index queue for a given machine based on a scoring function."""
        job_scores = {}
        jobs = self.local_schedule.jobs
        scheduling_jobs = self.scheduling_instance.jobs
        factory_graph = self.factory_instance.factory_graph
        unload_time_matrix = factory_graph.unload_transport_time_matrix
        location_map = factory_graph.location_index_map

        for job_id, local_schedule_job in jobs.items():
            this_job = scheduling_jobs[job_id]
            current_op = this_job.current_processing_operation

            # Apply conditions for job exclusion
            if (this_job.n_p_ops_for_curr_tw - current_op <= 0 or
                    this_job.assigned_machine is not None or
                    this_job.operations_matrix[current_op, machine_index] <= 0):
                job_scores[job_id] = -1e5
            else:
                job_scores[job_id] = (
                        - (this_job.operations_matrix[current_op, machine_index] + unload_time_matrix[location_map[this_job.current_location], machine_index]) / (MAX_PRCS_TIME + MAX_TSPT_TIME)
                        - this_job.job_progress_for_current_time_window
                )

        # Rank jobs based on their score, jobs with higher score are more prioritized
        job_index_by_score = sorted(job_scores.keys(), key=lambda j: job_scores[j], reverse=True)
        if self.num_jobs <= self.n_jobs_handled_by_machine:
            self.jobs_handled_by_machines[machine_index][:self.num_jobs] = job_index_by_score
        else:
            self.jobs_handled_by_machines[machine_index] = job_index_by_score[:self.n_jobs_handled_by_machine]

    def _machine_kendall_tau(self, machine_1, machine_2):
        """Calculate Kendall's Tau correlation between two machines."""
        rank_1 = self.jobs_handled_by_machines[machine_1]
        rank_2 = self.jobs_handled_by_machines[machine_2]

        concordant, discordant = 0, 0
        for i, j in combinations(range(len(rank_1)), 2):
            diff_1 = rank_1[i] - rank_1[j]
            diff_2 = rank_2[i] - rank_2[j]
            if diff_1 * diff_2 > 0:
                concordant += 1
            elif diff_1 * diff_2 < 0:
                discordant += 1

        return (concordant - discordant) / (concordant + discordant) if (concordant + discordant) > 0 else 0

    def _find_top_n_similar_machines(self, machine_1):
        """Find top N most similar machines to machine_1 using Kendall's Tau."""
        similarities = np.full(self.num_machines, -1.0)

        # Compare k1 with every other machine
        for machine_2 in range(self.num_machines):
            if machine_2 != machine_1:
                similarities[machine_2] = self._machine_kendall_tau(machine_1, machine_2)

        # Get indices of the top n most similar machines (excluding k1 itself)
        top_n_indices = np.argsort(similarities)[-self.n_neighbor_machines:][::-1]  # Sort and pick top n

        return top_n_indices

    def _get_machine_obs(self, machine_index, done=False):
        """Get observation for a specific machine."""
        machine = self.factory_instance.machines[machine_index]
        
        # dynamic action masking (DAM) logic for the machine:
        machine_action_mask = np.zeros((self.num_machine_actions,), dtype=np.int32)
        if machine.machine_status == 0:
            machine_action_mask[self.n_jobs_handled_by_machine:self.n_jobs_handled_by_machine + 3] = 1
            machine_action_mask[self.n_jobs_handled_by_machine + 4] = 1
        elif machine.machine_status == 1 or machine.machine_status == 2:
            machine_action_mask[self.n_jobs_handled_by_machine + 4] = 1
        elif machine.machine_status == 3:
            machine_action_mask[self.n_jobs_handled_by_machine + 3:] = 1
        
        if machine.reliability >= 0.8:
            machine_action_mask[self.n_jobs_handled_by_machine:self.n_jobs_handled_by_machine+4] = 0
        
        self._update_job_index_queue_for_machines(machine_index)
        
        # Job Features
        job_features = np.full(
            (self.n_jobs_handled_by_machine, self.n_job_features_for_machine),
            -1, dtype=np.float32
        )
        
        if machine.current_processing_task is None:
            jobs = self.local_schedule.jobs
            scheduling_jobs = self.scheduling_instance.jobs
            factory_graph = self.factory_instance.factory_graph
            unload_time_matrix = factory_graph.unload_transport_time_matrix
            location_map = factory_graph.location_index_map
            
            for job_action_id, job_id in enumerate(self.jobs_handled_by_machines[machine_index]):
                if job_id in jobs:
                    this_job = scheduling_jobs[job_id]
                    current_op = this_job.current_processing_operation
                    
                    if this_job.n_p_ops_for_curr_tw - current_op > 0 and this_job.assigned_machine is None:
                        if this_job.operations_matrix[current_op, machine_index] > 0:
                            if this_job.job_status == 0:
                                job_remaining_finish_time = 0
                                if len(this_job.scheduled_results) > 0:
                                    prev_finish_time = this_job.scheduled_results[-1][3]
                                else:
                                    prev_finish_time = self.local_result.time_window_start
                                job_waiting_time = self.current_time_after_step - prev_finish_time
                            elif this_job.job_status == 1:
                                machine_id = int(this_job.scheduled_results[-1][2].lstrip("machine"))
                                job_remaining_finish_time = self.factory_instance.machines[
                                    machine_id].estimated_remaining_time_to_finish
                                job_waiting_time = 0
                            else:
                                raise ValueError(f"Invalid job{this_job.job_id}'s status {this_job.job_status}!")
                            
                            job_features[job_action_id] = [
                                job_id,  # [0] job_id
                                this_job.job_status,  # [1] job's internal status
                                this_job.job_progress_for_current_time_window,  # [2] job's progress
                                job_remaining_finish_time,  # [3] job's estimated remaining time to finish the operation
                                this_job.operations_matrix[current_op, machine_index],  # [4] processing time for this machine
                                unload_time_matrix[location_map[this_job.current_location], machine_index],  # [5] distance from this machine to this job
                                job_waiting_time,  # [6] job's waiting time since previous operation finished
                            ]
                            if machine.machine_status != 3:
                                machine_action_mask[job_action_id] = 1
        
        if max(machine_action_mask[:self.num_machine_actions - 1]) <= 0 and not done:
            return {
                "action_mask": machine_action_mask,
            }
        
        # Neighbor Machines Features
        neighbor_machines_features = np.zeros(
            (self.n_neighbor_machines, self.n_neighbor_machines_features),
            dtype=np.float32
        )
        top_n_indices = self._find_top_n_similar_machines(machine_index)
        machines = self.factory_instance.machines

        for i, neighbor_idx in enumerate(top_n_indices):
            neighbor_machines_features[i] = [
                machines[neighbor_idx].machine_id,
                machines[neighbor_idx].machine_status,
                -1 if machines[neighbor_idx].current_processing_task is None else machines[
                    neighbor_idx].current_processing_task,
                machines[neighbor_idx].reliability,
                machines[neighbor_idx].estimated_remaining_time_to_finish,
                machines[neighbor_idx].cumulative_tasks,
            ]
        
        # Machine Features
        machine_features = np.array([
            machine.machine_status,
            machine.reliability,
            -1 if machine.current_processing_task is None else machine.current_processing_task,
            machine.estimated_remaining_time_to_finish,
            machine.dummy_work_time / max(machine.dummy_total_time, 1.0),
            machine.cumulative_tasks,
            self.initial_estimated_makespan - self.current_time_after_step,
            self.remaining_operations,
        ], dtype=np.float32)
        
        return {
            "action_mask": machine_action_mask,
            "observation": {
                "job_features": job_features,
                "machine_features": machine_features,
                "neighbor_machines_features": neighbor_machines_features,
            }
        }

    def _update_job_index_queue_for_transbots(self, transbot_index):
        current_transbot = self.factory_instance.agv[transbot_index]

        # Design a more reasonable score function
        job_scores = {}
        location_map = self.factory_instance.factory_graph.location_index_map
        unload_time_matrix = self.factory_instance.factory_graph.unload_transport_time_matrix
        pickup_dropoff_points = self.factory_instance.factory_graph.pickup_dropoff_points

        for job_id in self.local_schedule.jobs:
            this_job = self.scheduling_instance.jobs[job_id]

            if (this_job.n_p_ops_for_curr_tw - this_job.current_processing_operation <= 0 or
                    this_job.assigned_machine is None or
                    this_job.job_status == 2 or
                    this_job.assigned_transbot is not None):
                job_scores[job_id] = -1e5
                continue

            job_location_index = location_map[this_job.current_location]
            job_to_machine = unload_time_matrix[job_location_index, this_job.assigned_machine]

            if job_to_machine <= 0:
                job_scores[job_id] = -1e5
                continue

            job_location = pickup_dropoff_points[this_job.current_location]
            transbot_to_job = abs(current_transbot.current_location[0] - job_location[0]) + abs(
                current_transbot.current_location[1] - job_location[1])

            if this_job.job_status == 0:
                job_remaining_finish_time = 0
            elif this_job.job_status == 1:
                machine_id = int(this_job.scheduled_results[-1][2].lstrip("machine"))
                job_remaining_finish_time = self.factory_instance.machines[machine_id].estimated_remaining_time_to_finish
            else:
                raise ValueError(f"Inalid job{this_job.job_id}'s status {this_job.job_status}!")

            job_scores[job_id] = (
                        - 1.0 * max(transbot_to_job, job_remaining_finish_time) / (MAX_PRCS_TIME + MAX_TSPT_TIME)
                        - 1.0 * this_job.job_progress_for_current_time_window)

        # Rank jobs based on their score, jobs with higher score are more prioritized
        job_index_by_score = sorted(job_scores.keys(), key=lambda j: job_scores[j], reverse=True)

        if self.num_jobs <= self.n_jobs_handled_by_transbot:
            self.jobs_handled_by_transbots[transbot_index][:self.num_jobs] = job_index_by_score
        else:
            self.jobs_handled_by_transbots[transbot_index] = job_index_by_score[:self.n_jobs_handled_by_transbot]

    def _transbot_kendall_tau(self, transbot_1, transbot_2):
        """Calculate Kendall's Tau correlation between two transbots."""
        rank_1 = self.jobs_handled_by_transbots[transbot_1]
        rank_2 = self.jobs_handled_by_transbots[transbot_2]

        concordant, discordant = 0, 0
        for i, j in combinations(range(len(rank_1)), 2):
            diff_1 = rank_1[i] - rank_1[j]
            diff_2 = rank_2[i] - rank_2[j]
            if diff_1 * diff_2 > 0:
                concordant += 1
            elif diff_1 * diff_2 < 0:
                discordant += 1

        return (concordant - discordant) / (concordant + discordant) if (concordant + discordant) > 0 else 0

    def _find_top_n_similar_transbots(self, transbot_1):
        """Find top N most similar transbots to transbot_1 using Kendall's Tau."""
        similarities = np.full(self.num_transbots, -1.0)

        # Compare k1 with every other transbot
        for transbot_2 in range(self.num_transbots):
            if transbot_2 != transbot_1:
                similarities[transbot_2] = self._transbot_kendall_tau(transbot_1, transbot_2)

        # Get indices of the top n most similar machines (excluding k1 itself)
        top_n_indices = np.argsort(similarities)[-self.n_neighbor_transbots:][::-1]  # Sort and pick top n

        return top_n_indices


    def _get_transbot_obs(self, transbot_index, done=False):
        """Get observation for a specific transbot."""
        transbot = self.factory_instance.agv[transbot_index]
        
        # dynamic action masking logic for the transbot
        transbot_action_mask = np.zeros((self.num_transbot_actions,), dtype=np.int32)
        if transbot.agv_status == 0:
            transbot_action_mask[self.n_jobs_handled_by_transbot:] = 1
        elif transbot.agv_status == 1:
            transbot_action_mask[self.n_jobs_handled_by_transbot:] = 1
        elif transbot.agv_status == 2 or transbot.agv_status == 3:
            transbot_action_mask[self.n_jobs_handled_by_transbot + 1] = 1
        elif transbot.agv_status == 4:
            transbot_action_mask[self.n_jobs_handled_by_transbot] = 1
        
        if transbot.battery.soc >= 0.8:
            transbot_action_mask[self.n_jobs_handled_by_transbot] = 0
        
        self._update_job_index_queue_for_transbots(transbot_index)
        
        job_features = np.full(
            (self.n_jobs_handled_by_transbot, self.n_job_features_for_transbot),
            -1, dtype=np.float32
        )
        
        if transbot.agv_status in (0, 1):
            location_map = self.factory_instance.factory_graph.location_index_map
            unload_time_matrix = self.factory_instance.factory_graph.unload_transport_time_matrix
            pickup_dropoff_points = self.factory_instance.factory_graph.pickup_dropoff_points
            
            for job_action_id, job_id in enumerate(self.jobs_handled_by_transbots[transbot_index]):
                if job_id not in self.local_schedule.jobs:
                    continue
                
                job = self.scheduling_instance.jobs[job_id]
                if job.job_status in [2, 3] or job.assigned_machine is None:
                    continue
                
                job_location_index = location_map[job.current_location]
                job_to_machine = unload_time_matrix[job_location_index, job.assigned_machine]
                
                if job_to_machine <= 0 or job.assigned_transbot is not None:
                    continue
                
                job_location = pickup_dropoff_points[job.current_location]
                transbot_to_job = abs(transbot.current_location[0] - job_location[0]) + abs(
                    transbot.current_location[1] - job_location[1])
                
                if job.job_status == 0:
                    job_remaining_finish_time = 0
                    if len(job.scheduled_results) > 0:
                        prev_finish_time = job.scheduled_results[-1][3]
                    else:
                        prev_finish_time = self.local_result.time_window_start
                    job_waiting_time = self.current_time_after_step - prev_finish_time
                elif job.job_status == 1:
                    machine_id = int(job.scheduled_results[-1][2].lstrip("machine"))
                    job_remaining_finish_time = self.factory_instance.machines[machine_id].estimated_remaining_time_to_finish
                    job_waiting_time = 0.0
                else:
                    raise ValueError(f"Invalid job{job.job_id}'s status {job.job_status}!")
                
                job_features[job_action_id] = [
                    job_id,  # [0] job_id
                    job.job_status,  # [1] job's internal status
                    job.job_progress_for_current_time_window,  # [2] job's progress
                    job_remaining_finish_time,  # [3] job's estimated remaining time to finish the operation
                    job_location[0],  # [4] job location x
                    job_location[1],  # [5] job location y
                    transbot_to_job,  # [6] transport time for this transbot to handle this operation
                    job_waiting_time,  # [7] job's waiting time since previous operation finished
                ]
                
                if not transbot.is_for_charging and not transbot.finish_unload:
                    transbot_action_mask[job_action_id] = 1
        
        if max(transbot_action_mask[:self.num_transbot_actions - 1]) <= 0 and not done:
            return {
                "action_mask": transbot_action_mask,
            }
        
        # Neighbor Transbots Features
        neighbor_transbots_features = np.zeros(
            (self.n_neighbor_transbots, self.n_neighbor_transbots_features),
            dtype=np.float32
        )
        top_n_indices = self._find_top_n_similar_transbots(transbot_1=transbot_index)
        for i, transbot_k_index in enumerate(top_n_indices):
            transbot_k = self.factory_instance.agv[transbot_k_index]
            neighbor_transbots_features[i] = [
                transbot_k.agv_id,
                transbot_k.agv_status,
                -1 if transbot_k.current_task is None else transbot_k.current_task,
                transbot_k.current_location[0],
                transbot_k.current_location[1],
                transbot_k.battery.soc,
                transbot_k.estimated_remaining_time_to_finish,
                transbot_k.cumulative_tasks
            ]
        
        transbot_features = np.array([
            transbot.agv_status,
            -1 if transbot.current_task is None else transbot.current_task,
            transbot.current_location[0],
            transbot.current_location[1],
            transbot.battery.soc,
            transbot.estimated_remaining_time_to_finish,
            transbot.t_since_prev_r,
            transbot.dummy_work_time / max(transbot.dummy_total_time, 1.0),
            transbot.cumulative_tasks,
            self.initial_estimated_makespan - self.current_time_after_step,
            self.remaining_operations,
        ], dtype=np.float32)
        
        return {
            "action_mask": transbot_action_mask,
            "observation": {
                "job_features": job_features,
                "transbot_features": transbot_features,
                "neighbor_transbots_features": neighbor_transbots_features,
            }
        }

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)

        self.resetted = True
        self.terminated = False
        self.truncated = False

        if options:
            self.factory_instance = options["factory_instance"]
            self.scheduling_instance = options["scheduling_instance"]
            self.local_schedule = options["local_schedule"]
            self.local_schedule_name = options.get("local_schedule_name", None)
            start_time = options["start_t_for_curr_time_window"]
            self.current_time_before_step = start_time
            self.current_time_after_step = start_time
            self.local_result_file = options.get("local_result_file", None)
            self.local_result = LocalResult()
            self.local_result.time_window_start = start_time
            self.time_deviation = start_time - self.local_schedule.time_window_start
            self.current_window = options.get("current_window", None)
            self.instance_n_jobs = options.get("instance_n_jobs", None)
            self.current_instance_id = options.get("current_instance_id", None)
        else:
            raise ValueError(f"Invalid options!")

        self.has_saved_instance_snapshot = False
        self.num_jobs = len(self.local_schedule.jobs)
        self.snapshot = None
        self.initial_estimated_makespan = self.local_schedule.local_makespan + self.time_deviation
        self.time_upper_bound = dfjspt_params.episode_time_upper_bound
        self.reward_this_step = 0.0
        self.total_cost = 0.0

        self.decision_stage = 0
        self.all_agents_have_made_decisions = False

        # Initialize job tracking
        self.total_n_ops_for_curr_tw = self.local_schedule.n_ops_in_tw
        self.remaining_operations = self.total_n_ops_for_curr_tw

        for job_id, local_schedule_job in self.local_schedule.jobs.items():
            job_result = Local_Job_result(job_id=job_id)
            self.local_result.add_job_result(job_result)

            this_job = self.scheduling_instance.jobs[job_id]
            this_job.at_for_curr_tw = local_schedule_job.available_time + self.time_deviation
            this_job.eft_for_curr_tw = local_schedule_job.estimated_finish_time + self.time_deviation

            for operation_id in local_schedule_job.operations:
                if local_schedule_job.operations[operation_id].is_current_window:
                    job_result.add_operation_result(Operation_result(
                        job_id=job_id,
                        operation_id=operation_id,
                    ))
                    this_job.p_ops_for_cur_tw.append(int(operation_id))

            if len(this_job.p_ops_for_cur_tw) > 0:
                this_job.reset_job_for_current_time_window()

        # Get initial observation
        observation = {
            "observation": self._get_global_observation(),
            "action_mask": self._get_global_action_mask()
        }

        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one step in the environment."""
        self.resetted = False
        self.reward_this_step = 0.0

        # Extract individual actions from the MultiDiscrete action
        if self.decision_stage == 0:
            # Machine stage - extract machine actions
            machine_actions = action[:self.num_machines]

            # Execute forced actions for machines that don't have meaningful choices
            for machine_id in range(self.num_machines):
                machine_obs = self._get_machine_obs(machine_id)
                if "action_mask" not in machine_obs or max(machine_obs["action_mask"][:-1]) <= 0:
                    # Force do-nothing action
                    machine_actions[machine_id] = self.num_machine_actions - 1

            # Process machine actions in random order
            machine_indices = list(range(self.num_machines))
            random.shuffle(machine_indices)

            for machine_id in machine_indices:
                current_machine = self.factory_instance.machines[machine_id]
                self._handle_machine_action(machine_id, current_machine, int(machine_actions[machine_id]))
            self.all_agents_have_made_decisions = False

        else:
            # Transbot stage - extract transbot actions
            transbot_actions = action[self.num_machines:self.num_machines + self.num_transbots]

            # Environment steps forward for 1 time step
            self.current_time_before_step = self.current_time_after_step
            self.current_time_after_step += 1.0

            # Execute forced actions for transbots that don't have meaningful choices
            for transbot_id in range(self.num_transbots):
                transbot_obs = self._get_transbot_obs(transbot_id)
                if "action_mask" not in transbot_obs or max(transbot_obs["action_mask"][:-1]) <= 0:
                    # Force do-nothing action
                    transbot_actions[transbot_id] = self.num_transbot_actions - 1

            # Process transbot actions in random order
            transbot_indices = list(range(self.num_transbots))
            random.shuffle(transbot_indices)

            for transbot_id in transbot_indices:
                current_transbot = self.factory_instance.agv[transbot_id]
                self._handle_transbot_action(transbot_id, current_transbot, int(transbot_actions[transbot_id]))

            # Apply time step penalty
            self.reward_this_step -= 5.0 / (self.initial_estimated_makespan - self.local_result.time_window_start)

            # Step all machines forward
            machine_indices = list(range(self.num_machines))
            random.shuffle(machine_indices)
            for machine_id in machine_indices:
                self._step_a_machine_for_one_step(machine_index=machine_id)

            # Step all transbots forward
            transbot_indices = list(range(self.num_transbots))
            random.shuffle(transbot_indices)
            for transbot_id in transbot_indices:
                self._step_a_transbot_for_one_step(transbot_index=transbot_id)

            self.all_agents_have_made_decisions = True
            # Check termination and truncation
            self.terminated = self._check_done()
            self.truncated = self._check_truncated()

            if self.terminated:
                self.reward_this_step += 5.0
                self._log_episode_termination()

        # Update decision stage
        self.decision_stage = 1 - self.decision_stage

        # Get next observation
        observation = {
            "observation": self._get_global_observation(),
            "action_mask": self._get_global_action_mask()
        }

        reward = self.reward_this_step
        info = self._get_info()

        return observation, reward, self.terminated, self.truncated, info

    def _step_a_machine_for_one_step(self, machine_index):
        current_machine = self.factory_instance.machines[machine_index]
        # Promote different evolutions according to different internal status
        # 0 (Idling): waiting for 1 time step
        if current_machine.machine_status == 0:
            # Checks whether it can start its scheduled task
            if current_machine.current_maintenance_method is not None:
                current_machine.start_maintenance(start_time=self.current_time_before_step)
                self.maintenance_cost_for_machines[machine_index] += self.maintenance_costs[
                    current_machine.current_maintenance_method]
                self.maintenance_counts_for_machines[machine_index] += 1

                current_machine.update_maintenance_process()

                if self._check_machine_finish_task(machine_id=machine_index):
                    current_machine.finish_maintenance(finish_time=self.current_time_after_step)

                # logging.info(f"Machine {machine_index} starts maintenance ({action - self.num_jobs}) at time {self.current_time_before_step}.")

            elif current_machine.current_processing_task is not None:
                current_job = self.scheduling_instance.jobs[current_machine.current_processing_task]
                # Check whether the job is currently processable
                if current_job.job_status == 0 and current_machine.machine_status == 0 and current_job.current_location == current_machine.location:
                    # Get the processing time of the operation on machine
                    processing_duration = current_job.operations_matrix[
                        current_job.current_processing_operation, machine_index]
                    estimated_processing_duration = int(processing_duration / current_machine.reliability)
                    noise = int(random.randint(-5, 5) / current_machine.reliability)
                    actual_processing_duration = max(1, estimated_processing_duration + noise)

                    # Update internal status of the job
                    self.local_result.jobs[current_job.job_id].operations[
                        current_job.current_processing_operation
                    ].actual_start_processing_time = self.current_time_before_step
                    self.local_result.jobs[current_job.job_id].operations[
                        current_job.current_processing_operation
                    ].assigned_machine = current_machine.machine_id

                    current_job.start_processing(start_time=self.current_time_before_step,
                                                 estimated_duration=estimated_processing_duration)

                    # Update internal status of the machine
                    current_machine.start_processing(start_time=self.current_time_before_step,
                                                     estimated_processing_duration=estimated_processing_duration,
                                                     actual_processing_duration=actual_processing_duration)
                    self.remaining_operations -= 1

                    # print(f"Machine {machine_index} starts processing Job {current_job.job_id} at time {self.current_time_before_step}.")
                    # print(f"Remaining ops is {self.remaining_operations}")
                    # logging.info(
                    #     f"Machine {machine_index} starts processing Job {current_job.job_id} at time {self.current_time_before_step}.")
                    current_job.update_processing()
                    current_machine.update_degradation_process()

                    if self._check_machine_finish_task(machine_id=machine_index):
                        if len(current_machine.scheduled_results) > 1:
                            actual_processing_duration = self.current_time_after_step - \
                                                         current_machine.scheduled_results[-2][2]
                        else:
                            actual_processing_duration = self.current_time_after_step - \
                                                         current_machine.scheduled_results[-1][2]
                        # actual_processing_duration = current_machine.actual_processing_duration
                        current_machine.finish_processing(finish_time=self.current_time_after_step)
                        self.reward_this_step += 10.0 * (MAX_PRCS_TIME + MAX_TSPT_TIME - actual_processing_duration) / (self.total_n_ops_for_curr_tw * (MAX_PRCS_TIME + MAX_TSPT_TIME))
                        current_job.finish_processing(finish_time=self.current_time_after_step)

                        if current_job.job_id not in self.local_result.jobs:
                            self.local_result.add_job_result(Local_Job_result(job_id=current_job.job_id))
                        if current_job.current_processing_operation - 1 not in self.local_result.jobs[
                            current_job.job_id].operations:
                            self.local_result.jobs[current_job.job_id].add_operation_result(Operation_result(
                                job_id=current_job.job_id,
                                operation_id=current_job.current_processing_operation - 1,
                                assigned_machine=machine_index,
                                actual_start_processing_time=self.local_result.time_window_start,
                            ))
                        self.local_result.jobs[current_job.job_id].operations[
                            current_job.current_processing_operation - 1
                            ].actual_finish_processing_time = self.current_time_after_step

                else:
                    current_machine.update_waiting_process()

            else:
                current_machine.update_waiting_process()

        # 1 (Processing): degrading for 1 time step
        elif current_machine.machine_status == 1:
            current_job = self.scheduling_instance.jobs[current_machine.current_processing_task]
            current_job.update_processing()
            current_machine.update_degradation_process()

            if self._check_machine_finish_task(machine_id=machine_index):
                if len(current_machine.scheduled_results) > 1:
                    actual_processing_duration = self.current_time_after_step - current_machine.scheduled_results[-2][2]
                else:
                    actual_processing_duration = self.current_time_after_step - current_machine.scheduled_results[-1][2]
                # actual_processing_duration = current_machine.actual_processing_duration
                current_machine.finish_processing(finish_time=self.current_time_after_step)
                self.reward_this_step += 10.0 * (MAX_PRCS_TIME + MAX_TSPT_TIME - actual_processing_duration) / (self.total_n_ops_for_curr_tw * (MAX_PRCS_TIME + MAX_TSPT_TIME))
                current_job.finish_processing(finish_time=self.current_time_after_step)

                if current_job.job_id not in self.local_result.jobs:
                    self.local_result.add_job_result(Local_Job_result(job_id=current_job.job_id))
                if current_job.current_processing_operation - 1 not in self.local_result.jobs[
                    current_job.job_id].operations:
                    self.local_result.jobs[current_job.job_id].add_operation_result(Operation_result(
                        job_id=current_job.job_id,
                        operation_id=current_job.current_processing_operation - 1,
                        assigned_machine=machine_index,
                        actual_start_processing_time=self.local_result.time_window_start,
                    ))
                self.local_result.jobs[current_job.job_id].operations[
                    current_job.current_processing_operation - 1
                    ].actual_finish_processing_time = self.current_time_after_step

        # 2 (Maintenance): maintaining for 1 time step
        elif current_machine.machine_status == 2:
            current_machine.update_maintenance_process()

            if self._check_machine_finish_task(machine_id=machine_index):
                current_machine.finish_maintenance(finish_time=self.current_time_after_step)

        # 3 (Faulty): broken for 1 time step
        elif current_machine.machine_status == 3:
            if current_machine.current_maintenance_method is not None:
                current_machine.start_maintenance(start_time=self.current_time_before_step)
                self.maintenance_cost_for_machines[machine_index] += self.maintenance_costs[
                    current_machine.current_maintenance_method]
                self.maintenance_counts_for_machines[machine_index] += 1
                current_machine.update_maintenance_process()
            else:
                current_machine.update_waiting_process()

    def _step_a_transbot_for_one_step(self, transbot_index):
        current_transbot = self.factory_instance.agv[transbot_index]

        # Promote different evolutions according to different internal status
        # 0 (Idling): waiting for 1 time step
        if current_transbot.agv_status == 0:
            # Checks whether it can start its scheduled task
            if current_transbot.current_task is not None:
                # for charging
                if current_transbot.current_task == -1:
                    charging_station_location = self.factory_instance.factory_graph.pickup_dropoff_points[
                        self.factory_instance.factory_graph.nearest_charging_station(
                            current_transbot.current_location
                        )
                    ]
                    transbot_to_station = abs(current_transbot.current_location[0] - charging_station_location[0]) \
                                          + abs(current_transbot.current_location[1] - charging_station_location[1])

                    if transbot_to_station > 0:
                        unload_path = a_star_search(
                            graph=self.factory_instance.factory_graph,
                            start=current_transbot.current_location,
                            goal=charging_station_location
                        )
                        current_transbot.start_unload_transporting(
                            target_location=charging_station_location,
                            unload_path=unload_path,
                            start_time=self.current_time_before_step
                        )

                        self._handle_transbot_unload_move(current_transbot=current_transbot)
                    else:
                        current_transbot.start_charging(start_time=self.current_time_before_step)
                        current_transbot.update_charging_process()
                        if self._check_transbot_finish_charging(transbot_id=transbot_index):
                            current_transbot.finish_charging(finish_time=self.current_time_after_step)

                # for transporting
                else:
                    current_job = self.scheduling_instance.jobs[current_transbot.current_task]
                    job_location = self.factory_instance.factory_graph.pickup_dropoff_points[
                        current_job.current_location]
                    # if current_job.assigned_machine is None:
                    #     print(f"Job {current_job.job_id}'s assigned machine is None!")
                    #     print(vars(current_job))
                    #     print(current_job.scheduled_results)
                    machine_location = self.factory_instance.factory_graph.pickup_dropoff_points[
                        self.factory_instance.machines[current_job.assigned_machine].location]
                    transbot_to_job = abs(current_transbot.current_location[0] - job_location[0]) \
                                      + abs(current_transbot.current_location[1] - job_location[1])

                    # Check whether the job is currently transportable
                    if transbot_to_job == 0:  # if the transbot is at the same location with the job:
                        if current_job.job_status == 0:
                            # Loaded transport can start immediately
                            loaded_path = a_star_search(
                                graph=self.factory_instance.factory_graph,
                                start=job_location,
                                goal=machine_location
                            )
                            current_transbot.start_loaded_transporting(
                                target_location=machine_location,
                                loaded_path=loaded_path,
                                start_time=self.current_time_before_step
                            )
                            # Update internal status of the job
                            self.local_result.jobs[current_job.job_id].operations[
                                current_job.current_processing_operation
                            ].actual_start_transporting_time = self.current_time_before_step
                            self.local_result.jobs[current_job.job_id].operations[
                                current_job.current_processing_operation
                            ].assigned_transbot = current_transbot.agv_id
                            current_job.start_transporting(start_time=self.current_time_before_step,
                                                           estimated_duration=len(loaded_path))

                            self._handle_transbot_loaded_move(current_transbot=current_transbot)

                        else:
                            current_transbot.idling_process()
                    else:
                        # if current_job.estimated_remaining_time_for_current_task <= transbot_to_job:
                        # the transbot can start to go to the job
                        unload_path = a_star_search(
                            graph=self.factory_instance.factory_graph,
                            start=current_transbot.current_location,
                            goal=job_location
                        )
                        current_transbot.start_unload_transporting(
                            target_location=job_location,
                            unload_path=unload_path,
                            start_time=self.current_time_before_step
                        )
                        self._handle_transbot_unload_move(current_transbot=current_transbot)
            else:
                current_transbot.idling_process()

        # 1 (Unload Transporting): moving for 1 time step
        elif current_transbot.agv_status == 1:
            # If the congestion exceeds 10.0, an error message will be displayed
            if current_transbot.congestion_time >= 4.0:
                # print(f"Transbot {current_transbot.agv_id} waiting for too long in {current_transbot.current_location} during unload transporting!")
                # raise Exception(f"Transbot {transbot_index} has been congested for 10 time steps!")
                if len(current_transbot.current_path) > 0:
                    print(
                        f"Transbot {current_transbot.agv_id} waiting for too long in {current_transbot.current_location} during unload transporting!")
                    if current_transbot.is_for_charging:  # go to a charging station
                        goal = self.factory_instance.factory_graph.pickup_dropoff_points[
                            self.factory_instance.factory_graph.nearest_charging_station(
                                current_transbot.current_location
                            )
                        ]
                    else:  # go to a job
                        goal = self.factory_instance.factory_graph.pickup_dropoff_points[
                            self.scheduling_instance.jobs[
                                current_transbot.current_task
                            ].current_location
                        ]
                    replan_path = a_star_search(
                        graph=self.factory_instance.factory_graph,
                        start=current_transbot.current_location,
                        goal=goal,
                    )
                    current_transbot.current_path = replan_path
                else:
                    pass

            self._handle_transbot_unload_move(current_transbot=current_transbot)

        # 2 (Loaded Transporting): moving for 1 time step
        elif current_transbot.agv_status == 2:
            # If the congestion exceeds 10.0, an error message will be displayed
            if current_transbot.congestion_time >= 4.0:
                # print(f"Transbot {current_transbot.agv_id} waiting for too long in {current_transbot.current_location} during loaded transporting!")
                # raise Exception(f"Transbot {transbot_index} has been congested for 10 time steps!")
                if len(current_transbot.current_path) > 0:
                    goal = self.factory_instance.factory_graph.pickup_dropoff_points[
                        self.factory_instance.machines[
                            self.scheduling_instance.jobs[
                                current_transbot.current_task
                            ].assigned_machine
                        ].location
                    ]
                    replan_path = a_star_search(
                        graph=self.factory_instance.factory_graph,
                        start=current_transbot.current_location,
                        goal=goal,
                    )
                    current_transbot.current_path = replan_path
                else:
                    pass

            self._handle_transbot_loaded_move(current_transbot=current_transbot)

        # 3 (Charging): charging for 1 time step
        elif current_transbot.agv_status == 3:
            current_transbot.update_charging_process()
            if self._check_transbot_finish_charging(transbot_id=transbot_index):
                current_transbot.finish_charging(finish_time=self.current_time_after_step)

        # 4 (Low battery):
        elif current_transbot.agv_status == 4:
            if current_transbot.current_task != -1:
                # raise ValueError(f"The transbot {current_transbot.agv_id} should go to charge!")
                print(f"The transbot {current_transbot.agv_id} should go to charge!")

                # If transbot changes its decision, release the binding relationship with the previous job
                if current_transbot.current_task is not None:
                    if current_transbot.current_task >= 0:
                        this_job = self.scheduling_instance.jobs[current_transbot.current_task]
                        this_job.assigned_transbot = None
                current_transbot.current_task = -1
                charging_station_location = self.factory_instance.factory_graph.pickup_dropoff_points[
                    self.factory_instance.factory_graph.nearest_charging_station(
                        current_transbot.current_location
                    )
                ]
                unload_path = a_star_search(
                    graph=self.factory_instance.factory_graph,
                    start=current_transbot.current_location,
                    goal=charging_station_location
                )
                # print(
                #     f"from {current_transbot.current_location} to {charging_station_location}, unload_path = {unload_path}")
                current_transbot.start_unload_transporting(
                    target_location=charging_station_location,
                    unload_path=unload_path,
                    start_time=self.current_time_before_step
                )

                self._handle_transbot_unload_move(current_transbot=current_transbot)

            else:
                charging_station_location = self.factory_instance.factory_graph.pickup_dropoff_points[
                    self.factory_instance.factory_graph.nearest_charging_station(
                        current_transbot.current_location
                    )
                ]
                unload_path = a_star_search(
                    graph=self.factory_instance.factory_graph,
                    start=current_transbot.current_location,
                    goal=charging_station_location
                )
                # print(
                #     f"from {current_transbot.current_location} to {charging_station_location}, unload_path = {unload_path}")
                current_transbot.start_unload_transporting(
                    target_location=charging_station_location,
                    unload_path=unload_path,
                    start_time=self.current_time_before_step
                )

                self._handle_transbot_unload_move(current_transbot=current_transbot)

    def _handle_machine_action(self, machine_index, current_machine, action):
        # perform a processing task
        if 0 <= action < self.n_jobs_handled_by_machine:
            job_id = self.jobs_handled_by_machines[machine_index][action]
            # Check the validity of the processing action
            if self._check_machine_processing_action(machine_index, job_id):
                current_machine.current_processing_task = job_id
                self.scheduling_instance.jobs[job_id].assigned_to_machine(machine_index)

        # perform maintenance
        elif self.n_jobs_handled_by_machine <= action < self.n_jobs_handled_by_machine + 4:
            maintenance_method = action - self.n_jobs_handled_by_machine
            # Check the validity of the maintenance action
            self._check_machine_maintenance_action(machine_index, maintenance_method)
            current_machine.current_maintenance_method = maintenance_method

        # do-nothing
        elif action == self.n_jobs_handled_by_machine + 4:
            pass
        else:
            raise Exception(f"Invalid action ({action}) for machine {machine_index}!")

    def _check_machine_processing_action(self, machine_index, processing_action):
        # Check what status is the machine currently in, must be 0 (idling) or 1 or 2 to continue
        if self.factory_instance.machines[machine_index].machine_status not in (0, 1, 2):
            raise Exception(f"Only idling machine can choose a job!")

        # Check whether the job is in the current problem
        if processing_action not in self.local_schedule.jobs:  # the job is not in this problem
            raise Exception(f"Job {processing_action} is not in the current problem!")

        current_job = self.scheduling_instance.jobs[processing_action]
        current_operation = current_job.current_processing_operation

        # Check whether the job has been finished
        if current_job.job_status == 3:
            raise Exception(f"Job {processing_action} has been finished!")

        # Check whether the job and operation can be processed by the machine
        if current_job.operations_matrix[current_operation, machine_index] <= 0:
            raise Exception(
                f"Machine {machine_index} cannot process job {processing_action}'s operation {current_operation}!")

        # Check whether the job has already assigned to another machine
        if current_job.assigned_machine is not None:
            self.reward_this_step -= 1.0 / self.total_n_ops_for_curr_tw
            return False
        else:
            return True

    def _check_machine_maintenance_action(self, machine_index, maintenance_method):
        current_machine = self.factory_instance.machines[machine_index]
        # Check what status is the machine currently in, must be 0 (idling) or 3 (failed) to continue
        if current_machine.machine_status == 1 or current_machine.machine_status == 2:
            raise Exception(f"...!")
        if current_machine.machine_status == 0 and maintenance_method == 3:
            raise Exception(f"machine {machine_index} is not failed, so cannot choose CM ({maintenance_method})!")
        if current_machine.machine_status == 4 and maintenance_method != 3:
            raise Exception(f"machine {machine_index} is failed, so can only choose CM (not {maintenance_method})!")

    def _handle_transbot_action(self, transbot_index, current_transbot, action):
        # perform the transporting task
        if 0 <= action < self.n_jobs_handled_by_transbot:
            job_id = self.jobs_handled_by_transbots[transbot_index][action]

            # Check the validity of the transporting action
            if self._check_transbot_transporting_action(transbot_index=transbot_index,
                                                        transporting_action=job_id):
                # If transbot changes its decision, release the binding relationship with the previous job
                if current_transbot.current_task is not None:
                    if current_transbot.current_task != job_id:
                        old_job = self.scheduling_instance.jobs[current_transbot.current_task]
                        old_job.assigned_transbot = None

                        if current_transbot.agv_status == 1:
                            current_transbot.finish_unload_transporting(finish_time=self.current_time_before_step)

                        current_transbot.current_task = job_id
                        current_transbot.t_since_choose_job = self.current_time_before_step
                        new_job = self.scheduling_instance.jobs[job_id]
                        new_job.assigned_to_transbot(transbot_index)
                else:
                    current_transbot.current_task = job_id
                    current_transbot.t_since_choose_job = self.current_time_before_step
                    new_job = self.scheduling_instance.jobs[job_id]
                    new_job.assigned_to_transbot(transbot_index)

            else:
                pass

        # (move for) charging
        elif action == self.n_jobs_handled_by_transbot:
            # Check the validity of the charging action
            self._check_transbot_charging_action(transbot_index=transbot_index,
                                                 charging_action=action)

            # If transbot changes its decision, release the binding relationship with the previous job
            if current_transbot.current_task is not None:
                if current_transbot.current_task >= 0:
                    this_job = self.scheduling_instance.jobs[current_transbot.current_task]
                    this_job.assigned_transbot = None
                    if current_transbot.agv_status == 1:
                        current_transbot.finish_unload_transporting(finish_time=self.current_time_before_step)

            current_transbot.current_task = -1

        # do-nothing
        elif action == self.n_jobs_handled_by_transbot + 1:
            if current_transbot.agv_status == 4:
                raise Exception(f"...!")

        else:
            raise Exception(f"Invalid action ({action}) for transbot {transbot_index}!")

    def _check_transbot_transporting_action(self, transbot_index, transporting_action):
        current_transbot = self.factory_instance.agv[transbot_index]
        # Check what status is the transbot currently in, must be 0 (idling) or 1 (unload trans) to continue
        if current_transbot.agv_status not in  (0, 1):
            raise Exception(f"current_transbot is not idling nor unload transporting, thus cannot make other decisions!")

        if current_transbot.is_for_charging or current_transbot.current_task == -1:
            raise ValueError(f"Transbot {transbot_index} is for charging, cannot change decision!")

        # Check whether the job is in the current problem
        if transporting_action not in self.local_schedule.jobs:  # the job is in this problem
            raise Exception(f"Job {transporting_action} is not in the current problem!")

        current_job = self.scheduling_instance.jobs[transporting_action]

        # Check whether the job has been finished
        if current_job.job_status == 3:
            raise Exception(f"Job {transporting_action} has been finished!")

        # Check whether the job has assigned to a machine
        if current_job.assigned_machine is None:
            raise Exception(f"Job {transporting_action} hasn't assigned to a machine, so the target is None!")

        # Check whether the job needs transportation
        if current_job.job_status == 2:  # if the job is in transporting, this transbot don't consider it
            raise Exception(f"Job {transporting_action} is in transporting!")
        job_location_index = self.factory_instance.factory_graph.location_index_map[
            current_job.current_location]
        job_to_machine = self.factory_instance.factory_graph.unload_transport_time_matrix[
            job_location_index, current_job.assigned_machine  # machine_location_index == machine_index
        ]
        if job_to_machine == 0:  # the job is at its destination and doesn't need to be transported
            raise Exception(f"Job {transporting_action} doesn't need to be transported!")

        # Check whether the job has already assigned to another transbot
        if current_job.assigned_transbot is not None:
            if current_job.assigned_transbot == current_transbot.agv_id:
                raise Exception(f"Why transbot {current_transbot.agv_id} re-choose job {transporting_action}?")
            self.reward_this_step -= 1.0 / self.total_n_ops_for_curr_tw
            if current_transbot.current_task is not None:
                if current_transbot.current_task == -1:
                    raise ValueError(f"Transbot {transbot_index} is for charging, cannot change decision!")
                else:
                    if current_transbot.agv_status == 1:
                        pass
                    else:
                        pass
            return False
        else:
            return True

    def _check_transbot_charging_action(self, transbot_index, charging_action):
        current_transbot = self.factory_instance.agv[transbot_index]
        # Check what status is the transbot currently in,
        # must be 0 (idling), 1 (unload trans) or 4 (low battery) to continue
        if current_transbot.agv_status not in (0, 1, 4):
            raise Exception(f"...!")

    def _check_machine_finish_task(self, machine_id):
        current_machine = self.factory_instance.machines[machine_id]
        if self.current_time_after_step >= current_machine.start_time_of_the_task + current_machine.actual_processing_duration:
            return True
        else:
            return False

    def _check_transbot_finish_transporting(self, transbot_id):
        current_transbot = self.factory_instance.agv[transbot_id]
        if current_transbot.current_location == current_transbot.target_location:
            return True
        else:
            return False

    def _check_transbot_finish_charging(self, transbot_id):
        current_transbot = self.factory_instance.agv[transbot_id]
        if current_transbot.charging_time <= 0:
            return True
        else:
            return False

    def _handle_transbot_unload_move(self, current_transbot):
        # Check whether current_transbot has a walkable path
        if self._check_walkable_path(current_transbot=current_transbot):

            # Check if next_location is walkable
            next_location = current_transbot.current_path[0]
            if self.factory_instance.factory_graph.is_walkable(x=next_location[0], y=next_location[1]):
                del current_transbot.current_path[0]

                self._transbot_move_to_the_next_location(current_transbot=current_transbot,
                                                         next_location=next_location,
                                                         load=0)

                if self._check_transbot_finish_transporting(transbot_id=current_transbot.agv_id):
                    if current_transbot.current_task >= 0:
                        # Check whether the job is currently transportable
                        current_job = self.scheduling_instance.jobs[current_transbot.current_task]
                        if current_job.assigned_transbot is None:
                            raise ValueError(f"Job {current_job.job_id} has not transbot! Cannot start transporting.")
                        if current_job.assigned_machine is None:
                            raise ValueError(f"Job {current_job.job_id}'s assigned machine is None!")
                        job_location = self.factory_instance.factory_graph.pickup_dropoff_points[
                            current_job.current_location]
                        machine_location = self.factory_instance.factory_graph.pickup_dropoff_points[
                            self.factory_instance.machines[current_job.assigned_machine].location]
                        transbot_to_job = abs(current_transbot.current_location[0] - job_location[0]) \
                                          + abs(current_transbot.current_location[1] - job_location[1])
                        if transbot_to_job > 0:
                            raise Exception(f"Transbot {current_transbot.agv_id} hasn't get job {current_transbot.current_task}!")
                        current_transbot.finish_unload_transporting(finish_time=self.current_time_after_step)
                        if current_job.job_status == 0:
                            # Loaded transport can start immediately
                            loaded_path = a_star_search(
                                graph=self.factory_instance.factory_graph,
                                start=job_location,
                                goal=machine_location
                            )
                            current_transbot.start_loaded_transporting(
                                target_location=machine_location,
                                loaded_path=loaded_path,
                                start_time=self.current_time_after_step
                            )
                            # Update internal status of the job
                            self.local_result.jobs[current_job.job_id].operations[
                                current_job.current_processing_operation
                            ].actual_start_transporting_time = self.current_time_before_step
                            self.local_result.jobs[current_job.job_id].operations[
                                current_job.current_processing_operation
                            ].assigned_transbot = current_transbot.agv_id
                            current_job.start_transporting(start_time=self.current_time_after_step,
                                                           estimated_duration=len(loaded_path))
                    elif current_transbot.current_task == -1:
                        current_transbot.finish_unload_transporting(finish_time=self.current_time_after_step)
                        current_transbot.start_charging(start_time=self.current_time_after_step)
                else:
                    # Mark the new position of the transbot as an obstacle
                    self.factory_instance.factory_graph.set_obstacle(location=current_transbot.current_location)

            else:
                if current_transbot.congestion_time > 1:
                    # Randomly move in other passable directions
                    self._move_to_another_walkable_location(current_transbot=current_transbot,
                                                            next_location=next_location,
                                                            load=0)
                else:
                    # waiting
                    current_transbot.congestion_time += 1.0
                    current_transbot.moving_one_step(direction=(0, 0), load=0)
        else:
            # waiting
            current_transbot.congestion_time += 1.0
            current_transbot.moving_one_step(direction=(0, 0), load=0)

    def _handle_transbot_loaded_move(self, current_transbot):
        # Check whether current_transbot has a walkable path
        if self._check_walkable_path(current_transbot=current_transbot):
            # Check if next_location is walkable
            next_location = current_transbot.current_path[0]
            if self.factory_instance.factory_graph.is_walkable(x=next_location[0], y=next_location[1]):
                del current_transbot.current_path[0]

                self._transbot_move_to_the_next_location(current_transbot=current_transbot,
                                                         next_location=next_location,
                                                         load=1)

                if self._check_transbot_finish_transporting(transbot_id=current_transbot.agv_id):
                    current_job = self.scheduling_instance.jobs[current_transbot.current_task]
                    if self.factory_instance.factory_graph.pickup_dropoff_points[
                        f"machine_{current_job.assigned_machine}"] != current_transbot.current_location:
                        raise ValueError(f"job's location mismatch machine's location!")
                    current_job.finish_transporting(finish_time=self.current_time_after_step,
                                                    current_location=f"machine_{current_job.assigned_machine}")
                    self.local_result.jobs[current_job.job_id].operations[
                        current_job.current_processing_operation
                    ].actual_finish_transporting_time = self.current_time_after_step

                    actual_duration = self.current_time_after_step - current_transbot.prev_loaded_finish_time
                    current_transbot.finish_loaded_transporting(finish_time=self.current_time_after_step)
                    self.reward_this_step += 10.0 * (2 * MAX_TSPT_TIME - actual_duration) / (self.total_n_ops_for_curr_tw * MAX_TSPT_TIME)
                    current_transbot.t_since_prev_r = 0.0
                else:
                    # Mark the new position of the transbot as an obstacle
                    self.factory_instance.factory_graph.set_obstacle(location=current_transbot.current_location)

            else:
                if current_transbot.congestion_time > 1:
                    # Randomly move in other passable directions
                    self._move_to_another_walkable_location(current_transbot=current_transbot,
                                                            next_location=next_location,
                                                            load=1)
                else:
                    # waiting
                    current_transbot.congestion_time += 1.0
                    current_transbot.moving_one_step(direction=(0, 0), load=1)
                    current_job = self.scheduling_instance.jobs[current_transbot.current_task]
                    current_job.update_transporting(current_location=current_transbot.current_location)
        else:
            # waiting
            current_transbot.congestion_time += 1.0
            current_transbot.moving_one_step(direction=(0, 0), load=1)
            current_job = self.scheduling_instance.jobs[current_transbot.current_task]
            current_job.update_transporting(current_location=current_transbot.current_location)

    def _transbot_move_to_the_next_location(self, current_transbot, next_location, load: int):
        current_transbot.congestion_time = 0.0
        direction = (next_location[0] - current_transbot.current_location[0],
                     next_location[1] - current_transbot.current_location[1])
        # Mark the old location of the transbot as walkable
        self.factory_instance.factory_graph.set_walkable(location=current_transbot.current_location)
        # transbot move for one step
        current_transbot.moving_one_step(direction=direction, load=load)
        if load > 0:
            current_job = self.scheduling_instance.jobs[current_transbot.current_task]
            current_job.update_transporting(current_location=current_transbot.current_location)

    def _move_to_another_walkable_location(self, current_transbot, next_location, load: int):
        walkable_next_direction = self.factory_instance.factory_graph.check_adjacent_positions_walkable(
            current_location=current_transbot.current_location,
            occupied_location=next_location
        )
        if walkable_next_direction is None:
            # waiting
            current_transbot.congestion_time += 1.0
            current_transbot.moving_one_step(direction=(0, 0), load=load)
            if load > 0:
                current_job = self.scheduling_instance.jobs[current_transbot.current_task]
                current_job.update_transporting(current_location=current_transbot.current_location)
        else:
            current_transbot.congestion_time = 0.0
            if load > 0:
                current_transbot.current_path.insert(0, current_transbot.current_location)
            else:
                current_transbot.current_path.insert(0, current_transbot.current_location)
            # Mark the old location of the transbot as walkable
            self.factory_instance.factory_graph.set_walkable(location=current_transbot.current_location)
            # transbot move for one step
            current_transbot.moving_one_step(direction=walkable_next_direction, load=load)
            if load > 0:
                current_job = self.scheduling_instance.jobs[current_transbot.current_task]
                current_job.update_transporting(current_location=current_transbot.current_location)
            # Mark the new position of the transbot as an obstacle
            self.factory_instance.factory_graph.set_obstacle(location=current_transbot.current_location)

    def _check_walkable_path(self, current_transbot):
        if len(current_transbot.current_path) == 0:
            # Try to re-plan a walkable path
            start = current_transbot.current_location
            if current_transbot.agv_status == 1:  # Unload transporting
                if current_transbot.is_for_charging:  # go to a charging station
                    goal = self.factory_instance.factory_graph.pickup_dropoff_points[
                        self.factory_instance.factory_graph.nearest_charging_station(
                            current_transbot.current_location
                        )
                    ]
                else:  # go to a job
                    goal = self.factory_instance.factory_graph.pickup_dropoff_points[
                        self.scheduling_instance.jobs[
                            current_transbot.current_task
                        ].current_location
                    ]
            elif current_transbot.agv_status == 2:  # Loaded transporting
                goal = self.factory_instance.factory_graph.pickup_dropoff_points[
                            self.factory_instance.machines[
                                self.scheduling_instance.jobs[
                                    current_transbot.current_task
                                ].assigned_machine
                            ].location
                        ]
            else:
                raise ValueError(f"Incorrect transbot status {current_transbot.agv_status}!")

            replan_path = a_star_search(
                graph=self.factory_instance.factory_graph,
                start=start,
                goal=goal,
            )
            if len(replan_path) > 0:
                current_transbot.current_path = replan_path
                return True
            else:
                return False
        else:
            return True

    def _check_done(self):
        """Check if all jobs are completed."""
        if all(job.is_completed_for_current_time_window for job in self.scheduling_instance.jobs):
            return True
        else:
            return False

    def _check_truncated(self):
        """Check if time limit is exceeded."""
        if (self.current_time_after_step - self.local_result.time_window_start) >= self.time_upper_bound:
            return True
        else:
            return False

    def _get_info(self):
        """Return current decision stage info."""
        return {"current_decision_stage": self.decision_stage}

    def _log_episode_termination(self):
        """Log episode termination information."""
        print(
            f"This episode is successfully terminated in {self.current_time_after_step - self.local_result.time_window_start} time steps.")
        makespan = self.current_time_after_step
        self.local_result.actual_local_makespan = makespan
        print(f"Actual Makespan is {makespan}.")
        print(f"Actual delta Makespan is {makespan - self.local_result.time_window_start}.")

        # self._save_instance_snapshot(final=True)

        if self.local_result_file:
            os.makedirs(os.path.dirname(self.local_result_file), exist_ok=True)
            with open(self.local_result_file, "wb") as local_result_file:
                pickle.dump(self.local_result, local_result_file)

    def _initialize_rendering(self):
        """Initialize rendering settings (same as multi-agent)."""
        factory_graph = self.factory_instance.factory_graph
        self.figsize = (12, 6)
        
        # Machine color mapping
        machine_ids = [f"Machine {m.machine_id}" for m in self.factory_instance.machines]
        colormap = plt.colormaps["tab20"] if self.num_machines <= 20 else cm.get_cmap("hsv", self.num_machines)
        self.machine_color_map = {m_id: colormap(i / self.num_machines) for i, m_id in enumerate(machine_ids)}
        
        # Transbot color mapping
        transbot_ids = [f"Transbot {t.agv_id}" for t in self.factory_instance.agv]
        colormap = plt.colormaps["Set3"] if self.num_transbots <= 12 else cm.get_cmap("Purples", self.num_transbots)
        self.transbot_color_map = {t_id: colormap(i / self.num_transbots) for i, t_id in enumerate(transbot_ids)}

    # Include all visualization methods (render, close, plot_gantt methods, etc.)
    # These would be identical to the multi-agent version

    def render(self):
        """Render the environment (same as multi-agent)."""
        if self.all_agents_have_made_decisions or self.resetted:
            if self.render_mode is None:
                logging.warning("You are calling render method without specifying any render mode.")
                return None
            if self.render_mode not in ["human", "rgb_array"]:
                raise NotImplementedError(f"Render mode '{self.render_mode}' is not supported.")
            if self.resetted:
                plt.ion()  # Enable interactive mode
                self.fig, self.ax = plt.subplots(figsize=self.figsize)
                self.ax.set_xlim(-1, self.factory_instance.factory_graph.width + 1)
                self.ax.set_ylim(-1, self.factory_instance.factory_graph.height + 1)
                self.ax.set_xlabel("X Position")
                self.ax.set_ylabel("Y Position")
                x_ticks = range(-1, self.factory_instance.factory_graph.width + 1, 1)
                y_ticks = range(-1, self.factory_instance.factory_graph.height + 1, 1)
                self.ax.set_xticks(x_ticks)
                self.ax.set_yticks(y_ticks)
                self.ax.set_aspect('equal', adjustable='box')
                self.fig.subplots_adjust(right=0.6)
            self._update_dynamic_elements()
            if self.render_mode == "human":
                plt.show()
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                plt.pause(1 / self.metadata["render_fps"])
            elif self.render_mode == "rgb_array":
                raise NotImplementedError(f"Render mode '{self.render_mode}' is not supported.")

    def close(self):
        """Close the environment (same as multi-agent)."""
        if self.render_mode is not None:
            if self.fig is not None:
                plt.close(self.fig)
                self.fig = None
                plt.ioff()

    def _plot_static_elements(self):
        """
        Plot static elements: obstacles, pickup/dropoff points (machines, charging stations, warehouse).
        """
        # Plot obstacles
        for x, y in self.factory_instance.factory_graph.obstacles:
            self.ax.add_patch(pch.Rectangle((x - 0.5, y - 0.5), 1, 1, color='dimgray'))
        # for x in range(self.factory_instance.factory_graph.width):
        #     for y in range(self.factory_instance.factory_graph.height):
        #         if not self.factory_instance.factory_graph.is_walkable(x, y):
        #             self.ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color='dimgray'))

        # Plot pickup/dropoff points (machines, charging stations, warehouse)
        for point_name, location in self.factory_instance.factory_graph.pickup_dropoff_points.items():
            x, y = location
            if point_name == "warehouse" or "charging" in point_name:
                self.ax.add_patch(pch.Circle((x, y), radius=0.45, alpha=0.6, color='lightgray', label=f'{point_name}'))
                # self.ax.scatter(x, y, color='lightgray', alpha=0.6, s=self.pickup_dropoff_size, label=f'{point_name}')
            elif "machine" in point_name:
                machine_id = int(point_name.split('_')[-1])
                machine_color = self.machine_color_map[f"Machine {machine_id}"]
                machine_handle = self.ax.add_patch(pch.Circle((x, y), radius=0.45, alpha=0.6, color=machine_color,
                                                              label=f'Machine {machine_id} ({x}, {y})'))
                # self.machine_handles.append(machine_handle)
                # self.machine_labels.append(f'Machine {machine_id} ({x}, {y})')
            # if "machine" in point_name or "charging" in point_name or point_name == "warehouse":
            #     self.ax.scatter(x, y, color='lightgray', alpha=0.6, s=600, label=f'{point_name}')

    def _update_dynamic_elements(self):
        """
        Update and plot dynamic elements: transbots and jobs.
        This method clears the dynamic elements and redraws them for the current time step.
        """
        # Clear previous dynamic elements
        self.ax.cla()
        # self.ax.grid(True)
        x_ticks = range(-1, self.factory_instance.factory_graph.width + 1, 1)
        y_ticks = range(-1, self.factory_instance.factory_graph.height + 1, 1)
        self.ax.set_xticks(x_ticks)
        self.ax.set_yticks(y_ticks)
        self.ax.grid(True, linestyle='--', linewidth=0.5)

        # Replot the static elements (obstacles, pickup/dropoff points)
        self._plot_static_elements()

        # Display current time on the canvas
        current_time_text = f"Time: {self.current_time_after_step:.1f}"
        self.ax.text(0.95, 0.95, current_time_text, transform=self.ax.transAxes,
                     fontsize=12, color='black', ha='right', va='top', fontweight='normal',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.1'))

        # Plot transbots' current positions with dynamic updates
        for transbot in self.factory_instance.agv:
            x, y = transbot.current_location
            transbot_color = self.transbot_color_map[f"Transbot {transbot.agv_id}"]
            # color = np.random.rand(3,)  # Random color for each transbot
            transbot_handle = self.ax.add_patch(pch.Circle((x, y), radius=0.3, color=transbot_color,
                                                           label=f'Transbot {transbot.agv_id} ({x}, {y})'))
            # self.transbot_handles.append(transbot_handle)
            # self.transbot_labels.append(f'Transbot {transbot.agv_id} ({x}, {y})')

        # Plot jobs' current positions with dynamic updates
        for job in self.scheduling_instance.jobs:
            if job.moving_location is None:
                x, y = self.factory_instance.factory_graph.pickup_dropoff_points[job.current_location]
            else:
                x, y = job.moving_location

            job_handle = self.ax.add_patch(pch.RegularPolygon((x, y), 3, radius=0.15, color='yellow',
                                                              label=f'Job {job.job_id} ({x}, {y})'))
            # self.job_handles.append(job_handle)
            # self.job_labels.append(f'Job {job.job_id} ({x}, {y})')

        # handles = self.job_handles + self.machine_handles + self.transbot_handles
        # labels = self.job_labels + self.machine_labels + self.transbot_labels

        # Remove duplicate labels from the legend (only show the first occurrence)
        handles, labels = self.ax.get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if any(x in label for x in ['charging', 'warehouse']):
                continue
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)

        self.ax.legend(unique_handles, unique_labels, loc='upper left',
                       bbox_to_anchor=(1.01, 1), borderaxespad=0., ncol=3)

    def plot_jobs_gantt(self, plot_end_time, plot_start_time=None, save_fig_dir=None):
        """
        Plot a Gantt chart with operations colored by their assigned resources (e.g., machines or transbots).

        Args:
            plot_start_time:
            plot_end_time:
            save_fig_dir (str, optional): If provided, saves the plot to this directory.
        """

        instance_jobs = self.scheduling_instance.jobs
        # Generate unique colors for resources using a colormap
        resource_ids = set(
            operation[2]  # resource assigned to the operation
            for job in instance_jobs
            for operation in job.scheduled_results
        )

        num_resources = len(resource_ids)
        colormap = plt.colormaps["tab20"] if num_resources <= 20 else cm.get_cmap("nipy_spectral", num_resources)
        resource_color_map = {resource: colormap(i / num_resources) for i, resource in enumerate(sorted(resource_ids))}

        # Find time window for the Gantt chart (from the earliest start time to the latest end time)
        # min_time = min(
        #     operation[2] for transbot in transbots for operation in transbot.scheduled_results
        # )
        min_time = self.local_result.time_window_start
        max_time = plot_end_time
        time_window = (min_time, max_time)

        # Create the Gantt chart figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))

        yticks = []
        yticklabels = []

        # Plot each job
        for i, job in enumerate(instance_jobs):
            yticks.append(i)
            yticklabels.append(f"Job {job.job_id}")

            if len(job.scheduled_results) % 2 != 0:
                job.scheduled_results.append((job.scheduled_results[-1][:3] + (plot_end_time,)))

            # Plot each operation for the current job
            for ops_id in range(0, len(job.scheduled_results), 2):
                op_type, op_id, resource, start_time = job.scheduled_results[ops_id]
                end_time = job.scheduled_results[ops_id + 1][3]

                # Determine color based on resource
                color = resource_color_map[resource]

                # Plot a rectangle for the operation
                ax.add_patch(
                    Rectangle(
                        (start_time, i - 0.15),  # (x, y) position for the left corner of the rectangle
                        end_time - start_time,  # width (duration)
                        0.3,  # height (job row height)
                        # color=color,
                        facecolor=color,
                        edgecolor="white",
                        linewidth=1,
                    )
                )

        # Configure axes
        ax.set_xlim(time_window[0], time_window[1])
        ax.set_ylim(-0.5, len(instance_jobs) - 0.5)
        ax.set_xlabel("Time")
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_title("Gantt Chart for Jobs")

        # Create a legend for the resources
        legend_patches = [Rectangle((0, 0), 1, 1, color=color) for resource, color in resource_color_map.items()]
        ax.legend(legend_patches, list(resource_color_map.keys()), title="Resources", loc="upper right",
                  bbox_to_anchor=(1.01, 1), borderaxespad=0., ncol=3)

        plt.tight_layout()

        # Save or display the plot
        if save_fig_dir is not None:
            plt.savefig(save_fig_dir + "_gantt.png")

        plt.show()

    def plot_machines_gantt(self, plot_end_time, plot_start_time=None, save_fig_dir=None):
        """
        Plot a Gantt chart with operations colored by jobs (and maintenance or charging).

        Args:
            plot_start_time:
            plot_end_time:
            save_fig_dir (str, optional): If provided, saves the plot to this directory.
        """

        machines = self.factory_instance.machines
        # Generate unique colors for resources using a colormap
        job_ids = set(
            operation[1]  # resource assigned to the operation
            for machine in machines
            for operation in machine.scheduled_results
            if operation[0] == "Processing"
        )

        num_jobs = len(job_ids)
        colormap = plt.colormaps["tab20"] if num_jobs <= 20 else cm.get_cmap("nipy_spectral", num_jobs)
        job_color_map = {job: colormap(i / num_jobs) for i, job in enumerate(sorted(job_ids))}

        # Find time window for the Gantt chart (from the earliest start time to the latest end time)
        # min_time = min(
        #     operation[2] for transbot in transbots for operation in transbot.scheduled_results
        # )
        min_time = self.local_result.time_window_start
        max_time = plot_end_time
        time_window = (min_time, max_time)

        # Create the Gantt chart figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))

        yticks = []
        yticklabels = []

        # Plot each machine
        for i, machine in enumerate(machines):
            yticks.append(i)
            yticklabels.append(f"Machine {machine.machine_id}")

            if len(machine.scheduled_results) % 2 != 0:
                machine.scheduled_results.append((machine.scheduled_results[-1][:3] + (plot_end_time,)))

            # Plot each operation for the current machine
            for ops_id in range(0, len(machine.scheduled_results), 2):
                op_type, job_id, start_time = machine.scheduled_results[ops_id]
                end_time = machine.scheduled_results[ops_id + 1][2]

                if op_type == "Maintenance":

                    # Plot a rectangle for the operation
                    rec_height = 0.3 + 0.1 * (job_id + 1)
                    ax.add_patch(
                        Rectangle(
                            (start_time, i - rec_height/2),  # (x, y) position for the left corner of the rectangle
                            end_time - start_time,  # width (duration)
                            rec_height,  # height (job row height)
                            color="grey"
                        )
                    )
                else:
                    # Determine color based on resource
                    color = job_color_map[job_id]

                    # Plot a rectangle for the operation
                    ax.add_patch(
                        Rectangle(
                            (start_time, i - 0.15),  # (x, y) position for the left corner of the rectangle
                            end_time - start_time,  # width (duration)
                            0.3,  # height (job row height)
                            color=color
                        )
                    )

        # Configure axes
        ax.set_xlim(time_window[0], time_window[1])
        ax.set_ylim(-0.5, len(machines) - 0.5)
        ax.set_xlabel("Time")
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_title("Gantt Chart for Machines")

        # Create a legend for the jobs
        legend_patches = [Rectangle((0, 0), 1, 1, color=color) for job, color in job_color_map.items()]
        ax.legend(legend_patches, list(job_color_map.keys()), title="Jobs", loc="upper right",
                  bbox_to_anchor=(1.01, 1), borderaxespad=0., ncol=3)

        plt.tight_layout()

        # Save or display the plot
        if save_fig_dir is not None:
            plt.savefig(save_fig_dir + "_gantt.png")

        plt.show()

    def plot_transbots_gantt(self, plot_end_time, plot_start_time=None, save_fig_dir=None):
        """
        Plot a Gantt chart with operations colored by jobs (and maintenance or charging).

        Args:
            plot_start_time:
            plot_end_time:
            save_fig_dir (str, optional): If provided, saves the plot to this directory.
        """

        transbots = self.factory_instance.agv
        # Generate unique colors for resources using a colormap
        job_ids = set(
            operation[1]  # resource assigned to the operation
            for transbot in transbots
            for operation in transbot.scheduled_results
            if operation[1] >= 0
        )

        num_jobs = len(job_ids)
        colormap = plt.colormaps["tab20"] if num_jobs <= 20 else cm.get_cmap("nipy_spectral", num_jobs)
        job_color_map = {job: colormap(i / num_jobs) for i, job in enumerate(sorted(job_ids))}

        # Find time window for the Gantt chart (from the earliest start time to the latest end time)
        # min_time = min(
        #     operation[2] for transbot in transbots for operation in transbot.scheduled_results
        # )
        min_time = self.local_result.time_window_start
        max_time = plot_end_time
        time_window = (min_time, max_time)

        # Create the Gantt chart figure and axis
        fig, ax = plt.subplots(figsize=(12, 8))

        yticks = []
        yticklabels = []

        # Plot each transbot
        for i, transbot in enumerate(transbots):
            yticks.append(i)
            yticklabels.append(f"Transbot {transbot.agv_id}")

            if len(transbot.scheduled_results) % 2 != 0:
                transbot.scheduled_results.append((transbot.scheduled_results[-1][:3] + (plot_end_time,)))

            # Plot each operation for the current transbot
            for ops_id in range(0, len(transbot.scheduled_results), 2):
                op_type, job_id, start_time = transbot.scheduled_results[ops_id]
                end_time = transbot.scheduled_results[ops_id + 1][2]

                if op_type == "Unload Transporting":
                    if job_id == -1:
                        ax.add_patch(
                            Rectangle(
                                (start_time, i - 0.25),  # (x, y) position for the left corner of the rectangle
                                end_time - start_time,  # width (duration)
                                0.5,  # height (job row height)
                                # color="grey"
                                facecolor="white",
                                edgecolor="grey",
                                linewidth=1
                            )
                        )
                    else:
                        # Determine color based on job
                        color = job_color_map[job_id]

                        # Plot a rectangle for the operation
                        ax.add_patch(
                            Rectangle(
                                (start_time, i - 0.15),  # (x, y) position for the left corner of the rectangle
                                end_time - start_time,  # width (duration)
                                0.3,  # height (job row height)
                                # color=color
                                facecolor="white",
                                edgecolor=color,
                                linewidth=1
                            )
                        )
                elif op_type == "Loaded Transporting":
                    # Determine color based on job
                    color = job_color_map[job_id]

                    # Plot a rectangle for the operation
                    ax.add_patch(
                        Rectangle(
                            (start_time, i - 0.15),  # (x, y) position for the left corner of the rectangle
                            end_time - start_time,  # width (duration)
                            0.3,  # height (job row height)
                            color=color
                        )
                    )
                elif op_type == "Charging":
                    ax.add_patch(
                        Rectangle(
                            (start_time, i - 0.25),  # (x, y) position for the left corner of the rectangle
                            end_time - start_time,  # width (duration)
                            0.5,  # height (job row height)
                            color="grey"
                        )
                    )


        # Configure axes
        ax.set_xlim(time_window[0], time_window[1])
        ax.set_ylim(-0.5, len(transbots) - 0.5)
        ax.set_xlabel("Time")
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_title("Gantt Chart for Transbots")

        # Create a legend for the jobs
        legend_patches = [Rectangle((0, 0), 1, 1, color=color) for job, color in job_color_map.items()]
        ax.legend(legend_patches, list(job_color_map.keys()), title="Jobs", loc="upper right",
                  bbox_to_anchor=(1.01, 1), borderaxespad=0., ncol=3)

        plt.tight_layout()

        # Save or display the plot
        if save_fig_dir is not None:
            plt.savefig(save_fig_dir + "_gantt.png")

        plt.show()


# Example usage functions for testing
def run_one_local_instance_single_agent(
        local_instance_file,
        num_episodes=10,
        do_render=False,
        do_plot_gantt=True,
        detailed_log=True,
):
    """Run episodes using the single-agent environment."""
    print(f"Starting instance {local_instance_file} with SINGLE AGENT:\n")
    
    config = {
        "n_machines": dfjspt_params.n_machines,
        "n_transbots": dfjspt_params.n_transbots,
        "factory_instance_seed": dfjspt_params.factory_instance_seed,
        "render_mode": "human" if do_render else None,
    }
    
    scheduling_env = LocalSchedulingSingleAgentEnv(config)
    makespans = []
    
    for episode_id in range(num_episodes):
        makespans = run_one_episode_single_agent(
            episode_id=episode_id+1,
            scheduling_env=scheduling_env,
            makespans=makespans,
            do_render=do_render,
            do_plot_gantt=do_plot_gantt,
            local_instance_file=local_instance_file,
            detailed_log=detailed_log,
        )
    
    print(f"\nMin makespan across {num_episodes} episodes is {np.min(makespans)}.")
    print(f"Average makespan across {num_episodes} episodes is {np.average(makespans)}.")


def run_one_episode_single_agent(
        episode_id, scheduling_env,
        makespans,
        do_render=False,
        do_plot_gantt=True,
        local_instance_file=None,
        detailed_log=True,
):
    """Run one episode using the single-agent environment."""
    from local_realtime_scheduling.Agents.generate_training_data import generate_reset_options_for_training
    
    print(f"\nStarting episode {episode_id}:")
    decision_count = 0
    
    env_reset_options = generate_reset_options_for_training(
        local_schedule_filename=local_instance_file,
        for_training=False,
    )
    
    observation, info = scheduling_env.reset(options=env_reset_options)
    if do_render:
        scheduling_env.render()

    done = False
    truncated = False
    total_reward = 0.0
    
    while not done and not truncated:
        # Extract action mask
        action_mask = observation['action_mask']
        
        # Generate actions for all agents based on the global action mask
        actions = []
        
        # Machine actions (first part of action vector)
        for machine_id in range(scheduling_env.num_machines):
            mask_start = machine_id * scheduling_env.num_machine_actions
            mask_end = mask_start + scheduling_env.num_machine_actions
            agent_mask = action_mask[mask_start:mask_end]
            
            valid_actions = [i for i, valid in enumerate(agent_mask) if valid == 1]
            if valid_actions:
                if len(valid_actions) > 1:
                    valid_actions.pop(-1)  # Remove do-nothing if other actions available
                actions.append(np.random.choice(valid_actions))
            else:
                actions.append(scheduling_env.num_machine_actions - 1)  # Do-nothing
        
        # Transbot actions (second part of action vector)  
        transbot_mask_start = scheduling_env.num_machines * scheduling_env.num_machine_actions
        for transbot_id in range(scheduling_env.num_transbots):
            mask_start = transbot_mask_start + transbot_id * scheduling_env.num_transbot_actions
            mask_end = mask_start + scheduling_env.num_transbot_actions
            agent_mask = action_mask[mask_start:mask_end]
            
            valid_actions = [i for i, valid in enumerate(agent_mask) if valid == 1]
            if valid_actions:
                if len(valid_actions) > 1:
                    valid_actions.pop(-1)  # Remove do-nothing if other actions available
                actions.append(np.random.choice(valid_actions))
            else:
                actions.append(scheduling_env.num_transbot_actions - 1)  # Do-nothing
        
        action = np.array(actions)
        observation, reward, done, truncated, info = scheduling_env.step(action)
        decision_count += 1
        
        if do_render:
            scheduling_env.render()

        total_reward += reward

    # print(f"Episode length is {decision_count}.")
    if do_render:
        scheduling_env.close()
    if do_plot_gantt:
        scheduling_env.plot_jobs_gantt(
            plot_end_time=scheduling_env.current_time_after_step,
            save_fig_dir=None,
        )
        scheduling_env.plot_machines_gantt(
            plot_end_time=scheduling_env.current_time_after_step,
            save_fig_dir=None,
        )
        scheduling_env.plot_transbots_gantt(
            plot_end_time=scheduling_env.current_time_after_step,
            save_fig_dir=None,
        )
    
    if detailed_log:
        print(f"Current timestep is {scheduling_env.current_time_after_step}.")
        if scheduling_env.local_result.actual_local_makespan is not None:
            print(f"Actual makespan = {scheduling_env.local_result.actual_local_makespan}")
            print(f"Actual delta makespan = {scheduling_env.local_result.actual_local_makespan - scheduling_env.local_result.time_window_start}")
            makespans.append(scheduling_env.local_result.actual_local_makespan - scheduling_env.local_result.time_window_start)
        else:
            makespans.append(scheduling_env.current_time_after_step - scheduling_env.local_result.time_window_start)
            print(f"Truncated makespan = {scheduling_env.current_time_after_step}")
            print(f"Truncated delta makespan = {scheduling_env.current_time_after_step - scheduling_env.local_result.time_window_start}")
        
        print(f"Total reward for episode {episode_id}: {total_reward}")
        print(f"Min makespan up to now is {np.min(makespans)}.")
        print(f"Average makespan up to now is {np.average(makespans)}.")
    else:
        if scheduling_env.local_result.actual_local_makespan is not None:
            makespans.append(scheduling_env.local_result.actual_local_makespan - scheduling_env.local_result.time_window_start)
        else:
            makespans.append(scheduling_env.current_time_after_step - scheduling_env.local_result.time_window_start)
    
    return makespans


if __name__ == "__main__":
    import time
    
    # Test parameters
    instance_n_jobs = 10
    scheduling_instance_id = 100
    current_window = 0
    num_ops = 66
    
    file_name = f"local_schedule_J{instance_n_jobs}I{scheduling_instance_id}_{current_window}_ops{num_ops}.pkl"
    
    # Test single-agent environment
    start_time = time.time()
    run_one_local_instance_single_agent(
        local_instance_file=file_name,
        # do_plot_gantt=True,
        do_plot_gantt=False,
        num_episodes=10
    )
    print(f"Total running time for single-agent: {time.time() - start_time}") 