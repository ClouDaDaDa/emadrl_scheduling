# from memory_profiler import profile
# @profile
def func(content: str):
    print(content)

import os
import pickle
import random
from gymnasium import spaces
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as pch
import matplotlib.cm as cm
# from matplotlib import animation
# import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import logging
logging.basicConfig(level=logging.INFO)

func("import part 1")

# from System.SchedulingInstance import SchedulingInstance
from System.FactoryInstance import FactoryInstance
from configs import dfjspt_params
from local_realtime_scheduling.Environment.ExecutionResult import LocalResult, Local_Job_result, Operation_result
from local_realtime_scheduling.Environment.path_planning import a_star_search
from local_realtime_scheduling.InterfaceWithGlobal.divide_global_schedule_to_local_from_pkl import LocalSchedule, \
    Local_Job_schedule

func("import part 2")

from ray.rllib.env.multi_agent_env import MultiAgentEnv

func("import part 3")

MAX_PRCS_TIME = dfjspt_params.max_prcs_time
MAX_TSPT_TIME = dfjspt_params.max_tspt_time


class LocalSchedulingMultiAgentEnv(MultiAgentEnv):
    """
    A Multi-agent Environment for Integrated Production, Transportation and Maintenance Real-time Scheduling.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
                 config,
                 ):
        """
        :param config: including:
        n_machines
        n_transbots
        factory_instance_seed
        """
        super().__init__()

        func("Env initialized.")

        # Initialize parameters
        # self.num_jobs = None
        self.num_machines = config["n_machines"]
        self.num_transbots = config["n_transbots"]

        # Dynamic agent filtering configuration
        self.enable_dynamic_agent_filtering = config.get("enable_dynamic_agent_filtering", True)
        self.no_obs_solution_type = config.get("no_obs_solution_type", None)

        # Add fast-forward protection
        self.max_fast_forward_steps = 500
        self.fast_forward_counter = 0

        # self.local_schedule = config["local_schedule"]
        # self.local_result_file = config["local_result_file"] if "local_result_file" in config else None
        # self.local_schedule = None
        # self.local_result_file = None
        # self.local_result = None

        self.factory_instance = FactoryInstance(
            seed=config["factory_instance_seed"],
            n_machines=self.num_machines,
            n_transbots=self.num_transbots,
        )
        # self.scheduling_instance = None

        # Light Maintenance (LM), Middle Maintenance (MM), Overhaul (OH), and Corrective Maintenance (CM)
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

        # Define agent IDs
        self.machine_agents = [f"machine{i}" for i in range(self.num_machines)]
        self.transbot_agents = [f"transbot{i}" for i in range(self.num_transbots)]
        self.agents = self.possible_agents = self.machine_agents + self.transbot_agents

        self.decision_stage = 0  # 0 for machines and 1 for transbots
        self.all_agents_have_made_decisions = False
        # # Initialize the index lists
        # self.machine_index = list(range(self.num_machines))
        # self.transbot_index = list(range(self.num_transbots))
        # self.machine_action_agent_count = 0
        # self.transbot_action_agent_count = 0
        # self.current_actor = None

        self.terminateds = set()
        self.truncateds = set()
        self.resetted = False
        self.rewards = {}

        # Define observation and action spaces for each agent type
        self.observation_spaces = {}
        self.action_spaces = {}

        if self.enable_dynamic_agent_filtering and self.no_obs_solution_type == "dummy_agent":
            self.dummy_agent = "dummy"
            self.agents = self.possible_agents = self.machine_agents + self.transbot_agents + [self.dummy_agent]
            self.observation_spaces[self.dummy_agent] = spaces.Box(
                low=-1.0, high=1.0,
                shape=(1,)
            )
            self.action_spaces[self.dummy_agent] = spaces.Discrete(1)

        # local view for a machine agent
        # Number of jobs that can be observed and selected by a machine
        self.n_jobs_handled_by_machine = 20
        self.jobs_handled_by_machines = -np.ones((self.num_machines, self.n_jobs_handled_by_machine), dtype=np.int32)
        # Number of machines that can be observed by a machine
        self.n_neighbor_machines = min(5, self.num_machines - 1)
        self.neighbor_machines = -np.ones((self.num_machines, self.n_neighbor_machines), dtype=np.int32)
        # Machines: Observation space and action space
        # possible actions:
        # [0,n_jobs-1] for start processing, [n_jobs,n_jobs+3] for start maintenance, n_jobs+4 for do nothing
        self.num_machine_actions = self.n_jobs_handled_by_machine + 5
        self.n_job_features_for_machine = 7
        # self.n_job_features_for_machine = 6
        self.n_neighbor_machines_features = 6
        self.n_machine_features = 8
        for machine_agent_id in self.machine_agents:
            # observations:
            self.observation_spaces[machine_agent_id] = spaces.Dict({
                # Indicating which actions are valid: 1 for valid, 0 for invalid
                "action_mask": spaces.Box(
                    0, 1, shape=(self.num_machine_actions,), dtype=np.int32
                ),
                "observation": spaces.Dict({
                    # Features of the observed jobs
                    "job_features": spaces.Box(
                        low=-1.0, high=float('inf'),
                        shape=(self.n_jobs_handled_by_machine, self.n_job_features_for_machine)
                    ),
                    # Features of neighbor machines
                    "neighbor_machines_features": spaces.Box(
                        low=-float('inf'), high=float('inf'),
                        shape=(self.n_neighbor_machines, self.n_neighbor_machines_features)
                    ),
                    # The current machine's own features
                    "machine_features": spaces.Box(
                        low=-float('inf'), high=float('inf'),
                        shape=(self.n_machine_features,)
                    ),
                }),
            })
            self.action_spaces[machine_agent_id] = spaces.Discrete(self.num_machine_actions)

        # local view for a transbot agent
        # Number of jobs that can be observed and selected by a transbot
        self.n_jobs_handled_by_transbot = 20
        self.jobs_handled_by_transbots = -np.ones((self.num_transbots, self.n_jobs_handled_by_transbot), dtype=np.int32)
        # Number of transbots that can be observed by a transbot
        self.n_neighbor_transbots = min(5, self.num_transbots - 1)
        self.neighbor_transbots = -np.ones((self.num_transbots, self.n_neighbor_transbots), dtype=np.int32)
        # Transbots: Observation space and action space
        # possible actions:
        # [0,num_jobs-1] for start transportation, num_jobs for start charging, num_jobs+1 for do nothing
        self.num_transbot_actions = self.n_jobs_handled_by_transbot + 2
        self.n_job_features_for_transbot = 8
        # self.n_job_features_for_transbot = 7
        self.n_neighbor_transbots_features = 8
        self.n_transbot_features = 11
        for transbot_agent_id in self.transbot_agents:
            # observations:
            self.observation_spaces[transbot_agent_id] = spaces.Dict({
                # Indicating which actions are valid: 1 for valid, 0 for invalid
                "action_mask": spaces.Box(0, 1,
                                          shape=(self.num_transbot_actions,), dtype=np.int32),
                "observation": spaces.Dict({
                    # Features of the observed jobs
                    "job_features": spaces.Box(
                        low=-1.0, high=float('inf'),
                        shape=(self.n_jobs_handled_by_transbot, self.n_job_features_for_transbot)
                    ),
                    # Features of neighbor transbots
                    "neighbor_transbots_features": spaces.Box(
                        low=-float('inf'), high=float('inf'),
                        shape=(self.n_neighbor_transbots, self.n_neighbor_transbots_features)
                    ),
                    # The current transbot's own features
                    "transbot_features": spaces.Box(
                        low=-float('inf'), high=float('inf'),
                        shape=(self.n_transbot_features,)
                    ),
                }),
            })
            self.action_spaces[transbot_agent_id] = spaces.Discrete(self.num_transbot_actions)

        # Rendering settings
        self.render_mode = config.get("render_mode", None)
        self.fig, self.ax = None, None
        if self.render_mode in ["human", "rgb_array"]:
            self._initialize_rendering()

    def _initialize_state(self):
        """Initialize the observation state for all machine agents."""
        obs = {}
        for machine_id in self.machine_agents:
            machine_obs = self._get_machine_obs(machine_id)
            if "observation" in machine_obs:
                obs[machine_id] = machine_obs
        return obs

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
                    this_job.job_progress >= 1.0 or
                    this_job.operations_matrix[current_op, machine_index] <= 0):
                job_scores[job_id] = -1e5
            else:
                job_scores[job_id] = (
                        - (this_job.operations_matrix[current_op, machine_index] + unload_time_matrix[
                            location_map[this_job.current_location], machine_index]) / (MAX_PRCS_TIME + MAX_TSPT_TIME)
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

    def _get_machine_obs(self, machine_agent_id, done=False):
        """Retrieve machine-specific observations including job features and neighbor machine features."""

        machine_index = int(machine_agent_id.lstrip("machine"))
        machine = self.factory_instance.machines[machine_index]

        # dynamic action masking (DAM) logic for the machine:
        # 1 for valid and 0 for invalid action
        machine_action_mask = np.zeros((self.num_machine_actions,), dtype=np.int32)
        # idling: can choose a job, or perform a maintenance except CM, or do nothing
        if machine.machine_status == 0:
            machine_action_mask[self.n_jobs_handled_by_machine:self.n_jobs_handled_by_machine + 3] = 1
            # machine_action_mask[self.n_jobs_handled_by_machine + 3] = 0
            machine_action_mask[self.n_jobs_handled_by_machine + 4] = 1
        # processing or under maintenance: can choose a job, or do nothing
        elif machine.machine_status == 1 or machine.machine_status == 2:
            machine_action_mask[self.n_jobs_handled_by_machine + 4] = 1
        # faulty: can perform CM or do nothing
        elif machine.machine_status == 3:
            machine_action_mask[self.n_jobs_handled_by_machine + 3:] = 1

        if machine.reliability >= 0.8:
            machine_action_mask[self.n_jobs_handled_by_machine:self.n_jobs_handled_by_machine + 4] = 0

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
                if job_id in jobs:  # the job is in this problem
                    this_job = scheduling_jobs[job_id]
                    current_op = this_job.current_processing_operation

                    # the job still has operation to be processed
                    # the job's next pending operation has not been assigned to a machine
                    if this_job.n_p_ops_for_curr_tw - current_op > 0 and this_job.assigned_machine is None and this_job.job_progress <1.0:
                        # this machine can handle the job's next pending operation
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
                                raise ValueError(f"Inalid job{this_job.job_id}'s status {this_job.job_status}!")

                            job_features[job_action_id] = [
                                job_id,  # [0] job_id
                                this_job.job_status,  # [1] job's internal status
                                this_job.job_progress_for_current_time_window,  # [2] job's progress
                                # [3] job's estimated remaining time to finish the operation
                                job_remaining_finish_time,
                                # [4] processing time for this machine to handle this operation
                                this_job.operations_matrix[current_op, machine_index],
                                # [5] distance from this machine to this job
                                unload_time_matrix[location_map[this_job.current_location], machine_index],
                                job_waiting_time,
                            ]
                            # if machine.machine_status in {0, 1, 2} and machine.current_processing_task is None:
                            if machine.machine_status != 3:
                                machine_action_mask[job_action_id] = 1

        if self.enable_dynamic_agent_filtering:
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
            machine.machine_status,  # [0] machine status
            machine.reliability,  # [1] reliability
            -1 if machine.current_processing_task is None else machine.current_processing_task,  # [2] current task's id
            machine.estimated_remaining_time_to_finish,  # [3] time to finish the current task
            machine.dummy_work_time / max(machine.dummy_total_time, 1.0),  # [4] machine's utilization
            machine.cumulative_tasks,  # [5] cumulative number of tasks
            self.initial_estimated_makespan - self.current_time_after_step,  # [6] The overall time progress
            self.remaining_operations,  # [7] Global feature of the number of remaining operations
        ], dtype=np.float32)

        return {
            "action_mask": machine_action_mask,
            "observation": {
                "job_features": job_features,
                "neighbor_machines_features": neighbor_machines_features,
                "machine_features": machine_features,
            }
        }

    def _update_job_index_queue_for_transbots(self, transbot_index):
        current_transbot = self.factory_instance.agv[transbot_index]
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
                job_remaining_finish_time = self.factory_instance.machines[
                    machine_id].estimated_remaining_time_to_finish
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
        similarities = np.full(self.num_transbots, -1.0)

        # Compare k1 with every other transbot
        for transbot_2 in range(self.num_transbots):
            if transbot_2 != transbot_1:
                similarities[transbot_2] = self._transbot_kendall_tau(transbot_1, transbot_2)

        # Get indices of the top n most similar machines (excluding k1 itself)
        top_n_indices = np.argsort(similarities)[-self.n_neighbor_transbots:][::-1]  # Sort and pick top n

        return top_n_indices

    def _get_transbot_obs(self, transbot_agent_id, done=False):
        transbot_index = int(transbot_agent_id.lstrip("transbot"))
        transbot = self.factory_instance.agv[transbot_index]

        # dynamic action masking logic for the transbot: 1 for valid and 0 for invalid action
        transbot_action_mask = np.zeros((self.num_transbot_actions,), dtype=np.int32)
        # idling (0): can choose a job, or go to charge, or do nothing
        if transbot.agv_status == 0:
            transbot_action_mask[self.n_jobs_handled_by_transbot:] = 1
        # unload transporting (1): can change its task or insist the current task
        elif transbot.agv_status == 1:
            transbot_action_mask[self.n_jobs_handled_by_transbot:] = 1
        # loaded transporting (2) or charging (3): can only do nothing
        elif transbot.agv_status == 2 or transbot.agv_status == 3:
            transbot_action_mask[self.n_jobs_handled_by_transbot + 1] = 1
        # low battery (4): can only go to charge
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
                if job_id not in self.local_schedule.jobs:  # the job is not in this problem
                    continue

                job = self.scheduling_instance.jobs[job_id]
                if job.job_status in [2, 3] or job.assigned_machine is None:
                    continue
                # the job still has operation to be processed
                # the job's next pending operation has been assigned to a machine
                # the job is not in transporting
                job_location_index = location_map[job.current_location]
                job_to_machine = unload_time_matrix[job_location_index, job.assigned_machine]

                if job_to_machine <= 0 or job.assigned_transbot is not None:
                    continue
                # the job is not at its destination and needs to be transported
                # it hasn't been assigned to another transbot
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
                    job_remaining_finish_time = self.factory_instance.machines[
                        machine_id].estimated_remaining_time_to_finish
                    job_waiting_time = 0.0
                else:
                    raise ValueError(f"Inalid job{job.job_id}'s status {job.job_status}!")

                job_features[job_action_id] = [
                    job_id,  # [0] job_id
                    job.job_status,  # [1] job's internal status
                    job.job_progress_for_current_time_window,  # [2] job's progress
                    job_remaining_finish_time,  # [3]
                    job_location[0],
                    job_location[1],
                    transbot_to_job,  # [6] transport time for this transbot to handle this operation
                    job_waiting_time,
                ]

                # if transbot.agv_status in (0, 1) and not transbot.is_for_charging and transbot.current_task is None:
                if (
                        # transbot.agv_status in (0, 1) and
                        not transbot.is_for_charging and
                        not transbot.finish_unload
                ):
                    # if not transbot.is_for_charging:
                    transbot_action_mask[job_action_id] = 1

        if self.enable_dynamic_agent_filtering:
            if max(transbot_action_mask[:self.num_transbot_actions - 1]) <= 0 and not done:
                return {
                    "action_mask": transbot_action_mask,
                }

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
            transbot.agv_status,  # [0] transbot status
            -1 if transbot.current_task is None else transbot.current_task,  # [1] current task's id
            transbot.current_location[0],  # [2] current location x
            transbot.current_location[1],  # [3] current location y
            transbot.battery.soc,  # [4] battery's SOC
            transbot.estimated_remaining_time_to_finish,  # [5] time to finish the current task
            transbot.t_since_prev_r,  # [6] time since previous get a positive reward
            transbot.dummy_work_time / max(transbot.dummy_total_time, 1.0),  # [7] transbot's utilization
            transbot.cumulative_tasks,  # [8] transbot's cumulative tasks
            self.initial_estimated_makespan - self.current_time_after_step,
            # [9] Global feature of the overall time progress
            self.remaining_operations,  # [10] Global feature of the number of remaining operations
        ], dtype=np.float32)

        return {
            "action_mask": transbot_action_mask,
            "observation": {
                "job_features": job_features,
                "neighbor_transbots_features": neighbor_transbots_features,
                "transbot_features": transbot_features,
            }
        }

    def reset(self, seed=None, options=None):
        """
        Reset the environment.

        :param options: A dict including:
            - factory_instance: FactoryInstance
            - scheduling_instance: SchedulingInstance
            - local_schedule: LocalSchedule
            - current_window: int
            - start_t_for_curr_time_window: float
            - local_result_file: str or None
        :return: observations, infos
        """

        func("Env reset.")

        if options is None:
            raise ValueError(f"Options must be provided with scheduling instances!")

        self.resetted = True
        self.terminateds.clear()
        self.truncateds.clear()
        self.rewards = {}
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Reset filtering statistics and fast-forward counter for new episode
        self.fast_forward_counter = 0

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
        self.has_saved_instance_snapshot = False
        self.num_jobs = len(self.local_schedule.jobs)
        self.snapshot = None
        # self.max_distance = np.max(self.factory_instance.factory_graph.unload_transport_time_matrix)
        # self.max_operations = max(job.operations_matrix.shape[0] for job in self.scheduling_instance.jobs)
        self.initial_estimated_makespan = self.local_schedule.local_makespan + self.time_deviation
        self.time_upper_bound = dfjspt_params.episode_time_upper_bound
        self.reward_this_step = 0.0
        self.total_cost = 0.0

        self.decision_stage = 0
        self.all_agents_have_made_decisions = False
        # # Randomly shuffle the index lists
        # random.shuffle(self.machine_index)
        # random.shuffle(self.transbot_index)
        # self.machine_action_agent_count = 0
        # self.transbot_action_agent_count = 0

        # # Reset all machines and jobs
        # for machine in self.factory_instance.machines:
        #     # machine.reset_machine_for_current_time_window()
        #     machine.reset_machine()
        #
        # for agv in self.factory_instance.agv:
        #     # agv.reset_agv_for_current_time_window()
        #     agv.reset_agv()
        #
        # for job in self.scheduling_instance.jobs:
        #     # job.reset_job_for_current_time_window()
        #     job.reset_job()

        # self.total_n_ops_for_curr_tw = 0
        self.total_n_ops_for_curr_tw = self.local_schedule.n_ops_in_tw
        self.remaining_operations = self.total_n_ops_for_curr_tw

        for job_id, local_schedule_job in self.local_schedule.jobs.items():
            job_result = Local_Job_result(job_id=job_id)
            self.local_result.add_job_result(job_result)

            this_job = self.scheduling_instance.jobs[job_id]
            # todo: how to check if the job is available
            this_job.at_for_curr_tw = local_schedule_job.available_time + self.time_deviation
            this_job.eft_for_curr_tw = local_schedule_job.estimated_finish_time + self.time_deviation
            # self.total_n_ops_for_curr_tw += len(local_schedule_job.operations)

            # todo: how to handle ops with ops.is_addition_ops = True?
            # for operation_id, operation in local_schedule_job.operations.items():
            for operation_id in local_schedule_job.operations:
                if local_schedule_job.operations[operation_id].is_current_window:
                    job_result.add_operation_result(Operation_result(
                        job_id=job_id,
                        operation_id=operation_id,
                        # actual_start_transporting_time=None,
                        # actual_finish_transporting_time=None,
                        # assigned_transbot=None,
                        # actual_start_processing_time=None,
                        # actual_finish_processing_time=None,
                        # assigned_machine=None,
                    ))
                    this_job.p_ops_for_cur_tw.append(int(operation_id))

            if len(this_job.p_ops_for_cur_tw) > 0:
                this_job.reset_job_for_current_time_window()

        # if self.enable_dynamic_agent_filtering:
        #     # Use dynamic filtering to determine initial observations
        #     acting_machines, forced_machine_agents = self._get_acting_agents_for_next_stage()
        #     observations = {machine_id: self._get_machine_obs(machine_id) for machine_id in acting_machines}
        # else:
        #     # Original behavior: all machines get observations
        #     observations = self._initialize_state()

        # all machines get observations
        observations = self._initialize_state()

        infos = self._get_info()

        return observations, infos

    def step(self, action_dict):
        observations, terminated, truncated, infos = {}, {}, {}, {}
        self.reward_this_step = 0.0
        self.resetted = False
        self.all_agents_have_made_decisions = False

        # Process actions for agents that were in the previous observation dict
        # Convert the dictionary to a list of tuples and shuffle it
        shuffled_action_items = list(action_dict.items())
        random.shuffle(shuffled_action_items)

        # Process each agent's action
        # Decision Stage 0: Machines should make decisions
        if self.decision_stage == 0:
            # Set rewards for all machine agents
            for machine_agent_id in self.machine_agents:
                if machine_agent_id not in self.rewards or len(action_dict) > 0:
                    self.rewards[machine_agent_id] = 0.0

            if self.enable_dynamic_agent_filtering:
                # Execute forced actions for machines that don't have meaningful choices
                for agent in self.machine_agents:
                    if agent not in action_dict:
                        machine_index = int(agent.lstrip("machine"))
                        current_machine = self.factory_instance.machines[machine_index]
                        self._handle_machine_action(machine_index, current_machine, self.num_machine_actions - 1)

            # Process actions from agents that had meaningful choices
            for agent_id, action in shuffled_action_items:
                if agent_id.startswith("machine"):
                    machine_index = int(agent_id.lstrip("machine"))
                    current_machine = self.factory_instance.machines[machine_index]
                    # if len(action_dict) > 0:
                    #     self.rewards[agent_id] = 0.0
                    self._handle_machine_action(machine_index, current_machine, int(action))
                else:
                    if self.no_obs_solution_type != "dummy_agent":
                        raise Exception(f"Only machines can action in decision_stage {self.decision_stage}!")

            # all transbots get observations
            for transbot_agent_id in self.transbot_agents:
                transbot_obs = self._get_transbot_obs(transbot_agent_id=transbot_agent_id)
                if "observation" in transbot_obs:
                    observations[transbot_agent_id] = transbot_obs
                if transbot_agent_id not in self.rewards or len(action_dict) > 0:
                    self.rewards[transbot_agent_id] = 0.0

        # Decision Stage 1: Transbots should make decisions
        else:
            # Environment steps forward for 1 time step
            self.all_agents_have_made_decisions = True
            self.current_time_after_step += 1.0

            # if self.current_time_after_step - self.local_result.time_window_start > 1300:
            #     print("too many steps!")

            if len(action_dict) > 0:
                for transbot_agent_id in self.transbot_agents:
                    self.rewards[transbot_agent_id] = 0.0
            if self.enable_dynamic_agent_filtering:
                # Execute forced actions for transbots that don't have meaningful choices
                for agent in self.transbot_agents:
                    if agent not in action_dict:
                        transbot_index = int(agent.lstrip("transbot"))
                        current_transbot = self.factory_instance.agv[transbot_index]
                        self._handle_transbot_action(transbot_index, current_transbot, self.num_transbot_actions - 1)

            # Process actions from agents that had meaningful choices
            for agent_id, action in shuffled_action_items:
                if agent_id.startswith("transbot"):
                    transbot_index = int(agent_id.lstrip("transbot"))
                    current_transbot = self.factory_instance.agv[transbot_index]
                    # if len(action_dict) > 0:
                    #     self.rewards[agent_id] = 0.0
                    self._handle_transbot_action(transbot_index, current_transbot, int(action))
                else:
                    if self.no_obs_solution_type != "dummy_agent":
                        raise Exception(f"Only transbots can action in decision_stage {self.decision_stage}!")

            # Step all machines and transbots forward
            random.shuffle(self.machine_agents)
            for machine_agent_id in self.machine_agents:
                if len(action_dict) > 0:
                    self.rewards[machine_agent_id] = 0.0
                machine_index = int(machine_agent_id.lstrip("machine"))
                self._step_a_machine_for_one_step(machine_index=machine_index)

            random.shuffle(self.transbot_agents)
            for transbot_agent_id in self.transbot_agents:
                # if transbot_agent_id not in self.rewards:
                #     self.rewards[transbot_agent_id] = 0.0
                transbot_index = int(transbot_agent_id.lstrip("transbot"))
                self._step_a_transbot_for_one_step(transbot_index=transbot_index)

            self.current_time_before_step = self.current_time_after_step
            # Apply time step penalty
            self.reward_this_step -= 5.0 / (self.initial_estimated_makespan - self.local_result.time_window_start)

            # Check termination and truncation
            is_terminated = self._check_done()
            is_truncated = self._check_truncated()

            if self.enable_dynamic_agent_filtering and self.no_obs_solution_type == "dummy_agent":
                terminated[self.dummy_agent] = is_terminated
                if terminated[self.dummy_agent]:
                    self.terminateds.add(self.dummy_agent)
                    observations[self.dummy_agent] = np.array([0.], dtype=np.float32)
                    self.rewards[self.dummy_agent] = 0.0

                truncated[self.dummy_agent] = is_truncated
                if truncated[self.dummy_agent]:
                    self.truncateds.add(self.dummy_agent)
                    observations[self.dummy_agent] = np.array([0.], dtype=np.float32)
                    self.rewards[self.dummy_agent] = 0.0

            # Update rewards and termination status for all agents
            for machine_agent_id in self.machine_agents:
                machine_obs = self._get_machine_obs(machine_agent_id=machine_agent_id)
                if "observation" in machine_obs:
                    observations[machine_agent_id] = machine_obs

                self.rewards[machine_agent_id] += self.reward_this_step
                terminated[machine_agent_id] = is_terminated
                if terminated[machine_agent_id]:
                    self.terminateds.add(machine_agent_id)
                    self.rewards[machine_agent_id] += 5.0
                    observations[machine_agent_id] = self._get_machine_obs(machine_agent_id=machine_agent_id, done=True)
                truncated[machine_agent_id] = is_truncated
                if truncated[machine_agent_id]:
                    self.truncateds.add(machine_agent_id)
                    observations[machine_agent_id] = self._get_machine_obs(machine_agent_id=machine_agent_id, done=True)

            for transbot_agent_id in self.transbot_agents:
                self.rewards[transbot_agent_id] += self.reward_this_step
                terminated[transbot_agent_id] = is_terminated
                if terminated[transbot_agent_id]:
                    self.terminateds.add(transbot_agent_id)
                    self.rewards[transbot_agent_id] += 5.0
                    observations[transbot_agent_id] = self._get_transbot_obs(transbot_agent_id=transbot_agent_id,
                                                                             done=True)
                truncated[transbot_agent_id] = is_truncated
                if truncated[transbot_agent_id]:
                    self.truncateds.add(transbot_agent_id)
                    observations[transbot_agent_id] = self._get_transbot_obs(transbot_agent_id=transbot_agent_id,
                                                                             done=True)

        # Update decision stage
        self.decision_stage = 1 - self.decision_stage
        infos = self._get_info()
        terminated["__all__"] = len(self.terminateds) == len(self.agents)
        truncated["__all__"] = len(self.truncateds) == len(self.agents)

        if terminated["__all__"]:
            self._log_episode_termination()

        # Safety check: if no observations and dynamic filtering is enabled, provide fallback observations
        if (self.enable_dynamic_agent_filtering and not observations and
                not terminated["__all__"] and not truncated["__all__"]):

            if self.no_obs_solution_type == "fast_forward":

                # Check if we've exceeded the maximum fast-forward steps
                if self.fast_forward_counter >= self.max_fast_forward_steps:

                    # Force termination to prevent infinite recursion
                    print(
                        f"Warning: Fast-forward limit ({self.max_fast_forward_steps}) reached. Forcing episode termination.")
                    for agent_id in self.agents:
                        terminated[agent_id] = False
                        truncated[agent_id] = True
                        self.rewards[agent_id] = -10
                        observations[agent_id] = self._get_machine_obs(agent_id, done=True) if agent_id.startswith(
                            "machine") else self._get_transbot_obs(agent_id, done=True)
                    terminated["__all__"] = False
                    truncated["__all__"] = True
                    return observations, self.rewards, terminated, truncated, infos

                # Increment fast-forward counter and continue recursion
                self.fast_forward_counter += 1

                # Fast-forward by recursively calling step with empty action dict
                # This will continue until agents have meaningful choices or episode ends
                return self.step({})

            elif self.no_obs_solution_type == "dummy_agent":
                observations[self.dummy_agent] = np.array([0.], dtype=np.float32)
                self.rewards[self.dummy_agent] = 0.0

            else:
                raise ValueError(f"Invalid no_obs_solution_type {self.no_obs_solution_type}!")

        # # Final safety check: ensure we always return some observations for training
        # if not observations and not terminated["__all__"] and not truncated["__all__"]:
        #     # Provide observations for the appropriate agent type based on decision stage
        #     if self.decision_stage == 0:  # Next stage is machines
        #         for machine_agent_id in self.machine_agents:
        #             machine_obs = self._get_machine_obs(machine_agent_id)
        #             if "observation" in machine_obs:
        #                 observations[machine_agent_id] = machine_obs
        #                 break  # Just need one agent to prevent empty dict
        #     else:  # Next stage is transbots
        #         for transbot_agent_id in self.transbot_agents:
        #             transbot_obs = self._get_transbot_obs(transbot_agent_id)
        #             if "observation" in transbot_obs:
        #                 observations[transbot_agent_id] = transbot_obs
        #                 break  # Just need one agent to prevent empty dict

        # Reset fast-forward counter when we complete a full decision cycle
        self.fast_forward_counter = 0

        return observations, self.rewards, terminated, truncated, infos

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
                        # print(
                        #     f"Machine {machine_index} finishes processing Job {current_job.job_id} at time {self.current_time_after_step}.")
                        # self.rewards[f'machine{current_machine.machine_id}'] += 0.1
                        # self.rewards[f'machine{current_machine.machine_id}'] += 1.0 / self.total_n_ops_for_curr_tw
                        # self.rewards[f'machine{current_machine.machine_id}'] += 1.0 / actual_processing_duration
                        self.rewards[f'machine{current_machine.machine_id}'] += 10.0 * (
                                    MAX_PRCS_TIME + MAX_TSPT_TIME - actual_processing_duration) / (
                                                                                            self.total_n_ops_for_curr_tw * (
                                                                                                MAX_PRCS_TIME + MAX_TSPT_TIME))
                        # self.rewards[f'machine{current_machine.machine_id}'] += (MAX_PRCS_TIME - actual_processing_duration) / MAX_PRCS_TIME
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
                # print(
                #     f"Machine {machine_index} finishes processing Job {current_job.job_id} at time {self.current_time_after_step}.")
                # self.rewards[f'machine{current_machine.machine_id}'] += 1.0 / self.total_n_ops_for_curr_tw
                # self.rewards[f'machine{current_machine.machine_id}'] += 0.1
                # self.rewards[f'machine{current_machine.machine_id}'] += 1.0 / actual_processing_duration
                self.rewards[f'machine{current_machine.machine_id}'] += 10.0 * (
                            MAX_PRCS_TIME + MAX_TSPT_TIME - actual_processing_duration) / (
                                                                                    self.total_n_ops_for_curr_tw * (
                                                                                        MAX_PRCS_TIME + MAX_TSPT_TIME))
                # self.rewards[f'machine{current_machine.machine_id}'] += (2 * MAX_PRCS_TIME - actual_processing_duration) / MAX_PRCS_TIME
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
                            if len(current_job.scheduled_results) > 0 and self.current_time_before_step < \
                                    current_job.scheduled_results[-1][3]:
                                current_transbot.idling_process()
                            else:
                                # Loaded transport can start immediately
                                loaded_path = a_star_search(
                                    graph=self.factory_instance.factory_graph,
                                    start=job_location,
                                    goal=machine_location
                                )
                                # if len(current_job.scheduled_results) > 0:
                                #     job_prev_ops_finish_time = current_job.scheduled_results[-1][3]
                                # else:
                                #     job_prev_ops_finish_time = self.local_result.time_window_start
                                # current_transbot.start_loaded_transporting(
                                #     target_location=machine_location,
                                #     loaded_path=loaded_path,
                                #     start_time=max(self.current_time_before_step, job_prev_ops_finish_time),
                                # )
                                current_transbot.start_loaded_transporting(
                                    target_location=machine_location,
                                    loaded_path=loaded_path,
                                    start_time=self.current_time_before_step,
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
                        if True:
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
                        # else:
                        #     current_transbot.idling_process()
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
            # raise ValueError(f"Transbot {current_transbot.agv_id}'s unload path is empty!")
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
                            # print(vars(current_job))
                            # print(current_job.scheduled_results)
                            # print(vars(current_transbot))
                            # print(current_transbot.scheduled_results)
                            # print(f"Location of job {current_transbot.current_task} is {job_location}.")
                            # print(f"Location of transbot {current_transbot.agv_id} is {current_transbot.current_location}.")
                            raise Exception(
                                f"Transbot {current_transbot.agv_id} hasn't get job {current_transbot.current_task}!")
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
                                start_time=self.current_time_after_step,
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

                    # job_prev_processing_finish_time = self.local_result.time_window_start
                    # if len(current_job.scheduled_results) > 2:
                    #     job_prev_processing_finish_time = current_job.scheduled_results[-3][-1]
                    # task_start_time = max(current_transbot.t_since_choose_job, job_prev_processing_finish_time)
                    # actual_duration_since_choose_job = self.current_time_after_step - task_start_time
                    # actual_duration_since_choose_job = self.current_time_after_step - current_transbot.t_since_choose_job
                    actual_duration = self.current_time_after_step - current_transbot.prev_loaded_finish_time
                    current_transbot.finish_loaded_transporting(finish_time=self.current_time_after_step)
                    # self.rewards[f'transbot{current_transbot.agv_id}'] += 1.0 / self.total_n_ops_for_curr_tw
                    # self.rewards[f'transbot{current_transbot.agv_id}'] += 0.1
                    self.rewards[f'transbot{current_transbot.agv_id}'] += 10.0 * (
                                2 * MAX_TSPT_TIME - actual_duration) / (self.total_n_ops_for_curr_tw * MAX_TSPT_TIME)
                    # self.rewards[f'transbot{current_transbot.agv_id}'] += (2 * MAX_TSPT_TIME - actual_duration_since_choose_job) / MAX_TSPT_TIME
                    # self.rewards[f'transbot{current_transbot.agv_id}'] += 1.0 / actual_duration_since_choose_job
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
            self.rewards[f'machine{machine_index}'] -= 1.0 / self.total_n_ops_for_curr_tw
            # self.rewards[f'machine{machine_index}'] -= 0.05
            # print(f"Job {processing_action}'s operation {current_operation} has already assigned to machine {current_job.assigned_machine}!")
            # raise Exception(
            #     f"Job {processing_action}'s operation {current_operation} has already assigned to machine {current_job.assigned_machine}!")
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

    def _check_transbot_transporting_action(self, transbot_index, transporting_action):
        current_transbot = self.factory_instance.agv[transbot_index]
        # Check what status is the transbot currently in, must be 0 (idling) or 1 (unload trans) to continue
        if current_transbot.agv_status not in (0, 1):
            raise Exception(
                f"current_transbot is not idling nor unload transporting, thus cannot make other decisions!")

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
            # raise Exception(f"Job {transporting_action} has already assigned to another transbot!")
            # print(f"Job {transporting_action} has already assigned to another transbot ({current_job.assigned_transbot})!")
            self.rewards[f'transbot{transbot_index}'] -= 1.0 / self.total_n_ops_for_curr_tw
            # self.rewards[f'transbot{transbot_index}'] -= 0.05
            if current_transbot.current_task is not None:
                if current_transbot.current_task == -1:
                    raise ValueError(f"Transbot {transbot_index} is for charging, cannot change decision!")
                else:
                    if current_transbot.agv_status == 1:
                        pass
                    else:
                        pass
                # # if current_transbot.current_task != transporting_action:
                #     old_job = self.scheduling_instance.jobs[current_transbot.current_task]
                #     old_job.assigned_transbot = None
                #     # current_transbot.scheduled_results.append(
                #     #     ("Unload Transporting", current_transbot.current_task,
                #     #      self.current_time_after_step))
                #     current_transbot.current_task = None
            # current_transbot.agv_status = 0
            # if current_transbot.agv_status == 1:
            #     current_transbot.finish_unload_transporting(finish_time=self.current_time_after_step)
            # go to nearest drop_pick location? or continue to do its previous task?
            # current_transbot.start_unload_transporting...

            return False
        else:
            return True

    def _check_transbot_finish_charging(self, transbot_id):
        current_transbot = self.factory_instance.agv[transbot_id]
        if current_transbot.charging_time <= 0:
            return True
        else:
            return False

    def _check_transbot_charging_action(self, transbot_index, charging_action):
        current_transbot = self.factory_instance.agv[transbot_index]
        # Check what status is the transbot currently in,
        # must be 0 (idling), 1 (unload trans) or 4 (low battery) to continue
        if current_transbot.agv_status not in (0, 1, 4):
            raise Exception(f"...!")

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

                        # if current_transbot.agv_status == 1:
                        #     new_job_location = self.factory_instance.factory_graph.pickup_dropoff_points[
                        #         new_job.current_location]
                        #
                        #     unload_path = a_star_search(
                        #         graph=self.factory_instance.factory_graph,
                        #         start=current_transbot.current_location,
                        #         goal=new_job_location
                        #     )
                        #     current_transbot.start_unload_transporting(
                        #         target_location=new_job_location,
                        #         unload_path=unload_path,
                        #         start_time=self.current_time_after_step
                        #     )
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

    def _check_done(self):
        if all(job.is_completed_for_current_time_window for job in self.scheduling_instance.jobs):
            # if self.remaining_operations == 0:
            return True
        else:
            return False

    def _check_truncated(self):
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

    def _save_instance_snapshot(self, final=False, policy_name="MADRL", episode_id=None, instance_snapshot_dir=None):
        current_instance_snapshot = {
            "factory_instance": self.factory_instance,
            "scheduling_instance": self.scheduling_instance,
            "start_t_for_curr_time_window": self.current_time_after_step,
        }
        self.snapshot = current_instance_snapshot

        # Save snapshot to file if enabled
        if instance_snapshot_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            instance_snapshot_dir = current_dir + \
                                    "/instance_snapshots" + \
                                    f"/M{dfjspt_params.n_machines}T{dfjspt_params.n_transbots}W{dfjspt_params.time_window_size}" \
                                    + f"/{policy_name}/snapshot_J{self.instance_n_jobs}I{self.current_instance_id}" \
                                    + f"_{self.current_window}"

        # Add episode ID to filename if provided
        if episode_id is not None:
            instance_snapshot_dir = instance_snapshot_dir + f"_ep{episode_id}"

        if final:
            instance_snapshot_dir = instance_snapshot_dir + "_final"

        os.makedirs(os.path.dirname(instance_snapshot_dir + ".pkl"), exist_ok=True)
        with open(instance_snapshot_dir + ".pkl", "wb") as snapshot_file:
            pickle.dump(current_instance_snapshot, snapshot_file)

    def _initialize_rendering(self):
        """ Initializes rendering settings dynamically based on factory layout. """
        # # Adjust figure size dynamically based on the factory graph size
        # self.scale_factor = 1.0  # Scale factor for dynamic figure sizing
        # self.figsize = (self.factory_instance.factory_graph.width * self.scale_factor,
        #                 self.factory_instance.factory_graph.height * self.scale_factor)
        #
        # self.machine_color_ids = set(
        #     f"Machine {machine.machine_id}"
        #     for machine in self.factory_instance.machines
        # )
        # if self.num_machines <= 20:
        #     self.machine_colormap = plt.colormaps["tab20"]
        # else:
        #     self.machine_colormap = cm.get_cmap("hsv", self.num_machines)
        # self.machine_color_map = {resource: self.machine_colormap(i / self.num_machines) for i, resource in
        #                            enumerate(self.machine_color_ids)}
        #
        # self.transbot_color_ids = set(
        #     f"Transbot {transbot.agv_id}"
        #     for transbot in self.factory_instance.agv
        # )
        # if self.num_transbots <= 12:
        #     self.transbot_colormap = plt.colormaps["Set3"]
        # else:
        #     self.transbot_colormap = cm.get_cmap("Purples", self.num_transbots)
        # self.transbot_color_map = {resource: self.transbot_colormap(i / self.num_transbots) for i, resource in
        #                            enumerate(self.transbot_color_ids)}
        # factory_graph = self.factory_instance.factory_graph
        # self.figsize = (factory_graph.width, factory_graph.height)
        self.figsize = (12, 6)

        # Machine color mapping
        machine_ids = [f"Machine {m.machine_id}" for m in self.factory_instance.machines]
        colormap = plt.colormaps["tab20"] if self.num_machines <= 20 else cm.get_cmap("hsv", self.num_machines)
        self.machine_color_map = {m_id: colormap(i / self.num_machines) for i, m_id in enumerate(machine_ids)}

        # Transbot color mapping
        transbot_ids = [f"Transbot {t.agv_id}" for t in self.factory_instance.agv]
        colormap = plt.colormaps["Set3"] if self.num_transbots <= 12 else cm.get_cmap("Purples", self.num_transbots)
        self.transbot_color_map = {t_id: colormap(i / self.num_transbots) for i, t_id in enumerate(transbot_ids)}

    def render(self):
        """
        Render the factory layout for the current time step.
        This method visualizes the current state of the factory, including machines, transbots, and jobs.
        """

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
                # self.plot_size = 100 * (self.factory_instance.factory_graph.width + self.factory_instance.factory_graph.height) / 2
                # self.plot_size = 25 * np.pi
                self.ax.set_xlabel("X Position")
                self.ax.set_ylabel("Y Position")
                x_ticks = range(-1, self.factory_instance.factory_graph.width + 1, 1)
                y_ticks = range(-1, self.factory_instance.factory_graph.height + 1, 1)
                self.ax.set_xticks(x_ticks)
                self.ax.set_yticks(y_ticks)
                self.ax.set_aspect('equal', adjustable='box')
                # self.ax.grid(True, linestyle='--', linewidth=0.5)
                self.fig.subplots_adjust(right=0.6)

            # Update dynamic elements: transbots and jobs
            self._update_dynamic_elements()

            if self.render_mode == "human":
                # Render the dynamic machine data plot
                plt.show()

                # Redraw and pause for real-time display
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                plt.pause(1 / self.metadata["render_fps"])

            elif self.render_mode == "rgb_array":
                raise NotImplementedError(f"Render mode '{self.render_mode}' is not supported.")

    def close(self):
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

    def plot_transbot_trajectory(self, transbot_id: int, start_time: float, end_time: float,
                                 save_path: str = None, show_plot: bool = True, dpi: int = 300):
        """
        Plot professional 2D and 3D trajectory visualization for a single transbot.

        Args:
            transbot_id: ID of the transbot to plot
            start_time: Start time for trajectory filtering
            end_time: End time for trajectory filtering
            save_path: Optional path to save the plot (without extension)
            show_plot: Whether to display the plot interactively
            dpi: Resolution for saved plots

        Returns:
            tuple: (fig, (ax2d, ax3d)) matplotlib figure and axes objects
        """
        # Input validation
        if transbot_id < 0 or transbot_id >= len(self.factory_instance.agv):
            raise ValueError(
                f"Invalid transbot_id {transbot_id}. Must be in range [0, {len(self.factory_instance.agv) - 1}]")

        if start_time >= end_time:
            raise ValueError(f"start_time ({start_time}) must be less than end_time ({end_time})")

        # Filter trajectory data for the specified time range
        trajectory_hist = self.factory_instance.agv[transbot_id].trajectory_hist
        filtered_results = {t: loc for t, loc in trajectory_hist.items() if start_time <= t <= end_time}

        if not filtered_results:
            print(f"No trajectory data available for transbot {transbot_id} in time range [{start_time}, {end_time}]")
            return None, (None, None)

        # Sort and extract coordinates
        time_steps = sorted(filtered_results.keys())
        locations = [filtered_results[t] for t in time_steps]
        x_coords = [loc[0] for loc in locations]
        y_coords = [loc[1] for loc in locations]
        z_coords = time_steps

        # Create professional figure with optimal layout
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle(f'Transbot {transbot_id} Trajectory Analysis (Time: {start_time:.1f} - {end_time:.1f})',
                     fontsize=16, fontweight='bold', y=0.95)

        # Professional color scheme
        trajectory_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        trajectory_color = trajectory_colors[transbot_id % len(trajectory_colors)]

        # 2D visualization (left subplot)
        ax2d = fig.add_subplot(121)
        ax2d.set_aspect('equal', adjustable='box')

        # Plot factory layout (obstacles and important points)
        self._plot_factory_layout_2d(ax2d)

        # Plot trajectory with gradient coloring
        if len(time_steps) > 1:
            norm = plt.Normalize(vmin=min(time_steps), vmax=max(time_steps))
            cmap = plt.cm.get_cmap('viridis')

            # Plot trajectory segments with color gradient
            for i in range(len(time_steps) - 1):
                alpha = 0.3 + 0.7 * (i / max(1, len(time_steps) - 1))  # Increasing opacity
                ax2d.plot([x_coords[i], x_coords[i + 1]], [y_coords[i], y_coords[i + 1]],
                          color=cmap(norm(time_steps[i])), linewidth=3, alpha=alpha, zorder=3)

                # Add time-based markers
                if i % max(1, len(time_steps) // 10) == 0:  # Show every ~10th point
                    ax2d.scatter(x_coords[i], y_coords[i], c=cmap(norm(time_steps[i])),
                                 s=60, alpha=0.8, edgecolors='white', linewidth=1, zorder=4)

        # Highlight start and end points
        ax2d.scatter(x_coords[0], y_coords[0], color='lime', s=120, marker='o',
                     label='Start', zorder=5, edgecolors='darkgreen', linewidth=2)
        ax2d.scatter(x_coords[-1], y_coords[-1], color='red', s=120, marker='s',
                     label='End', zorder=5, edgecolors='darkred', linewidth=2)

        # Configure 2D plot
        ax2d.set_title('2D Trajectory View', fontsize=14, fontweight='bold', pad=20)
        ax2d.set_xlabel('X Coordinate', fontsize=12)
        ax2d.set_ylabel('Y Coordinate', fontsize=12)
        ax2d.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax2d.legend(loc='upper right', framealpha=0.9, fontsize=10)

        # Set proper limits with padding
        padding = 0.5
        ax2d.set_xlim(min(x_coords) - padding, max(x_coords) + padding)
        ax2d.set_ylim(min(y_coords) - padding, max(y_coords) + padding)

        # 3D visualization (right subplot)
        ax3d = fig.add_subplot(122, projection='3d')

        # Plot 3D trajectory
        if len(time_steps) > 1:
            # Main trajectory line
            ax3d.plot(x_coords, y_coords, z_coords, color=trajectory_color,
                      linewidth=2.5, alpha=0.8, label=f'Transbot {transbot_id}')

            # Add trajectory points with gradient
            colors = cmap(norm(z_coords))
            ax3d.scatter(x_coords, y_coords, z_coords, c=colors, s=30,
                         alpha=0.7, edgecolors='white', linewidth=0.5)

        # Highlight start and end points in 3D
        ax3d.scatter([x_coords[0]], [y_coords[0]], [z_coords[0]],
                     color='lime', s=100, marker='o', label='Start',
                     edgecolors='darkgreen', linewidth=2)
        ax3d.scatter([x_coords[-1]], [y_coords[-1]], [z_coords[-1]],
                     color='red', s=100, marker='s', label='End',
                     edgecolors='darkred', linewidth=2)

        # Configure 3D plot
        ax3d.set_title('3D Trajectory View', fontsize=14, fontweight='bold', pad=20)
        ax3d.set_xlabel('X Coordinate', fontsize=11)
        ax3d.set_ylabel('Y Coordinate', fontsize=11)
        ax3d.set_zlabel('Time Step', fontsize=11)
        ax3d.legend(loc='upper right', fontsize=10)

        # Set 3D view angle for better visualization
        ax3d.view_init(elev=20, azim=45)

        # Add statistics text
        total_distance = sum(abs(x_coords[i + 1] - x_coords[i]) + abs(y_coords[i + 1] - y_coords[i])
                             for i in range(len(x_coords) - 1))
        duration = end_time - start_time

        stats_text = f'Duration: {duration:.1f} time units\n' \
                     f'Total Distance: {total_distance:.1f} units\n' \
                     f'Avg Speed: {total_distance / max(duration, 1):.2f} units/time'

        fig.text(0.02, 0.02, stats_text, fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(f"{save_path}_trajectory.png", dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"Trajectory plot saved to: {save_path}_trajectory.png")

        # Show if requested
        if show_plot:
            plt.show()

        return fig, (ax2d, ax3d)

    def plot_multiple_transbot_trajectories(self, transbot_id_list: list, start_time: float, end_time: float,
                                            save_path: str = None, show_plot: bool = True, dpi: int = 300):
        """
        Plot professional 2D and 3D trajectory visualization for multiple transbots.

        Args:
            transbot_id_list: List of transbot IDs to plot (max 8 for readability)
            start_time: Start time for trajectory filtering
            end_time: End time for trajectory filtering
            save_path: Optional path to save the plot (without extension)
            show_plot: Whether to display the plot interactively
            dpi: Resolution for saved plots

        Returns:
            tuple: (fig, (ax2d, ax3d)) matplotlib figure and axes objects
        """
        # Input validation
        max_transbots = 8
        if len(transbot_id_list) > max_transbots:
            raise ValueError(
                f"Too many transbots ({len(transbot_id_list)}). Maximum is {max_transbots} for readability.")

        if not transbot_id_list:
            raise ValueError("transbot_id_list cannot be empty")

        if start_time >= end_time:
            raise ValueError(f"start_time ({start_time}) must be less than end_time ({end_time})")

        # Professional color palette
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        # Create professional figure
        fig = plt.figure(figsize=(18, 9))
        fig.suptitle(f'Multiple Transbot Trajectory Analysis (Time: {start_time:.1f} - {end_time:.1f})',
                     fontsize=18, fontweight='bold', y=0.95)

        # 2D visualization (left subplot)
        ax2d = fig.add_subplot(121)
        ax2d.set_aspect('equal', adjustable='box')

        # Plot factory layout
        self._plot_factory_layout_2d(ax2d)

        # 3D visualization (right subplot)
        ax3d = fig.add_subplot(122, projection='3d')

        # Track bounds for proper scaling
        all_x, all_y, all_t = [], [], []
        valid_transbots = []

        # Plot each transbot's trajectory
        for i, transbot_id in enumerate(transbot_id_list):
            if transbot_id < 0 or transbot_id >= len(self.factory_instance.agv):
                print(f"Warning: Invalid transbot_id {transbot_id}, skipping...")
                continue

            # Filter trajectory data
            trajectory_hist = self.factory_instance.agv[transbot_id].trajectory_hist
            filtered_results = {t: loc for t, loc in trajectory_hist.items() if start_time <= t <= end_time}

            if not filtered_results:
                print(f"Warning: No data for transbot {transbot_id} in time range [{start_time}, {end_time}]")
                continue

            # Extract coordinates
            time_steps = sorted(filtered_results.keys())
            locations = [filtered_results[t] for t in time_steps]
            x_coords = [loc[0] for loc in locations]
            y_coords = [loc[1] for loc in locations]
            z_coords = time_steps

            all_x.extend(x_coords)
            all_y.extend(y_coords)
            all_t.extend(z_coords)
            valid_transbots.append(transbot_id)

            # Select color
            color = colors[i % len(colors)]

            # 2D plot
            if len(time_steps) > 1:
                # Plot trajectory with alpha gradient
                for j in range(len(time_steps) - 1):
                    alpha = 0.3 + 0.7 * (j / max(1, len(time_steps) - 1))
                    ax2d.plot([x_coords[j], x_coords[j + 1]], [y_coords[j], y_coords[j + 1]],
                              color=color, linewidth=2.5, alpha=alpha, zorder=3)

                # Plot trajectory points
                ax2d.plot(x_coords, y_coords, 'o', color=color, markersize=4,
                          alpha=0.6, label=f'Transbot {transbot_id}', zorder=4)

            # Highlight start and end points in 2D
            ax2d.scatter(x_coords[0], y_coords[0], color=color, s=100, marker='o',
                         zorder=5, edgecolors='white', linewidth=2)
            ax2d.scatter(x_coords[-1], y_coords[-1], color=color, s=100, marker='s',
                         zorder=5, edgecolors='white', linewidth=2)

            # 3D plot
            if len(time_steps) > 1:
                ax3d.plot(x_coords, y_coords, z_coords, color=color, linewidth=2.5,
                          alpha=0.8, label=f'Transbot {transbot_id}')
                ax3d.scatter(x_coords, y_coords, z_coords, c=color, s=25, alpha=0.6)

            # Highlight start and end points in 3D
            ax3d.scatter([x_coords[0]], [y_coords[0]], [z_coords[0]],
                         color=color, s=80, marker='o', edgecolors='white', linewidth=1.5)
            ax3d.scatter([x_coords[-1]], [y_coords[-1]], [z_coords[-1]],
                         color=color, s=80, marker='s', edgecolors='white', linewidth=1.5)

        if not valid_transbots:
            print("No valid trajectory data found for any transbot in the specified time range.")
            plt.close(fig)
            return None, (None, None)

        # Configure 2D plot
        ax2d.set_title('2D Multi-Transbot Trajectories', fontsize=14, fontweight='bold', pad=20)
        ax2d.set_xlabel('X Coordinate', fontsize=12)
        ax2d.set_ylabel('Y Coordinate', fontsize=12)
        ax2d.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax2d.legend(loc='upper right', framealpha=0.9, fontsize=10, ncol=2 if len(valid_transbots) > 4 else 1)

        # Set proper limits with padding
        if all_x and all_y:
            padding = 0.5
            ax2d.set_xlim(min(all_x) - padding, max(all_x) + padding)
            ax2d.set_ylim(min(all_y) - padding, max(all_y) + padding)

        # Configure 3D plot
        ax3d.set_title('3D Multi-Transbot Trajectories', fontsize=14, fontweight='bold', pad=20)
        ax3d.set_xlabel('X Coordinate', fontsize=11)
        ax3d.set_ylabel('Y Coordinate', fontsize=11)
        ax3d.set_zlabel('Time Step', fontsize=11)
        ax3d.legend(loc='upper right', fontsize=9, ncol=2 if len(valid_transbots) > 4 else 1)
        ax3d.view_init(elev=20, azim=45)

        # Add summary statistics
        duration = end_time - start_time
        stats_text = f'Transbots analyzed: {len(valid_transbots)}\n' \
                     f'Time duration: {duration:.1f} units\n' \
                     f'Total time range: [{min(all_t):.1f}, {max(all_t):.1f}]'

        fig.text(0.02, 0.02, stats_text, fontsize=11,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(f"{save_path}_multi_trajectories.png", dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"Multi-trajectory plot saved to: {save_path}_multi_trajectories.png")

        # Show if requested
        if show_plot:
            plt.show()

        return fig, (ax2d, ax3d)

    def _plot_factory_layout_2d(self, ax):
        """
        Helper method to plot factory layout elements in 2D plots.

        Args:
            ax: matplotlib axes object
        """
        # Plot obstacles
        for x, y in self.factory_instance.factory_graph.obstacles:
            ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1,
                                       color='#2F2F2F', alpha=0.8, zorder=1))

        # Plot pickup/dropoff points
        pickup_dropoff_points = self.factory_instance.factory_graph.pickup_dropoff_points
        for point_name, location in pickup_dropoff_points.items():
            x, y = location
            if "machine" in point_name:
                ax.scatter(x, y, c='steelblue', s=150, marker='s', alpha=0.7,
                           edgecolors='darkblue', linewidth=1.5, zorder=2,
                           label='Machine' if point_name == list(pickup_dropoff_points.keys())[0] else "")
            elif "charging" in point_name:
                ax.scatter(x, y, c='gold', s=120, marker='^', alpha=0.7,
                           edgecolors='orange', linewidth=1.5, zorder=2,
                           label='Charging' if 'Charging' not in [t.get_text() for t in ax.texts] else "")
            elif point_name == "warehouse":
                ax.scatter(x, y, c='brown', s=180, marker='D', alpha=0.7,
                           edgecolors='darkred', linewidth=1.5, zorder=2, label='Warehouse')

        # Remove duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = []
        unique_handles = []
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels.append(label)
                unique_handles.append(handle)

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

                # Rectangle height
                rect_height = 0.3

                # Plot a rectangle for the operation (centered on the job line)
                ax.add_patch(
                    Rectangle(
                        (start_time, i - rect_height / 2),  # Center rectangle on job line
                        end_time - start_time,  # width (duration)
                        rect_height,  # height (job row height)
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
                    # Rectangle height for maintenance (varying by type)
                    rect_height = 0.3 + 0.1 * (job_id + 1)

                    # Plot a rectangle for the maintenance operation (centered on the machine line)
                    ax.add_patch(
                        Rectangle(
                            (start_time, i - rect_height / 2),  # Center rectangle on machine line
                            end_time - start_time,  # width (duration)
                            rect_height,  # height varies by maintenance type
                            color="grey"
                        )
                    )
                else:
                    # Determine color based on resource
                    color = job_color_map[job_id]

                    # Rectangle height for processing
                    rect_height = 0.3

                    # Plot a rectangle for the processing operation (centered on the machine line)
                    ax.add_patch(
                        Rectangle(
                            (start_time, i - rect_height / 2),  # Center rectangle on machine line
                            end_time - start_time,  # width (duration)
                            rect_height,  # height (job row height)
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
                        # Rectangle height for charging-related unload transport
                        # Use slightly smaller height to compensate for border visual effect
                        rect_height = 0.28  # Reduced from 0.3 to compensate for linewidth=1

                        ax.add_patch(
                            Rectangle(
                                (start_time, i - rect_height / 2),  # Center rectangle on transbot line
                                end_time - start_time,  # width (duration)
                                rect_height,  # height (transbot row height)
                                facecolor="white",
                                edgecolor="grey",
                                linewidth=1
                            )
                        )
                    else:
                        # Determine color based on job
                        color = job_color_map[job_id]

                        # Rectangle height for job-related unload transport
                        # Use slightly smaller height to compensate for border visual effect
                        rect_height = 0.28  # Reduced from 0.3 to compensate for linewidth=1

                        # Plot a rectangle for the operation (centered on the transbot line)
                        ax.add_patch(
                            Rectangle(
                                (start_time, i - rect_height / 2),  # Center rectangle on transbot line
                                end_time - start_time,  # width (duration)
                                rect_height,  # height (transbot row height)
                                facecolor="white",
                                edgecolor=color,
                                linewidth=1
                            )
                        )
                elif op_type == "Loaded Transporting":
                    # Determine color based on job
                    color = job_color_map[job_id]

                    # Rectangle height for loaded transport
                    # Keep standard height since we're using facecolor/edgecolor now
                    rect_height = 0.3

                    # Plot a rectangle for the operation (centered on the transbot line)
                    ax.add_patch(
                        Rectangle(
                            (start_time, i - rect_height / 2),  # Center rectangle on transbot line
                            end_time - start_time,  # width (duration)
                            rect_height,  # height (transbot row height)
                            facecolor=color,
                            edgecolor="white",
                            linewidth=1
                        )
                    )
                elif op_type == "Charging":
                    # Rectangle height for charging
                    # Use slightly smaller height to compensate for border visual effect
                    rect_height = 0.38  # Reduced from 0.4 to compensate for linewidth=1

                    ax.add_patch(
                        Rectangle(
                            (start_time, i - rect_height / 2),  # Center rectangle on transbot line
                            end_time - start_time,  # width (duration)
                            rect_height,  # height (transbot row height)
                            facecolor="yellow",
                            edgecolor="orange",
                            linewidth=1
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


def run_one_local_instance(
        local_instance_file,
        current_window=None,
        num_episodes=10,
        do_render=False,
        # do_render=True,
        do_plot_gantt=True,
        # do_plot_gantt=False,
        do_plot_trajectories=False,
        detailed_log=True,
):
    print(f"Starting instance {local_instance_file}:\n")

    config = {
        "n_machines": dfjspt_params.n_machines,
        "n_transbots": dfjspt_params.n_transbots,
        "factory_instance_seed": dfjspt_params.factory_instance_seed,
        "enable_dynamic_agent_filtering": getattr(dfjspt_params, 'enable_dynamic_agent_filtering', False),
        "render_mode": "human",
    }
    scheduling_env = LocalSchedulingMultiAgentEnv(config)

    makespans = []

    for episode_id in range(num_episodes):
        makespans = run_one_episode(episode_id=episode_id + 1,
                                    scheduling_env=scheduling_env,
                                    current_window=current_window,
                                    # local_schedule=local_schedule,
                                    makespans=makespans,
                                    do_render=do_render,
                                    do_plot_gantt=do_plot_gantt,
                                    do_plot_trajectories=do_plot_trajectories,
                                    local_instance_file=local_instance_file,
                                    detailed_log=detailed_log,
                                    )

    print(f"\nMin makespan across {num_episodes} episodes is {np.min(makespans)}.")
    print(f"Average makespan across {num_episodes} episodes is {np.average(makespans)}.")


def run_one_episode(
        episode_id, scheduling_env,
        makespans,
        current_window=None,
        local_schedule=None,
        do_render=False,
        # do_render=True,
        do_plot_gantt=True,
        # do_plot_gantt=False,
        do_plot_trajectories=False,
        local_instance_file=None,
        detailed_log=True,
):
    from local_realtime_scheduling.Agents.generate_training_data import generate_reset_options_for_training
    print(f"\nStarting episode {episode_id}:")
    decision_count = 0

    env_reset_options = generate_reset_options_for_training(
        local_schedule_filename=local_instance_file,
        for_training=False,
        # for_training=True,
    )

    observations, infos = scheduling_env.reset(options=env_reset_options)
    if do_render:
        scheduling_env.render()
    # print(f"decision_count = {decision_count}")
    decision_count += 1
    # print(f"remaining operations = {scheduling_env.remaining_operations}")
    done = {'__all__': False}
    truncated = {'__all__': False}
    total_rewards = {}
    for agent in scheduling_env.agents:
        total_rewards[agent] = 0.0

    while (not done['__all__']) and (not truncated['__all__']):
        actions = {}
        for agent_id, obs in observations.items():
            # print(f"current agent = {agent_id}")
            action_mask = obs['action_mask']
            valid_actions = [i for i, valid in enumerate(action_mask) if valid == 1]
            if valid_actions:
                if len(valid_actions) > 1:
                    valid_actions.pop(-1)
                actions[agent_id] = np.random.choice(valid_actions)
            else:
                raise Exception(f"No valid actions for agent {agent_id}!")
                # actions[agent_id] = 0  # Default to a no-op if no valid actions

        observations, rewards, done, truncated, info = scheduling_env.step(actions)
        if do_render:
            scheduling_env.render()
        # if scheduling_env.all_agents_have_made_decisions:
        #     print(f"Current timestep is {scheduling_env.current_time_after_step}.")
        #     print(f"Remaining operations is {scheduling_env.remaining_operations}.")
        # print(f"decision_count = {decision_count}")
        # print(f"remaining operations = {scheduling_env.remaining_operations}")
        decision_count += 1

        for agent, reward in rewards.items():
            total_rewards[agent] += reward

    if do_render:
        scheduling_env.close()
    # print(f"Actions: {actions}")
    # print(f"Rewards: {rewards}")
    # print(f"Done: {done}")
    # for job_id in range(scheduling_env.num_jobs):
    #     print(f"job {job_id}: {scheduling_env.scheduling_instance.jobs[job_id].scheduled_results}")
    # for machine_id in range(scheduling_env.num_machines):
    #     print(f"machine {machine_id}: {scheduling_env.factory_instance.machines[machine_id].scheduled_results}")
    # for transbot_id in range(scheduling_env.num_transbots):
    #     print(f"transbot {transbot_id}: {scheduling_env.factory_instance.agv[transbot_id].scheduled_results}")
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
    if do_plot_trajectories:
        scheduling_env.plot_transbot_trajectory(
            0,
            scheduling_env.local_result.time_window_start,
            scheduling_env.current_time_after_step,
        )
        scheduling_env.plot_transbot_trajectory(
            1,
            scheduling_env.local_result.time_window_start,
            scheduling_env.current_time_after_step,
        )

    if detailed_log:
        print(f"Current timestep is {scheduling_env.current_time_after_step}.")
        if scheduling_env.local_result.actual_local_makespan is not None:
            print(f"Actual makespan = {scheduling_env.local_result.actual_local_makespan}")
            print(
                f"Actual delta makespan = {scheduling_env.local_result.actual_local_makespan - scheduling_env.local_result.time_window_start}")
            makespans.append(
                scheduling_env.local_result.actual_local_makespan - scheduling_env.local_result.time_window_start)
        else:
            makespans.append(scheduling_env.current_time_after_step - scheduling_env.local_result.time_window_start)
            print(f"Truncated makespan = {scheduling_env.current_time_after_step}")
            print(
                f"Truncated delta makespan = {scheduling_env.current_time_after_step - scheduling_env.local_result.time_window_start}")
        print(f"Estimated makespan = {scheduling_env.initial_estimated_makespan}")
        print(
            f"Estimated delta makespan = {scheduling_env.initial_estimated_makespan - scheduling_env.local_result.time_window_start}")
        print(f"Total reward for episode {episode_id}: {total_rewards['machine0']}")

        func("Local Scheduling completed.")

        print(f"Min makespan up to now is {np.min(makespans)}.")
        print(f"Average makespan up to now is {np.average(makespans)}.")
    else:
        if scheduling_env.local_result.actual_local_makespan is not None:
            makespans.append(
                scheduling_env.local_result.actual_local_makespan - scheduling_env.local_result.time_window_start)
        else:
            makespans.append(scheduling_env.current_time_after_step - scheduling_env.local_result.time_window_start)

    return makespans


# Example usage
if __name__ == "__main__":
    import time

    func("main function begin.")

    instance_n_jobs = 10
    scheduling_instance_id = 100
    current_window = 0
    num_ops = 48

    num_instances = 1
    for local_instance in range(num_instances):
        file_name = f"local_schedule_J{instance_n_jobs}I{scheduling_instance_id}_{current_window}_ops{num_ops}.pkl"
        start_time = time.time()
        run_one_local_instance(
            local_instance_file=file_name,
            current_window=current_window,
            do_plot_gantt=True,
            # do_plot_gantt=False,
            num_episodes=1
        )
        print(f"Total running time is {time.time() - start_time}")

