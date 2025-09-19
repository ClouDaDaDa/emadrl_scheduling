import copy
import numpy as np
from enum import Enum


def machine_agent_heuristics(machine_obs):
    """
    :param machine_obs = {
                "action_mask": machine_action_mask,
                "observation": {
                    "job_features": job_features,
                    "time_to_makespan": self.current_time_after_step - self.initial_estimated_makespan,
                    "machine_features": machine_features,
                }
            }
    :return: machine_action
    """

    action_mask = copy.deepcopy(machine_obs["action_mask"])
    num_jobs = len(action_mask) - 5

    invalid_action_penalties = (1 - action_mask) * 1e8
    if machine_obs["observation"]["machine_features"][0] != 3:
        invalid_action_penalties[num_jobs:num_jobs+3] = 1e4
    # Cannot do nothing when idling
    if machine_obs["observation"]["machine_features"][0] == 0:
        invalid_action_penalties[num_jobs + 4] = 1e3
    # Choose the job with the shortest processing time (SPT)
    processing_time = np.zeros((len(action_mask),))
    processing_time[:num_jobs] = machine_obs["observation"]["job_features"][:, 4]
    processing_time[:num_jobs] += machine_obs["observation"]["job_features"][:, 5]
    action_score = processing_time + invalid_action_penalties
    machine_action = np.random.choice(np.where(action_score == action_score.min())[0])

    if num_jobs <= machine_action < num_jobs + 4:
        if machine_obs["observation"]["machine_features"][0] in [1, 2]:
            raise ValueError(f"Invalid action!")

    return machine_action






class MachineJobRule(Enum):
    # P_{i,j,k} + Q_{i,k}, where Q_{i,k} is the unload transport time from J_i to M_k
    SPT = "shortest_processing_time"
    # LPT = "longest_processing_time"

    # C_{i,j-1,k'}, which is the actual complete time of O_{i,j-1}
    EET = "earliest_end_time"

    MONR = "most_operations_number_remaining"

    SPRO = "slack_per_remaining_ops"

    # \text{Slack}_j = d_j - t - p_j
    LSF = "least_slack_first"

class MachineMaintRule(Enum):
    PERIODIC = "periodic"
    THRESHOLD = "threshold_driven"
    NEVER = "never"


def machine_heuristic(
    obs: dict,
    job_rule: MachineJobRule,
    maint_rule: MachineMaintRule,
    due_date: float,
    periodic_interval: float = None,
    threshold: float = 0.5,
) -> int:
    """
    Generic machine heuristic.
    :param obs: {
        "action_mask": np.ndarray of 0/1 shape (n_actions,),
        "observation": {
            "job_features": np.ndarray shape (n_jobs, n_features),
            "machine_features": np.ndarray shape (n_machine_features,)
        }
    }
    :param job_rule: SPT, EET, MONR, or SPRO
    :param maint_rule: PERIODIC, THRESHOLD, NEVER
    :param due_date: implicit deadline for all jobs (e.g. initial makespan)
    :param periodic_interval: ΔT for PERIODIC (required if PERIODIC)
    :param threshold: reliability threshold for THRESHOLD

    :return: chosen action index
    """
    mask = obs["action_mask"].astype(int)
    mf = obs["observation"]["machine_features"]
    jf = obs["observation"]["job_features"]
    n_actions = len(mask)
    n_jobs = jf.shape[0]

    # 1) maintenance masking / penalties
    penalties = (1 - mask) * 1e8

    status = int(mf[0])  # 0 idle,1 processing,2 maint,3 failed
    reliability = mf[1]
    current_time = due_date - mf[6]  # initial_makespan - (initial_makespan - current_time)
    # total_rem_ops = mf[7]

    # NEVER: prohibit indices [n_jobs,...,n_jobs+2]
    if maint_rule is MachineMaintRule.NEVER:
        penalties[n_jobs : n_jobs + 3] = 1e4

    # THRESHOLD-DRIVEN: allow only if reliability < threshold
    elif maint_rule is MachineMaintRule.THRESHOLD:
        if reliability >= threshold:
            penalties[n_jobs : n_jobs + 3] = 1e4

    # PERIODIC: allow only when elapsed_time % ΔT == 0
    elif maint_rule is MachineMaintRule.PERIODIC:
        if periodic_interval is None:
            raise ValueError("periodic_interval must be set for PERIODIC maintenance")
        if current_time % periodic_interval != 0:
            penalties[n_jobs: n_jobs + 3] = 1e4
        else:
            penalties[:n_jobs] = 1e4

    # CM (corrective) only when failed:
    if status != 3:
        penalties[n_jobs + 3] = 1e4

    # Cannot do-nothing if idle
    if status == 0:
        penalties[n_jobs + 4] = 1e3

    # 2) job selection scoring
    scores = np.zeros(n_actions, dtype=float) + penalties

    est_rem_time = jf[:, 3]
    p_time = jf[:, 4]
    unload_time = jf[:, 5]
    j_rem_ops = jf[:, 6]

    if job_rule is MachineJobRule.SPT:
        # processing time + unload distance
        pt = p_time + unload_time
        scores[:n_jobs] += pt

    # elif job_rule is MachineJobRule.LPT:
    #     pt = p_time + unload_time
    #     scores[:n_jobs] += -pt

    elif job_rule is MachineJobRule.EET:
        # earliest end time = - estimated remaining finish
        eet = est_rem_time
        # todo: eet = est_rem_time - waiting_time
        scores[:n_jobs] += eet

    elif job_rule is MachineJobRule.MONR:
        # most operations number remaining
        scores[:n_jobs] += -j_rem_ops  # maximize rem_ops → minimize -rem_ops

    elif job_rule is MachineJobRule.SPRO:
        # slack per remaining op = (global_slack) / rem_ops
        scores[:n_jobs] += (due_date - current_time) / (j_rem_ops + 1e-6)

    elif job_rule is MachineJobRule.LSF:
        scores[:n_jobs] += (due_date - current_time - p_time - unload_time)

    # pick minimal score action
    best = np.where(scores == scores.min())[0]
    return int(np.random.choice(best))

