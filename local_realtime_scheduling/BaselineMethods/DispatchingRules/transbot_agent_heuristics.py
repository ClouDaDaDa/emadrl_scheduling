import numpy as np
import copy
from enum import Enum


def transbot_agent_heuristics(transbot_obs):
    """
    :param transbot_obs = {
                "action_mask": transbot_action_mask,
                "observation": {
                    "job_features": job_features,
                    "transbot_features": transbot_features,
                }
            }
    :return: transbot_action
    """

    action_mask = copy.deepcopy(transbot_obs["action_mask"])
    num_jobs = len(action_mask) - 2


    invalid_action_penalties = (1 - action_mask) * 1e8
    if transbot_obs["observation"]["transbot_features"][0] != 4:
        invalid_action_penalties[num_jobs] = 1e4
    #     print(f"transbot need not go charging.")
    # else:
    #     print(f"transbot should go charging.")
    if transbot_obs["observation"]["transbot_features"][1] >= 0:
        invalid_action_penalties[:num_jobs] = 1e4
    else:
        # Cannot do nothing when idling
        if transbot_obs["observation"]["transbot_features"][0] == 0:
            invalid_action_penalties[num_jobs + 1] = 1e3
    # Choose the job with the shortest transporting time (STT)
    transporting_time = np.zeros((len(action_mask),))
    transporting_time[:num_jobs] = transbot_obs["observation"]["job_features"][:, 6]
    action_score = transporting_time + invalid_action_penalties
    transbot_action = np.random.choice(np.where(action_score == action_score.min())[0])

    return transbot_action



class TransbotJobRule(Enum):
    NEAREST = "nearest_first"
    EET = "earliest_end_time"
    MONR = "most_operations_number_remaining"
    SPRO = "slack_per_remaining_ops"

    # \text{Slack}_j = d_j - t - p_j
    LSF = "least_slack_first"

class TransbotChargeRule(Enum):
    THRESHOLD = "threshold"
    NEVER = "never"


def transbot_heuristic(
    obs: dict,
    job_rule: TransbotJobRule,
    charge_rule: TransbotChargeRule,
    threshold: float = 0.5,
) -> int:
    """
    Generic transbot heuristic.
    :param obs: {
        "action_mask": np.ndarray of 0/1 shape (n_actions,),
        "observation": {
            "job_features": np.ndarray shape (n_jobs, n_features),
            "transbot_features": np.ndarray shape (n_features,)
        }
    }
    :param job_rule: NEAREST, EET, MONR, SLK
    :param charge_rule: THRESHOLD, NEVER
    :param threshold: battery SOC threshold for charging
    """
    mask = obs["action_mask"].astype(int)
    tf = obs["observation"]["transbot_features"]
    jf = obs["observation"]["job_features"]
    n_actions = len(mask)
    n_jobs = jf.shape[0]
    penalties = (1 - mask) * 1e8

    status = int(tf[0])    # 0 idle,1 unload,2 loaded,3 charging,4 low_batt
    soc = tf[4]

    # charging rule
    if charge_rule is TransbotChargeRule.NEVER:
        penalties[n_jobs] = 1e4  # Charging is prohibited
    elif charge_rule is TransbotChargeRule.THRESHOLD:
        if soc > threshold:
            penalties[n_jobs] = 1e4

    # if carrying a load (tf[1]>=0), cannot pick new job
    if tf[1] >= 0:
        penalties[:n_jobs] = 1e4

    # cannot do-nothing when idle
    if status == 0:
        penalties[n_jobs + 1] = 1e3
    # charging when low battery
    elif status == 4:
        penalties[n_jobs] = 0

    # job selection scoring
    scores = np.zeros(n_actions, dtype=float) + penalties

    if job_rule is TransbotJobRule.NEAREST:
        # nearest = minimal transport time
        tt = jf[:, 6]
        scores[:n_jobs] += tt

    elif job_rule is TransbotJobRule.EET:
        eet = jf[:, 3]
        scores[:n_jobs] += eet

    elif job_rule is TransbotJobRule.MONR:
        rem_ops = jf[:, 7]
        scores[:n_jobs] += -rem_ops

    elif job_rule is TransbotJobRule.SPRO:
        slack = tf[9]
        rem_ops = jf[:, 7]
        scores[:n_jobs] += slack / (rem_ops + 1e-6)

    elif job_rule is TransbotJobRule.LSF:
        slack = tf[9]
        tt = jf[:, 6]
        scores[:n_jobs] += (slack - tt)

    best = np.where(scores == scores.min())[0]
    return int(np.random.choice(best))