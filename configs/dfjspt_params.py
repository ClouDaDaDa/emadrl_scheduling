import math

# local scheduling params
current_window = 0
time_window_size = 300
# time_window_size = 10000
# episode_time_upper_bound = 4000
# episode_time_upper_bound = 1500
episode_time_upper_bound = 2000

# lookahead parameters for bridging time windows
lookahead_windows = 1  # Number of future time windows to include as lookahead operations
lookahead_threshold_ratio = 0  # Threshold ratio for early start of lookahead operations (as ratio of window size)
enable_lookahead = False  # Allow dynamic revelation of lookahead operations during execution

# training params
"""
Note: for debugging, set no_tune = True, num_env_runners = 0, and num_learners = 0
"""
# no_tune = True
no_tune = False
stop_iters = 500
stop_reward = 2
# num_env_runners = 1
num_env_runners = 10
num_learners = 0
evaluation_num_env_runners = 1
# use_lstm = True
use_lstm = False
as_test = False
framework = "torch"
local_mode = False
use_custom_loss = True
il_loss_weight = 10.0

# curriculum learning
# enable_curriculum = True
enable_curriculum = False
# training_task_id = 0
# training_task_id_min = 0
# training_task_id_max = 0
task_id_range = 50
# task_pool_length = 0
# sorted_task_pool = []

# problem instance params
factory_instance_seed = 42

max_n_machines = 10
# max_n_machines = 4
min_prcs_time = 1
max_prcs_time = 60
n_machines_is_fixed = True
# n_machines = 15
n_machines = max_n_machines
is_fully_flexible = False
min_compatible_machines = 2
time_for_compatible_machines_are_same = False
time_viration_range = 5
prcs_time_factor = 1.0

max_n_transbots = 10
# max_n_transbots = 2
min_tspt_time = 3
max_tspt_time = math.ceil(n_machines ** 0.5) * 9
# loaded_transport_time_scale = 1.5
n_transbots_is_fixed = True
# n_transbots = 3
n_transbots = max_n_transbots
tspt_time_factor = 1.0

all_machines_are_perfect = True
min_quality = 1

max_n_jobs = 200
n_jobs_is_fixed = True
# n_jobs = 15
n_jobs = max_n_jobs
n_operations_is_n_machines = False
min_n_operations = 8
max_n_operations = 12
# min_n_operations = int(n_machines * 0.8)
# max_n_operations = int(n_machines * 1.2)
consider_job_insert = True
new_arrival_jobs = 0
earliest_arrive_time = 30
latest_arrive_time = 300

# normalized_scale = max_n_operations * max_prcs_time

n_instances = 1200
n_instances_for_training = 1000
n_instances_for_evaluation = 100
n_instances_for_testing = 100
instance_generator_seed = 1000
# layout_seed = 0
current_scheduling_instance_id = 0

# env params
perform_left_shift_if_possible = True

# instance selection params
randomly_select_instance = False
# current_instance_id = 0
# imitation_env_count = 0
# env_count = 0


# render params
JobAsAction = True
gantt_y_axis = "nJob"
drawMachineToPrcsEdges = True
default_visualisations = None




