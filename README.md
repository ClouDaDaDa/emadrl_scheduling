## Embodied Multi‑Agent Scheduling with Partial Global Plans (EMADRL)

This repository contains the official source code and experiment assets for our SCI submission "Embodied Multi‑Agent Scheduling for Workshop Operations with Partial Global Plans". It implements a global–local collaborative scheduling system where a higher‑level global planner provides long‑horizon guidance and local embodied agents (machines and transport robots) make millisecond‑level, constraint‑aware decisions in real time.

### Highlights
- **Global–local collaboration**: Global schedules are decomposed into rolling time windows with optional lookahead; execution feedback updates subsequent windows to stay aligned with global objectives.
- **Embodied agents**: Local scheduling explicitly models physical dynamics/constraints (machine degradation/maintenance, AGV charging and routing, congestion/collision avoidance, stochastic times).
- **EMADRL algorithm**: Dynamic observation selection, action‑space pruning via feasibility masks, and heterogeneous multi‑agent training that scales across workshop sizes.
- **Performance**: Clear makespan reductions vs. strong baselines and generalization from small to industrial‑scale instances without retraining (per paper results).


## Repository structure

```
emadrl_scheduling/
  configs/                       # Global configuration (problem size, training/runtime flags)
  global_scheduling/             # Global DFJSPT scheduler, rules, and conversion utilities
    dfjspt_env.py
    dfjspt_rule/                 # Global dispatching rules (e.g., EST/EET/EET)
    InterfaceWithLocal/
      convert_schedule_to_class.py   # Convert arrays/routes -> GlobalSchedule (pickle)
      global_schedules/              # Ground-truth global schedules (pkl), plotting
      load_pkl_schedule.py
  local_realtime_scheduling/
    Environment/
      LocalSchedulingMultiAgentEnv_v3_4.py  # Embodied multi-agent env (machines + transbots)
      ExecutionResult.py, path_planning.py  # Execution logging and routing
    Agents/
      train_hyper_param_tunning.py          # EMADRL training (RLlib PPO)
      action_mask_module.py                 # Torch RLModule with action masking (feasibility)
      customized_callback.py                # Training/eval callbacks
      generate_training_data.py             # Reset options from local schedules
    InterfaceWithGlobal/
      divide_global_schedule_to_local_from_pkl.py  # Split global schedule pkl -> rolling local windows (pkl)
      plot_local_gantt_from_pkl.py, plot_local_gantt_from_json.py, plot_global_gantt_from_json.py
      local_schedules/ M{M}T{T}W{W}/training|testing/  # Pre-generated local windows (pkl)
    BaselineMethods/
      DispatchingRules/                     # Heuristic baselines for machines/transbots
      SingleAgentDRL/                       # Single-agent DRL baseline
      NonEmbodiedScheduling/                # Event-driven non-embodied multi-agent baseline
    test_local_parallel/, test_global_parallel/      # Parallel comparison runners (Ray)
    result_analyzer.py, result_collector.py          # Result aggregation/plots
  System/                         # Domain model: Factory, Machine, AGV, Battery, Job, etc.
  requirements.txt
  README.md (this file)
```


## Quick start

### 1) Environment setup

Tested with Python 3.10+. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Torch and Ray are pinned in `requirements.txt`. CPU runs are supported; GPU is optional.


### 2) Configure problem size and training/runtime flags

Edit `configs/dfjspt_params.py` (key options below). Defaults provide ready-to-run settings with pre-generated local schedules under `local_realtime_scheduling/InterfaceWithGlobal/local_schedules/`.

- **Problem size**: `n_machines`, `n_transbots`, `n_jobs`, operations per job
- **Windowing**: `time_window_size`, `lookahead_windows`, `enable_lookahead`
- **Training**: `num_env_runners`, `num_learners`, `use_lstm`, `no_tune`, `stop_iters`, `stop_reward`
- **Curriculum**: `enable_curriculum`, `task_id_range`
- **Seeds**: `factory_instance_seed`, `instance_generator_seed`


### 3) Train EMADRL (embodied multi-agent PPO)

```bash
python local_realtime_scheduling/Agents/train_hyper_param_tunning.py \
  --framework torch --checkpoint-at-end --verbose 2
```

Logs and checkpoints are written to:
- `local_realtime_scheduling/Agents/ray_results/M{M}T{T}W{W}/`

Notes
- Training uses `LocalSchedulingMultiAgentEnv_v3_4` and the action-masked `ActionMaskingTorchRLModule`.
- By default, training samples local windows randomly from `InterfaceWithGlobal/local_schedules/M{M}T{T}W{W}/training/`.
- Set `configs/dfjspt_params.no_tune = True` for a simple `algo.train()` loop instead of Ray Tune.


### 4) Baselines

- Heuristic dispatching rules (machines + transbots):
  ```bash
  python local_realtime_scheduling/BaselineMethods/DispatchingRules/dispatching_rules_test_v3.py
  ```
  The script evaluates rules on a chosen local instance and can iterate over all testing windows.

- Single-agent DRL baseline:
  ```bash
  python local_realtime_scheduling/BaselineMethods/SingleAgentDRL/sa_train_hyper_param_tunning.py
  ```

- Non-embodied multi-agent baseline (event-driven, no physical constraints):
  ```bash
  python local_realtime_scheduling/BaselineMethods/NonEmbodiedScheduling/train_non_embodied.py
  ```


### 5) Parallel comparisons (Ray)

- Local-window level comparison across many testing windows and policies:
  ```bash
  python -c "from local_realtime_scheduling.test_local_parallel.compare_local_instances_parallel import run_parallel_local_instance_comparison; \
  run_parallel_local_instance_comparison(num_repeat=3, max_instances=200)"
  ```
  If you pass `checkpoint_dir` (an RLlib checkpoint), `'madrl'` will be included alongside heuristic policies.

- Global-instance level comparison (process all windows of each global instance sequentially):
  ```bash
  python -c "from local_realtime_scheduling.test_global_parallel.compare_global_instances_parallel import run_parallel_global_instance_comparison; \
  run_parallel_global_instance_comparison(num_repeat=3, job_sizes=[100,150], instance_ids=list(range(100,103)), resume_mode=False)"
  ```

Results are aggregated under timestamped folders in `local_realtime_scheduling/results_data/` with CSVs and pickles for analysis.


## Data pipeline: from global to local

1) A global scheduler produces ground-truth routes and timings (via `global_scheduling/dfjspt_env.py` and rules under `dfjspt_rule/`).
2) `InterfaceWithLocal/convert_schedule_to_class.py` converts arrays/routes to a pickled `GlobalSchedule` object.
3) `InterfaceWithGlobal/divide_global_schedule_to_local_from_pkl.py` splits each global schedule into rolling local windows with optional lookahead and saves pickled `LocalSchedule` files using the pattern:
   `local_schedule_J{jobs}I{instance}_{window}_ops{n_ops}.pkl`.

This repository already includes many local windows under:
`local_realtime_scheduling/InterfaceWithGlobal/local_schedules/M{M}T{T}W{W}/training|testing/`

To regenerate local schedules programmatically (when you have global schedule pickles in `global_scheduling/InterfaceWithLocal/global_schedules/M{M}T{T}/`):

```bash
python -c "from local_realtime_scheduling.InterfaceWithGlobal.divide_global_schedule_to_local_from_pkl import generate_local_schedules; \
generate_local_schedules(for_training=True); generate_local_schedules(for_training=False)"
```


## Embodied environment (local real-time layer)

- File: `local_realtime_scheduling/Environment/LocalSchedulingMultiAgentEnv_v3_4.py`
- Agents: `machine{i}` for machines, `transbot{j}` for transport robots.
- Observations (dict):
  - `action_mask`: 1 for feasible actions, 0 otherwise (physics/constraints enforced).
  - `observation`:
    - Machines: `job_features`, `neighbor_machines_features`, `machine_features`
    - Transbots: `job_features`, `neighbor_transbots_features`, `transbot_features`
- Actions: discrete decisions per agent; feasibility enforced by the `action_mask` (last action often corresponds to no-op/do-nothing).
- Reset options: produced by `Agents/generate_training_data.py` by sampling a `LocalSchedule` pickle and constructing consistent `FactoryInstance`/`SchedulingInstance` states with optional randomness for training.
- Rendering/plots: environment provides Gantt and trajectory plotting helpers; additional plotting utilities are under `InterfaceWithGlobal/`.


## Plotting

- Plot local Gantt from a `LocalSchedule` pickle:
  ```bash
  python local_realtime_scheduling/InterfaceWithGlobal/plot_local_gantt_from_pkl.py
  ```


## Key configuration knobs (excerpt from `configs/dfjspt_params.py`)

- `time_window_size` (int): Rolling window length.
- `lookahead_windows` (int): Number of future windows included as lookahead.
- `enable_lookahead` (bool): Dynamically reveal lookahead ops during execution.
- `episode_time_upper_bound` (int): Safety cap for episode length.
- `no_tune` (bool): If True, bypass Ray Tune and run a plain training loop.
- `num_env_runners`, `num_learners`: Parallel sampling/learning scale (Ray).
- `use_lstm` (bool): Enable LSTM in the RLModule.
- `n_machines`, `n_transbots`, `n_jobs`, `min_n_operations`…: Problem size.
- `factory_instance_seed`, `instance_generator_seed`: Reproducibility seeds.



## Ray 2.40 source modifications

To realize certain EMADRL training/runtime features, we applied a few small edits to the Ray 2.40 codebase. Please see:

`local_realtime_scheduling/Agents/note.md`

for a concise list of the touched files and how to apply these changes (or remove them). The repository pins `ray==2.40.0`. Baselines and utilities work with stock Ray; to fully reproduce EMADRL behavior, follow the note to patch your Ray installation as described.




