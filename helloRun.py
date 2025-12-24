from types import SimpleNamespace
from run import run_ga 


args_dict = {
    "pop_size": 10,
    "mutate_rate": 0.02,
    "cross_rate": 0.5,
    "cross_style": "cols",
    "n_trials": 1000,        
    "reservoir_type": "esn",   # or "lsm" later
    "input_nodes": 0,
    "output_nodes": 0,
    "order": 10,
    "task": "narma",
    "max_size": 200,
    "metric": None,         
    "n_states": 3,
    "output_file": "fitness_local.db",
    "heavy_log": False
}

# choose a unique ID for this experiment
args = SimpleNamespace(**args_dict, run_id=114)

run_ga(args.run_id, args)