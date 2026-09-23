from experiment_launcher import Launcher, is_local

from itertools import product

import hiyapyco
import os


def main():
    # Choices: air_hockey_vel, planar_air_hockey, planar_air_hockey_vel, quadrotor,
    env = "air_hockey_vel"
    # Choices: sac, baseline-atacom_sac, baseline-atacom_sac_dc, datacom_sac, datacom_sac_dc
    alg = "baseline-atacom_sac_dc"

    # Load configs based on the algorithm and environment, there are defaults for each algorithm. They are merged
    # with the environment specific config if it exists.
    configs = [os.path.join("configs", "defaults", f"{part}.yaml") for part in alg.split("_")]

    parts = ''
    for part in alg.split("_"):
        parts += f"{part}" if parts == '' else f"_{part}"
        if os.path.exists(os.path.join("configs", f"{parts}_{env}.yaml")):
            configs.append(os.path.join("configs", f"{parts}_{env}.yaml"))

    config = hiyapyco.load(os.path.join("configs", "defaults", "defaults.yaml"), *configs)
    
    # Check if on a slurm cluster or local machine
    local = is_local()
    if local:
        debug = True
        use_wandb = False
        n_seeds = 1    
        n_exp_in_parallel = 1
    else:
        debug = False
        use_wandb = True
        n_seeds = 10
        n_exp_in_parallel = 1

    # config['record'] = False
    n_record = int(config['record'])

    start_seed=0

    if n_record:
        launcher_record = Launcher(f'{env}_record', f"experiment", n_seeds=n_record, start_seed=start_seed, memory_per_core=10000, n_cores=1,
                        conda_env="d_atacom", hours=24, n_exps_in_parallel=n_record, partition="stud", gres='gpu:rtx2080:1')
    if n_seeds - n_record > 0:
        launcher = Launcher(env, f"experiment", n_seeds=n_seeds-n_record, start_seed=start_seed+n_record, memory_per_core=3000, n_cores=1,
                        conda_env="d_atacom", hours=24, n_exps_in_parallel=max(n_exp_in_parallel-n_record, 1), partition="stud")

    keys = []
    values = []
    for key, value in config.items():
        if type(value) is list:
            keys.append(key)
            values.append(value)

    if len(values) == 0:
        keys.append(next(iter(config.keys())))
        values.append([config[keys[0]]])

    for setting in product(*values):
        for i, (key, value) in enumerate(zip(keys, values)):
            update_param(config, key, setting[i], value)

        wandb_options = dict(
            wandb_enabled=use_wandb,
            wandb_entity=os.environ.get("WANDB_ENTITY", "paolo-magliano"),
            wandb_project=f"{env}",
            wandb_group=f"{alg}"
        )

        assert not " " in wandb_options["wandb_group"], "NO SPACE IN GROUP NAME"

        if n_record:
            launcher_record.add_experiment(env_name=env, alg=alg, debug=debug, n_exp_in_parallel=n_record, **config,
                                **wandb_options)
        if n_seeds - n_record > 0:
            config_not_record = config.copy()
            config_not_record['record'] = False
            launcher.add_experiment(env_name=env, alg=alg, debug=debug, n_exp_in_parallel=n_exp_in_parallel, **config_not_record,
                                **wandb_options)

    if n_record:
        launcher_record.run(local, False)
    if n_seeds - n_record > 0:
        launcher.run(local, False)


def update_param(config, key, value, value_list):
    if len(value_list) > 1:
        config.pop(key, None)
        config[f"{key}__"] = value
    else:
        config[key] = value


if __name__ == "__main__":
    main()
