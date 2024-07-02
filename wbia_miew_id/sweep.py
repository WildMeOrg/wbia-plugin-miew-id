import optuna
import yaml
from train import run
from helpers import get_config
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pickle
import signal, sys


import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Load configuration file.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.yaml",
        help="Path to the YAML configuration file. Default: configs/default_config.yaml",
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        default=40,
        help="Number of trials. Default: 100",
    )


    return parser.parse_args()


def objective(trial, config):

    # Specify the parameters you want to optimize

    image_size = 440 #trial.suggest_categorical("image_size", [192, 256, 384, 440, 512])
    n_epochs = 25#trial.suggest_int("epochs", 20, 40)
    config.engine.epochs = n_epochs

    # if trial.number > 0:

    # loss_module = trial.suggest_categorical("loss_module", ['arcface_subcenter_dynamic', 'elastic_arcface'])
    # config.model_params.loss_module = loss_module
    # config.data.image_size = [image_size, image_size]
    # Specify the parameters you want to optimize
    # config.data.train.n_filter_min = trial.suggest_int("train.n_filter_min", 3, 5)
    # image_size = 256 #trial.suggest_categorical("image_size", [192, 256, 384, 440, 512])

    config.model_params.s = trial.suggest_uniform("s", 30, 64)
    if config.model_params.loss_module == 'elastic_arcface':
        config.model_params.margin = trial.suggest_uniform("margin", 0.3, 0.7)
    if config.model_params.loss_module == 'arcface_subcenter_dynamic':
        config.model_params.k = trial.suggest_int("k", 2, 4)

    # The scheduler params are derived from one base paremeter to minimize the number of parameters to optimzie
    # lr_base = trial.suggest_loguniform("lr_base", 1e-5, 1e-3)
    # config.scheduler_params.lr_start = lr_base / 100
    # config.scheduler_params.lr_max = lr_base * 10
    # config.scheduler_params.lr_min = lr_base / 100

    lr_start = trial.suggest_loguniform("lr_start", 1e-7, 1e-4)
    config.scheduler_params.lr_start = lr_start 
    lr_max = trial.suggest_loguniform("lr_max", 5e-5, 1e-3)
    config.scheduler_params.lr_max = lr_max
    lr_min = trial.suggest_loguniform("lr_min", 1e-7, 1e-4)
    config.scheduler_params.lr_min = lr_min

    # # SWA parameters to test
    # config.engine.use_swa = trial.suggest_categorical("use_swa", [False, True])
    # if config.engine.use_swa:
    #     config.swa_params.swa_lr = trial.suggest_loguniform("swa_lr", 0.0001, 0.05)
    #     config.swa_params.swa_start = trial.suggest_int("swa_start", 20, 25)


    print("trial number: ", trial.number)
    print("config: ", dict(config))

    if trial.number > 0:
        config.exp_name = config.exp_name.rsplit('_', 1)[0] + f"_t{trial.number}"
    else:
        config.exp_name = config.exp_name + f"_t{trial.number}"

    if trial.number == 0:
        return 0.738
    if trial.number == 1:
        return 0.59
    if trial.number == 2:
        return 0.7422174440141309
    if trial.number == 3:
        return 0.7422935392772942
    else:
        try:
            result = run(config)
        except Exception as e:
            print("Exception occured: ", e)
            print(trial)
            result = 0

    return result

## Probably only needed to reproduce the results of same seed
def signal_handler(signum, frame):
    global study_sampler
    print("\nSaving the current state of the study before exiting...")
    with open('study_sampler.pkl', 'wb') as f:
        pickle.dump(study, f)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


if __name__ == "__main__":
    args = parse_args()
    config_path = args.config
    n_trials = args.n_trials

    config = get_config(config_path)

    study_name = config.exp_name
    study_storage = f"sqlite:///{study_name}.db"
    study_sampler = TPESampler(seed=0)

    study = optuna.create_study(
        study_name=study_name, storage=study_storage,
        sampler=study_sampler, pruner=MedianPruner(), direction="maximize", load_if_exists=True
    )

    study.enqueue_trial({
        's': 49.32675426153405,
        'margin': 0.32841442327915477,
        'k': 2,
        'lr_start': 0.00000341898067433194,
        'lr_max': 0.001,
        'lr_min': 0.000002
        })
    
    study.enqueue_trial({
        's': 48.65965913352904,
        'margin': 0.32841442327915477,
        'k': 4,
        'lr_start': 0.00000643117205013199,
        'lr_max': 0.00000643117205013199,
        'lr_min': 0.00000643117205013199
    })

    study.enqueue_trial({
        's': 48.65965913352904,
        'margin': 0.32841442327915477,
        'k': 4,
        'lr_start': 0.00000643117205013199,
        'lr_max': 0.0002557875307967728,
        'lr_min': 0.0000018662266976518
    })

    study.enqueue_trial({
        's': 51.960399844266306,
        'margin': 0.32841442327915477,
        'k': 3,
        'lr_start': 0.0000473498930449948,
        'lr_max': 0.000896858981000587,
        'lr_min': 0.00000141359355517523
    })

    comb_objective = lambda trial: objective(trial, config)

    study.optimize(comb_objective, n_trials=n_trials)

    print("Best trial:")
    trial_ = study.best_trial

    print(f"Value: {trial_.value}")

    print("Best parameters:")
    for key, value in trial_.params.items():
        print(f"    {key}: {value}")

    # saves best parameters
    save_dict = trial_.params
    save_dict['best_score'] = trial_.value

    with open('sweep.pkl', 'wb') as f:
        pickle.dump(save_dict, f, pickle.HIGHEST_PROTOCOL)
