import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import optuna
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from train import ProjectAgent
from evaluate import *
import torch
import torch.nn as nn
def objective(trial,env, **kwargs):
    n_splits = 4
    try:

        param_grid = {
            # "device_type": trial.suggest_categorical("device_type", ['gpu']),
            'objective': 'multiclass',
            'num_class': 3,
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 1000, step=50),
            # "min_data_in_bin":trial.suggest_int("min_data_in_leaf", 1, 200, step=50),
            # "lambda_l1": trial.suggest_float("lambda_l1", 0, 10),
            # "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.01, 10),
            'num_leaves': trial.suggest_int('num_leaves', 10, 1000),  # Adjusted for the smaller dataset
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.5),  # Adjusted for smaller dataset
            'max_depth': trial.suggest_int('max_depth', 5, 16),
            'min_child_samples': trial.suggest_int('min_child_samples', 40, 100),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-5, 1),
            'n_estimators': trial.suggest_int('n_estimators', 20, 1000,step=10),
            # 'random_state': 42
        }

        device = torch.device("cpu")
        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n
        nb_neurons = 20
        DQN = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
                                  nn.ReLU(),
                                  nn.Linear(nb_neurons, nb_neurons),
                                  nn.ReLU(),
                                  nn.Linear(nb_neurons, n_action)).to(device)

        agent = ProjectAgent(param_grid, DQN)
        _, _, _, _ = agent.train(env, 50)

        # Keep the following lines to evaluate your agent unchanged.
        score_agent: float = evaluate_HIV(agent=agent, nb_episode=1)
        score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=15)

        return score_agent+score_agent_dr

    except Exception as e:
        raise optuna.exceptions.TrialPruned(f'Skipping this trial due to an error: {str(e)}')



def run_study(X_train,Y_train, n_trials=10, n_jobs=2, **kwargs):
    env = TimeLimit(
        env=HIVPatient(domain_randomization=False), max_episode_steps=10
    )
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, env,**kwargs), n_trials=n_trials, n_jobs=n_jobs)

    # Get the best parameters
    best_params = study.best_params
    best_params['verbosity'] = -1
    print("Best params:", best_params)

    # Train the final model with the best parameters

    # Evaluate the final model
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trial:")
    study.trials_dataframe().to_excel(r'result_summary.xlsx')
    # fig = optuna.visualization.plot_param_importances(study)
    # fig.write_image(r"fig_params.png", format='png', engine='kaleido')
    # fig = optuna.visualization.plot_optimization_history(study)
    # fig.write_image(r"fig_history.png", format='png', engine='kaleido')

    trial = study.best_trial
    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return study

if __name__=="__main__":
    pass
    # X_train=pd.read_csv('X_train_debug.csv',index_col='ID')
    # Y_train=pd.read_csv('Y_train.csv',index_col='ID')
    # run_study(X_train, Y_train['TARGET'].values, n_trials=2, n_jobs=1) #