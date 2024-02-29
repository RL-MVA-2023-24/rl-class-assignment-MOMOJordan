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

    try:

        config = {
                  'nb_actions': env.action_space.n,
                  'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
                  'gamma': trial.suggest_float('gamma', 0.9, 0.999),
                  'buffer_size': trial.suggest_int('buffer_size', 100000, 1000000),
                  'epsilon_min': 0.01,
                  'epsilon_max': 1.,
                  'epsilon_decay_period': 1000,
                  'epsilon_delay_decay': 20,
                  'batch_size': trial.suggest_int('batch_size', 16, 128),
                  'nb_neurons': trial.suggest_int('nb_neurons', 8, 256)
                  }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Declare network
        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n
        nb_neurons = config['nb_neurons']
        DQN = torch.nn.Sequential(nn.Linear(state_dim, nb_neurons),
                                  nn.ReLU(),
                                  nn.Linear(nb_neurons, nb_neurons),
                                  nn.ReLU(),
                                  nn.Linear(nb_neurons, nb_neurons),
                                  nn.ReLU(),
                                  nn.Linear(nb_neurons, n_action)).to(device)

        # DQN config
        

        # Train agent
        agent = ProjectAgent(config, DQN)
        scores = agent.train(env, 1)

        # Keep the following lines to evaluate your agent unchanged.
        score_agent: float = evaluate_HIV(agent=agent, nb_episode=1)
        score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=2)

        return score_agent+score_agent_dr/2


    except Exception as e:
        raise optuna.exceptions.TrialPruned(f'Skipping this trial due to an error: {str(e)}')



def run_study(env, n_trials=10, n_jobs=2, **kwargs):

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