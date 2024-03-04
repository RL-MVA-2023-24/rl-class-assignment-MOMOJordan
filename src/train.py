from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import random
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from evaluate import evaluate_HIV, evaluate_HIV_population

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.


# Now is the floor is yours to implement the agent and train it.

class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = capacity  # capacity of the buffer
        self.data = []
        self.index = 0  # index of the next cell to be filled
        self.device = device

    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x: torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))

    def __len__(self):
        return len(self.data)


def greedy_action_dqn(network, state, device="cpu"):

    with (torch.no_grad()):
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        # Q += Q.abs().std() * torch.randn_like(Q) * 0.1
        return torch.argmax(Q).item()


class dqn_agent:
    def __init__(self, config={'1': 1}, model=None):
        super().__init__()
        self.patience=15
        self.early_stop=0
        self.best_score=-np.inf
        self.config = config
        device = "cpu"
        self.device = device
        self.nb_actions = config['nb_actions'] if 'nb_actions' in config else 0
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(buffer_size, device)
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max - self.epsilon_min) / self.epsilon_stop
        self.model = model
        self.target_model = deepcopy(self.model).to(device) if model is not None else None
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = None if model is None else config[
            'optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(
            self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config[
            'update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        self.monitoring_nb_trials = config['monitoring_nb_trials'] if 'monitoring_nb_trials' in config.keys() else 0

        self.seed = config.get('seed', 123456789)

        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.seed)
    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1 - D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


    def train(self, env, max_episode):
        episode_return = []

        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)
                self.epsilon=epsilon
            # select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = greedy_action_dqn(self.model, state)
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            # train
            for _ in range(self.nb_gradient_steps):
                self.gradient_step()
            # update target network if needed
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0:
                    self.target_model.load_state_dict(self.model.state_dict())
            if self.update_target_strategy == 'ema':
                target_state_dict = self.target_model.state_dict()
                model_state_dict = self.model.state_dict()
                tau = self.update_target_tau
                for key in model_state_dict:
                    target_state_dict[key] = tau * model_state_dict + (1 - tau) * target_state_dict
                self.target_model.load_state_dict(target_state_dict)
            # next transition
            step += 1
            if done or trunc:

                if episode>100:

                    score_agent=min(evaluate_HIV(agent=self, nb_episode=1), evaluate_HIV_population(agent=self, nb_episode=1))
                else:
                    score_agent=0

                if score_agent>self.best_score:

                    self.best_score=score_agent
                    try:
                        self.save()
                    except:
                        pass


                # if score_agent>=1e8:
                #     return episode_return

                episode += 1
                # Monitoring
                if self.monitoring_nb_trials > 0:
                    episode_return.append(episode_cum_reward)  # NEW NEW NEW
                    # print("Episode ", '{:2d}'.format(episode),
                    #       ", epsilon ", '{:6.2f}'.format(epsilon),
                    #       ", batch size ", '{:4d}'.format(len(self.memory)),
                    #       ", ep return ", '{:4.1f}'.format(episode_cum_reward),
                    #       ", real score ", '{:4.1f}'.format(score_agent),
                    #       sep='')
                else:
                    episode_return.append(episode_cum_reward)
                    # print("Episode ", '{:2d}'.format(episode),
                    #       ", epsilon ", '{:6.2f}'.format(epsilon),
                    #       ", batch size ", '{:4d}'.format(len(self.memory)),
                    #       ", ep return ", '{:4.1f}'.format(episode_cum_reward),
                    #       ", real score ", '{:4.1f}'.format(score_agent),
                    #       sep='')

                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return

class ProjectAgent(dqn_agent):
    def __init__(self, *args, **kwargs):

        # if len(args)!=0 or len(args)!=0:
        super().__init__(*args, **kwargs)
        self.model_loaded = False
        self.model_save_path = 'src/model_dqn0.pt'
        self.action_taken = {0: 0, 1: 0, 2: 0, 3: 0}

        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.seed)

    def act(self, observation, use_random=True):
        if self.model is None:
            self.load()

        action_taken = greedy_action_dqn(self.model, observation)

        return action_taken

    def save(self, path='model_dqn0.pt'):
        self.model_save_path = path
        torch.save({'config': self.config, 'model': self.model}, self.model_save_path)

    def load(self):
        self.model_loaded = True
        saved_result = torch.load(self.model_save_path, map_location=torch.device('cpu'))

        self.model = saved_result['model']
        self.config = saved_result['config']

        for key, value in saved_result['config'].items():
            try:
                setattr(self, key, value)
            except:
                print(key, value)

        self.seed = self.config['seed']
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.seed)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cartpole = env
    # Declare network
    state_dim = cartpole.observation_space.shape[0]
    n_action = cartpole.action_space.n

    config = {'nb_actions': env.action_space.n,
              'learning_rate': 0.001,
              'gamma': 0.95, # 0.98 good
              'buffer_size': int(1e5),
              'epsilon_min': 0.02,
              'epsilon_max': 1.,
              'epsilon_decay_period': 20000,
              'epsilon_delay_decay': 400,

              'gradient_steps': 2,
              'update_target_strategy': 'replace',  # or 'ema'
              'update_target_freq': 400,
              'update_target_tau': 0.005,
              'criterion': torch.nn.SmoothL1Loss(),
              'monitoring_nb_trials': 50,

              'batch_size': 800,
              'nb_neurons': 256,
              'seed': 25,
              }

    nb_neurons = config['nb_neurons']

    DQN = torch.nn.Sequential(nn.Linear(state_dim, 256),
                              nn.ReLU(),
                              nn.Linear(256, 256),
                              nn.ReLU(),
                              nn.Linear(256, 256),
                              nn.ReLU(),
                              nn.Linear(256, 256),
                              nn.ReLU(),
                              nn.Linear(256, 256),
                              nn.ReLU(),
                              nn.Linear(256, n_action)).to(device)

    # Train agent
    agent_best = ProjectAgent(config, DQN)

    scores = agent_best.train(cartpole, 210)

    agent_best.save()


