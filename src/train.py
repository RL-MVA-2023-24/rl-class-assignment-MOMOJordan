from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import random
import numpy as np
import torch

from copy import deepcopy

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=10
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


def greedy_action_dqn(network, state):
    device =  "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        return torch.argmax(Q).item()

class dqn_agent():
    def __init__(self, config={'1':1}, model=None):
        super().__init__()
        self.config=config
        device = "cpu"
        self.device=device
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
        self.optimizer = None if model is None else config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(
            self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config[
            'update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        self.monitoring_nb_trials = config['monitoring_nb_trials'] if 'monitoring_nb_trials' in config.keys() else 0

        self.seed=config.get('seed',123456789)

        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.seed)

    def set_seed(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.seed)

    def MC_eval(self, env, nb_trials):  # NEW NEW NEW
        MC_total_reward = []
        MC_discounted_reward = []
        for _ in range(nb_trials):
            x, _ = env.reset()
            done = False
            trunc = False
            total_reward = 0
            discounted_reward = 0
            step = 0
            while not (done or trunc):
                a = greedy_action_dqn(self.model, x)
                y, r, done, trunc, _ = env.step(a)
                x = y
                total_reward += r
                discounted_reward += self.gamma ** step * r
                step += 1
            MC_total_reward.append(total_reward)
            MC_discounted_reward.append(discounted_reward)
        return np.mean(MC_discounted_reward), np.mean(MC_total_reward)

    def V_initial_state(self, env, nb_trials):  # NEW NEW NEW
        with torch.no_grad():
            for _ in range(nb_trials):
                val = []
                x, _ = env.reset()
                val.append(self.model(torch.Tensor(x).unsqueeze(0).to(self.device)).max().item())
        return np.mean(val)

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
        MC_avg_total_reward = []  # NEW NEW NEW
        MC_avg_discounted_reward = []  # NEW NEW NEW
        V_init_state = []  # NEW NEW NEW
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        while episode < max_episode:
            # update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)
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
                episode += 1
                # Monitoring
                if self.monitoring_nb_trials > 0:
                    MC_dr, MC_tr = self.MC_eval(env, self.monitoring_nb_trials)  # NEW NEW NEW
                    V0 = self.V_initial_state(env, self.monitoring_nb_trials)  # NEW NEW NEW
                    MC_avg_total_reward.append(MC_tr)  # NEW NEW NEW
                    MC_avg_discounted_reward.append(MC_dr)  # NEW NEW NEW
                    V_init_state.append(V0)  # NEW NEW NEW
                    episode_return.append(episode_cum_reward)  # NEW NEW NEW
                    print("Episode ", '{:2d}'.format(episode),
                          ", epsilon ", '{:6.2f}'.format(epsilon),
                          ", batch size ", '{:4d}'.format(len(self.memory)),
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward),
                          ", MC tot ", '{:6.2f}'.format(MC_tr),
                          ", MC disc ", '{:6.2f}'.format(MC_dr),
                          ", V0 ", '{:6.2f}'.format(V0),
                          sep='')
                else:
                    episode_return.append(episode_cum_reward)
                    print("Episode ", '{:2d}'.format(episode),
                          ", epsilon ", '{:6.2f}'.format(epsilon),
                          ", batch size ", '{:4d}'.format(len(self.memory)),
                          ", ep return ", '{:4.1f}'.format(episode_cum_reward),
                          sep='')

                state, _ = env.reset()
                episode_cum_reward = 0
            else:
                state = next_state
        return episode_return, MC_avg_discounted_reward, MC_avg_total_reward, V_init_state
    
class ProjectAgent(dqn_agent):
    def __init__(self, *args, **kwargs):
        
        # if len(args)!=0 or len(args)!=0:
        super().__init__(*args,**kwargs)
        self.model_loaded = False
        self.model_save_path = 'model_dqn0.pt'
        self.action_taken = {0: 0, 1: 0, 2: 0, 3: 0}
        

    def act(self, observation, use_random=False):
        if not self.model_loaded:
            self.load()
            # agent_model=self.saved_agent.model
            # agent_model.eval()
        action_taken=greedy_action_dqn(self.saved_agent, observation)
        self.action_taken[action_taken]+=1
        print(self.action_taken)
        return action_taken

    def save(self, path='model_dqn0.pt'):
        self.model_save_path = path
        # torch.save(self.model.state_dict(), self.model_save_path)
        torch.save({'config':self.config,'model':self.model}, self.model_save_path)
        # pass

    def load(self):
        self.model_loaded = True
        saved_result=torch.load(self.model_save_path,map_location=torch.device('cpu'))

        for key, value in saved_result['config'].items():
            try:
                setattr(self, key, value)
            except:
                print(key,value)

        # self.seed=self.config['seed']
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.seed)

        self.saved_agent=saved_result['model']




