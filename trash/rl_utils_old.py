import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th
import torch.nn as nn
import os
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.integrate import odeint
from collections import deque
import json
import matplotlib.pyplot as plt

from constants import LOCATION_CHOOSEN, OUTPUT_DIR, DATA_CACHE_DIR, STRINGENCY_BASED_GDP, OPTIMAL_VALUES_FILE, MODELS_DIR
OUTPUT_RL = os.path.join(OUTPUT_DIR, "rl")

with open(OPTIMAL_VALUES_FILE, 'r') as f:
    optimal_values_read = f.read()
    optimal_values = json.loads(optimal_values_read)
optimal_beta = optimal_values['optimal_beta']
optimal_gamma = optimal_values['optimal_gamma']
optimal_stringency_weight = optimal_values['optimal_stringency_weight']

stringency_data_points = np.arange(0, 100, 0.5)
fit_line_loaded = np.poly1d(np.load(STRINGENCY_BASED_GDP))
predicted_gdp = fit_line_loaded(stringency_data_points)
MIN_GDP = min(predicted_gdp)
MAX_GDP = max(predicted_gdp)

data_path = os.path.join(DATA_CACHE_DIR, LOCATION_CHOOSEN + ".csv")
df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'])
df = df[(df['date'].dt.year == 2021) & (df['date'].dt.month >= 5) | (df['date'].dt.year == 2022)]

df['N'] = df['population']
df['S'] = df['population'] - (df['total_cases'] + df['people_fully_vaccinated'])
df['I'] = df['total_cases']
df['R'] = df['people_fully_vaccinated']
TOTAL_DAYS = (max(df['date']) - min(df['date'])).days
print(f'Total Days: {TOTAL_DAYS}')

START_STRINGENCY = df.loc[min(df.index), ['stringency_index']].item()
N = df.loc[min(df.index), ['N']].item()
y0 = df.loc[min(df.index), ['S']].item(), df.loc[min(df.index), ['I']].item(), df.loc[min(df.index), ['R']].item()
N_DISCRETE_ACTIONS = 7

def compute_cost(data, predictions):
    return np.abs(data - predictions).mean()

def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def time_varying_beta(optimal_beta, stringency_weight, stringency_index):
    beta = optimal_beta + (stringency_weight * stringency_index)
    return beta

def objective_function_2(params, y0, t, N, df, gamma, current_stringency):
    stringency_weight = params[0]
    beta_array = time_varying_beta(optimal_beta, stringency_weight, current_stringency)
    predictions = odeint(deriv, y0, t, args=(N, beta_array, gamma))
    S, I, R = predictions.T
    cost = compute_cost(df['S'], S) + compute_cost(df['I'], I) + compute_cost(df['R'], R)
    return cost

class SIREnvironment(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    def __init__(self):
        super(SIREnvironment, self).__init__()
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Box(low=-10.0, high=10, shape=(5+TOTAL_DAYS,), dtype=np.float64)
    
    def step(self, action):
        self.prev_actions.append(action)
            
        if action == 0:
            self.stringency_index = max(0, self.stringency_index - 10)
        elif action == 1:
            self.stringency_index = max(0, self.stringency_index - 5)
        elif action == 2:
            self.stringency_index = max(0, self.stringency_index - 2.5)
        elif action == 3:
            self.stringency_index = max(0, self.stringency_index + 0)
        elif action == 4:
            self.stringency_index = min(100, self.stringency_index + 2.5)
        elif action == 5:
            self.stringency_index = min(100, self.stringency_index + 5)
        elif action == 6:
            self.stringency_index = min(100, self.stringency_index + 10)
        
        # each action is on ith day
        t = np.linspace(0, self.ith_day, self.ith_day + 1)
        beta_for_stringency = time_varying_beta(self.beta_optimal, self.s_weight_optimal, self.stringency_index)
        predictions = odeint(deriv, self.y0, t, args=(self.N, beta_for_stringency, self.gamma_optimal))
        S, I, R = predictions.T
        self.store_S[self.ith_day] = S[-1]
        self.store_I[self.ith_day] = I[-1]
        self.store_R[self.ith_day] = R[-1]
        
        self.store_stringency[self.ith_day] = self.stringency_index
        self.store_gdp[self.ith_day] = fit_line_loaded(self.stringency_index)
        
        self.S_proportion = self.store_S[self.ith_day]/self.N
        self.I_proportion = self.store_I[self.ith_day]/self.N
        self.R_proportion = self.store_R[self.ith_day]/self.N
        self.normalized_GDP = (fit_line_loaded(self.stringency_index) - MIN_GDP) / (MAX_GDP - MIN_GDP)
        self.r_eff = (beta_for_stringency / self.gamma_optimal) * (self.store_S[self.ith_day] / self.N)
        self.store_r_eff[self.ith_day] = self.r_eff
        self.store_normalized_gdp[self.ith_day] = self.normalized_GDP
        
        # self.reward = self.normalized_GDP - (2 * self.r_eff)
        self.reward = self.normalized_GDP / self.r_eff
        self.store_reward[self.ith_day] = self.reward

        observation = [self.S_proportion, self.I_proportion, 
                       self.R_proportion, self.normalized_GDP, 
                       self.r_eff] + list(self.prev_actions)
        observation = np.array(observation)

        # after doing the action add to ith_day
        # and then if ith_day is the last day then end the environment
        self.ith_day += 1
        if self.ith_day > TOTAL_DAYS:
            self.terminated = True
        info = {}
        return observation, self.reward, self.terminated, self.truncated, info
    
    def render(self, score=0.0, learning=False):
        self.t = np.linspace(0, TOTAL_DAYS, TOTAL_DAYS + 1)
        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 12))
        axes[0, 0].plot(self.t, self.df['S']/self.N, 'b', alpha=0.5, lw=2, label='Susceptible Data')
        axes[0, 0].plot(self.t, self.df['I']/self.N, 'r', alpha=0.5, lw=2, label='Infected Data')
        axes[0, 0].plot(self.t, self.df['R']/self.N, 'g', alpha=0.5, lw=2, label='Recovered Data')
        axes[0, 0].plot(self.t, self.store_S/self.N, 'b--', alpha=0.5, lw=2, label='Susceptible (Model)')
        axes[0, 0].plot(self.t, self.store_I/self.N, 'r--', alpha=0.5, lw=2, label='Infected (Model)')
        axes[0, 0].plot(self.t, self.store_R/self.N, 'g--', alpha=0.5, lw=2, label='Recovered (Model)')
        axes[0, 0].set_xlabel('Time /days')
        axes[0, 0].set_ylabel('Percentage of Population')
        axes[0, 0].set_title('SIR Epidemic Trajectory')
        axes[0, 0].tick_params(length=0)
        axes[0, 0].grid(True)
        legend = axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        legend.get_frame().set_alpha(0.5)

        axes[0, 1].plot(self.t, self.store_stringency, label="Stringency")
        axes[0, 1].set_xlabel("Time /days")
        axes[0, 1].set_ylabel("Stringency Index")
        axes[0, 1].set_title("Time vs. Stringency")
        axes[0, 1].set_ylim(bottom=0, top=150)
        axes[0, 1].grid(True)

        axes[1, 0].plot(self.t, self.store_gdp, label="GDP")
        axes[1, 0].set_xlabel("Time /days")
        axes[1, 0].set_ylabel("GDP")
        axes[1, 0].set_title("Time vs. GDP")
        axes[1, 0].set_ylim(bottom=0, top=150)
        axes[1, 0].grid(True)

        axes[1, 1].plot(self.t, self.store_r_eff, label="R_eff")
        axes[1, 1].axhline(y=1, color='r', linestyle='--', label="R_eff = 1")
        first_time_r_eff_1 = next((t for t, r_eff in zip(self.t, self.store_r_eff) if r_eff <= 1), None)
        axes[1, 1].text(first_time_r_eff_1, 1, f'R_eff=1\nat t={int(first_time_r_eff_1)} days', ha='right', va='bottom')
        axes[1, 1].set_xlabel("Time /days")
        axes[1, 1].set_ylabel("R_eff")
        axes[1, 1].set_title("Time vs. R_eff")
        axes[1, 1].set_ylim(bottom=0, top=2.0)
        axes[1, 1].grid(True)
        axes[1, 1].legend()

        axes[2, 0].plot(self.t, self.store_reward, label="Reward")
        axes[2, 0].set_xlabel("Time /days")
        axes[2, 0].set_ylabel("Reward")
        axes[2, 0].set_title("Time vs. Reward")
        axes[2, 0].grid(True)
        axes[2, 0].legend()

        formatted_score = "{:.2f}".format(score)
        fig.suptitle(f"Episode Score: {formatted_score}", y=0.02)
        plt.tight_layout()
        if (learning == False):
            plt.savefig(os.path.join(OUTPUT_RL, str(score) + ".png"))
        plt.close()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.stringency_index = START_STRINGENCY
        self.N = N
        self.y0 = y0
        self.beta_optimal = optimal_beta
        self.gamma_optimal = optimal_gamma
        self.s_weight_optimal = optimal_stringency_weight
        self.df = df
        self.days_difference = (max(self.df['date']) - min(self.df['date'])).days
        self.t = np.linspace(0, self.days_difference, self.days_difference + 1)
        self.ith_day = 0
        
        self.store_S = np.zeros(TOTAL_DAYS + 1)
        self.store_I = np.zeros(TOTAL_DAYS + 1)
        self.store_R = np.zeros(TOTAL_DAYS + 1)
        
        self.store_stringency = np.zeros(TOTAL_DAYS + 1)
        self.store_gdp = np.zeros(TOTAL_DAYS + 1)
        self.store_normalized_gdp = np.zeros(TOTAL_DAYS + 1)
        self.store_r_eff = np.zeros(TOTAL_DAYS + 1)
        self.store_reward = np.zeros(TOTAL_DAYS + 1)
        
        # self.prev_reward = 0
        self.terminated = False
        self.truncated = False
        
        self.S_proportion = self.y0[0]/self.N
        self.I_proportion = self.y0[1]/self.N
        self.R_proportion = self.y0[2]/self.N
        self.normalized_GDP = (fit_line_loaded(self.stringency_index) - MIN_GDP) / (MAX_GDP - MIN_GDP)
        beta_for_stringency = time_varying_beta(self.beta_optimal, self.s_weight_optimal, self.stringency_index)
        self.r_eff = (beta_for_stringency / self.gamma_optimal) * (self.store_S[self.ith_day] / self.N)
        
        self.prev_actions = deque(maxlen = TOTAL_DAYS)
        for i in range(TOTAL_DAYS):
            self.prev_actions.append(-1) 
        
        # create the observation
        observation = [self.S_proportion, self.I_proportion, 
                       self.R_proportion, self.normalized_GDP, 
                       self.r_eff] + list(self.prev_actions)
        observation = np.array(observation)
        info = {}
        return observation, info

# Neural network for predicting action values
class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """_summary_"""
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 614):  # type: ignore
        super().__init__(observation_space, features_dim)
        self.LSTM = nn.LSTM(input_size=features_dim, hidden_size=100, num_layers=1)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # th.tensor(observations)
        observations.clone().detach()
        self.LSTM_output, self.LSTM_hidden = self.LSTM(observations)
        return self.LSTM_output + self.LSTM_hidden[0] + self.LSTM_hidden[1]
# class CustomCNN(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Box, features_dim: int=128):
#         super(CustomCNN, self).__init__(observation_space, features_dim)
#         # CxHxW images (channels first)
#         n_input_channels = observation_space.shape[0]
#         print("N INPUT CHANNELS: ", n_input_channels)
#         self.cnn = nn.Sequential(
#             nn.Linear(in_features=(n_input_channels), out_features=(100), bias=False),
#             nn.ReLU(),
#             nn.Flatten(),
#         )
#         with torch.no_grad():
#             n_flatten = self.cnn(
#                 torch.as_tensor(observation_space.sample()[None]).float()
#             ).shape[1]

#         self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

#     def forward(self, observations: torch.Tensor) -> torch.Tensor:
#         return self.linear(self.cnn(observations))
