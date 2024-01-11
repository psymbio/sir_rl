import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
import torch as th
import torch.nn as nn
import os
import gymnasium as gym
from gymnasium import spaces
from scipy.integrate import odeint
from collections import deque
import json
import matplotlib.pyplot as plt

from constants import LOCATION_CHOOSEN, OUTPUT_DIR, DATA_CACHE_DIR, STRINGENCY_BASED_GDP, OPTIMAL_VALUES_FILE, MODELS_DIR, RL_LEARNING_TYPE
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

data_path = os.path.join(DATA_CACHE_DIR, LOCATION_CHOOSEN + "_merged_data.csv")
df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'])
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
        if RL_LEARNING_TYPE == "normal":
            self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(5 + TOTAL_DAYS + 1,), dtype=np.float64)
        elif RL_LEARNING_TYPE == "deep":
            # code reference: https://github.com/DLR-RM/stable-baselines3/issues/1713
            self.observation_space = spaces.Dict({"stringency": spaces.Box(low=-10.0, high=10, shape=(TOTAL_DAYS + 1,), dtype=np.float64),
                                                  "normalized_gdp": spaces.Box(low=-10.0, high=10, shape=(TOTAL_DAYS + 1,), dtype=np.float64),
                                                  "r_eff": spaces.Box(low=-10.0, high=10, shape=(TOTAL_DAYS + 1,), dtype=np.float64),
                                                  "other_stats": spaces.Box(low=-10.0, high=10, shape=(3,), dtype=np.float64)
                                                  })
    def step(self, action):
        self.prev_actions.append(action)
        
        flag_inertia = False
        if action == 0:
            self.stringency_index = max(0, self.stringency_index - 10)
        elif action == 1:
            self.stringency_index = max(0, self.stringency_index - 5)
        elif action == 2:
            self.stringency_index = max(0, self.stringency_index - 2.5)
        elif action == 3:
            self.stringency_index = max(0, self.stringency_index + 0)
            # reward for inertia
            flag_inertia = True
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
        
        # REMEMBER: to change this definition of the reward in the render as well!!!
        # self.reward = self.normalized_GDP - (2 * self.r_eff)

        reward_inertia = 2 if flag_inertia else 0
        reward_r_eff = 20 if self.r_eff <= 1 else 0

        self.reward = self.normalized_GDP / self.r_eff
        self.store_reward[self.ith_day] = self.reward

        if RL_LEARNING_TYPE == "normal":
            observation = [self.S_proportion, self.I_proportion, 
                        self.R_proportion, self.normalized_GDP, 
                        self.r_eff] + list(self.prev_actions)
            observation = np.array(observation)
        else:
            observation = {}
            observation["stringency"] = np.array(self.prev_actions)
            observation["normalized_gdp"] = np.array(self.store_normalized_gdp)
            observation["r_eff"] = np.array(self.store_r_eff)
            observation["other_stats"] = np.array([self.S_proportion, self.I_proportion, self.R_proportion])

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
        axes[0, 0].plot(self.t, self.df['S']/self.N, 'b', alpha=0.5, lw=2, label='Susceptible (actual)')
        axes[0, 0].plot(self.t, self.df['I']/self.N, 'r', alpha=0.5, lw=2, label='Infected (actual)')
        axes[0, 0].plot(self.t, self.df['R']/self.N, 'g', alpha=0.5, lw=2, label='Recovered (actual)')
        axes[0, 0].plot(self.t, self.store_S/self.N, 'b--', alpha=0.5, lw=2, label='Susceptible (rl)')
        axes[0, 0].plot(self.t, self.store_I/self.N, 'r--', alpha=0.5, lw=2, label='Infected (rl)')
        axes[0, 0].plot(self.t, self.store_R/self.N, 'g--', alpha=0.5, lw=2, label='Recovered (rl)')
        axes[0, 0].set_xlabel('Time /days')
        axes[0, 0].set_ylabel('Percentage of Population')
        axes[0, 0].set_title('SIR Epidemic Trajectory')
        axes[0, 0].tick_params(length=0)
        axes[0, 0].grid(True)
        legend = axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        legend.get_frame().set_alpha(0.5)

        axes[0, 1].plot(self.t, self.store_stringency, 'g', label="Stringency (rl)")
        axes[0, 1].plot(self.t, self.df['stringency_index'], 'b', label="Stringency (actual)")
        axes[0, 1].set_xlabel("Time /days")
        axes[0, 1].set_ylabel("Stringency Index")
        axes[0, 1].set_title("Time vs. Stringency")
        axes[0, 1].set_ylim(bottom=0, top=150)
        axes[0, 1].grid(True)
        axes[0, 1].legend()

        axes[1, 0].plot(self.t, self.store_gdp, 'g', label="GDP (rl)")
        axes[1, 0].plot(self.t, self.df['gdp_normalized'], 'r', label="GDP (actual)")
        axes[1, 0].set_xlabel("Time /days")
        axes[1, 0].set_ylabel("GDP")
        axes[1, 0].set_title("Time vs. GDP")
        axes[1, 0].set_ylim(bottom=0, top=150)
        axes[1, 0].grid(True)
        axes[1, 0].legend()

        axes[1, 1].plot(self.t, self.store_r_eff, 'g', label="R_eff (rl)")
        axes[1, 1].plot(self.t, self.df['r_eff_actual'], 'r', label="R_eff (actual)")
        first_time_r_eff_1 = next((t for t, r_eff in zip(self.t, self.store_r_eff) if r_eff <= 1), None)
        first_time_r_eff_actual_1 = next((t for t, r_eff in zip(self.t, self.df['r_eff_actual']) if r_eff <= 1), None)
        axes[1, 1].set_xlabel("Time /days")
        axes[1, 1].set_ylabel("R_eff")
        axes[1, 1].set_title("Time vs. R_eff")
        axes[1, 1].set_ylim(bottom=0, top=2.0)
        axes[1, 1].grid(True)
        legend = axes[1, 1].legend()
        legend.get_texts()[0].set_text(f'R_eff (rl); R_eff=1 at {first_time_r_eff_1}')
        legend.get_texts()[1].set_text(f'R_eff (actual); R_eff=1 at {first_time_r_eff_actual_1}')

        axes[2, 0].plot(self.t, self.store_reward, 'g', label="Reward (rl)")
        normalized_gdp_actual = (self.df['gdp_normalized'] - MIN_GDP) / (MAX_GDP - MIN_GDP)
        inertia_rewards = [0]
        for i in range(1, len(self.df)):
            diff = self.df['stringency_index'][i] - self.df['stringency_index'][i - 1]
            inertia_reward = 2 if diff == 0 else 0
            inertia_rewards.append(inertia_reward)
        r_eff_rewards = []
        for i in range(0, len(self.df)):
            r_eff = self.df['r_eff_actual'][i]
            r_eff_reward = 20 if r_eff <= 1 else 0
            r_eff_rewards.append(r_eff_reward)
        # axes[2, 0].plot(self.t, normalized_gdp_actual / self.df['r_eff_actual'] + inertia_rewards + r_eff_rewards, 'r', label="Reward (actual)")
        axes[2, 0].plot(self.t, normalized_gdp_actual / self.df['r_eff_actual'], 'r', label="Reward (actual)")
        axes[2, 0].set_xlabel("Time /days")
        axes[2, 0].set_ylabel("Reward")
        axes[2, 0].set_title("Time vs. Reward")
        axes[2, 0].grid(True)
        axes[2, 0].legend()

        # formatted_actual_score = "{:.2f}".format((normalized_gdp_actual / self.df['r_eff_actual'] + inertia_rewards + r_eff_rewards).sum())
        formatted_actual_score = "{:.2f}".format((normalized_gdp_actual / self.df['r_eff_actual']).sum())
        formatted_score = "{:.2f}".format(score)
        axes[2, 1].text(0.5, 0.5, f"Episode Score (rl): {formatted_score}\nEpisode score (actual): {formatted_actual_score}", ha='center', va='center', transform=axes[2, 1].transAxes, fontsize=14)
        axes[2, 1].set_xticks([])
        axes[2, 1].set_yticks([])
        axes[2, 1].spines['top'].set_visible(False)
        axes[2, 1].spines['right'].set_visible(False)
        axes[2, 1].spines['bottom'].set_visible(False)
        axes[2, 1].spines['left'].set_visible(False)
        
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
        
        self.prev_actions = deque(maxlen = TOTAL_DAYS + 1)
        for i in range(TOTAL_DAYS + 1):
            self.prev_actions.append(-1) 
        
        # create the observation
        if RL_LEARNING_TYPE == "normal":
            observation = [self.S_proportion, self.I_proportion, 
                       self.R_proportion, self.normalized_GDP, 
                       self.r_eff] + list(self.prev_actions)
            observation = np.array(observation)
        elif RL_LEARNING_TYPE == "deep":
            observation = {}
            observation["stringency"] = np.array(self.prev_actions)
            observation["normalized_gdp"] = np.array(self.store_normalized_gdp)
            observation["r_eff"] = np.array(self.store_r_eff)
            observation["other_stats"] = np.array([self.S_proportion, self.I_proportion, self.R_proportion])
        info = {}
        return observation, info

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__(observation_space, features_dim=1)

        self.extractors = nn.ModuleDict()
        total_concat_size = 0

        for key, subspace in observation_space.spaces.items():
            if subspace.shape[0] == TOTAL_DAYS + 1:
                self.extractors[key] = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
                total_concat_size += 16
            elif key == "other_stats":  # Flatten for 'other_stats'
                self.extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            if isinstance(extractor, nn.LSTM):
                # LSTM expects input of shape (batch, seq_len, input_size)
                obs = observations[key].unsqueeze(-1)
                out, _ = extractor(obs)
                # We take the final hidden state
                encoded_tensor_list.append(out[:, -1, :])
            else:
                encoded_tensor_list.append(extractor(observations[key]))

        return th.cat(encoded_tensor_list, dim=1)