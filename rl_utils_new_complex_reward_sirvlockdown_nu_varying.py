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
print(f'Start stringency: {START_STRINGENCY}')
N = df.loc[min(df.index), ['N']].item()
y0 = df.loc[min(df.index), ['S']].item(), df.loc[min(df.index), ['I']].item(), df.loc[min(df.index), ['R']].item()
nu_varying = list(df['nu_varying_with_time'])
N_DISCRETE_ACTIONS = 7

def deriv(y, t, N, beta, gamma, nu_varying, lockdown):
    S, I, R = y
    t = min(int(t), len(lockdown) - 1)  # Ensure t is an integer and within the range of 'lockdown'
    dSdt = -beta * (1 - lockdown[int(t)]) * S * I / N - nu_varying[int(t)] * S
    dIdt = beta * (1 - lockdown[int(t)]) * S * I / N - gamma * I
    dRdt = gamma * I + nu_varying[int(t)] * S
    return dSdt, dIdt, dRdt

# def calculate_reward_weighted(gdp_min_max_normalized_list, r_eff_list):
#     reward_list = []
#     for gdp, r_eff in zip(gdp_min_max_normalized_list, r_eff_list):
#         gdp_reward = ((gdp ** 5)*300) - 80
#         r_eff_reward = ((r_eff**2)*-100) + 100
#         total_reward = gdp_reward + r_eff_reward
#         reward_list.append(total_reward)
#     return reward_list

def calculate_reward_weighted(gdp_min_max_normalized_list, r_eff_list):
    reward_list = []
    for gdp, r_eff in zip(gdp_min_max_normalized_list, r_eff_list):
        if r_eff > 1.5:
            reward_list.append(-20 * r_eff)
        elif r_eff >= 1.25 and r_eff <= 1.5:
            reward_list.append(100 * gdp)
        else:
            reward_list.append(200 * gdp)
    return reward_list

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
        
        diff_action = 0
        if action == 0:
            self.stringency_index = max(0, self.stringency_index - 10)
            diff_action = -10
        elif action == 1:
            self.stringency_index = max(0, self.stringency_index - 5)
            diff_action = -5
        elif action == 2:
            self.stringency_index = max(0, self.stringency_index - 2.5)
            diff_action = -2.5
        elif action == 3:
            self.stringency_index = max(0, self.stringency_index + 0)
            diff_action = 0
        elif action == 4:
            self.stringency_index = min(100, self.stringency_index + 2.5)
            diff_action = 2.5
        elif action == 5:
            self.stringency_index = min(100, self.stringency_index + 5)
            diff_action = 5
        elif action == 6:
            self.stringency_index = min(100, self.stringency_index + 10)
            diff_action = 10
        
        # each action is on ith day
        t_ith_day = np.linspace(0, self.ith_day, self.ith_day + 1)
        self.stringency_index_list.append(self.stringency_index)
        predictions = odeint(deriv, self.y0, t_ith_day, args=(self.N, self.optimal_beta, self.optimal_gamma, np.array(self.nu_varying), np.array(self.stringency_index_list[:TOTAL_DAYS]) / 100))
        S, I, R = predictions.T
        self.store_S[self.ith_day] = S[-1]
        self.store_I[self.ith_day] = I[-1]
        self.store_R[self.ith_day] = R[-1]
        
        self.store_stringency[self.ith_day] = self.stringency_index
        diff = self.store_stringency[self.ith_day] - self.store_stringency[self.ith_day - 1]
        self.store_gdp[self.ith_day] = fit_line_loaded(self.stringency_index)
        
        self.S_proportion = self.store_S[self.ith_day]/self.N
        self.I_proportion = self.store_I[self.ith_day]/self.N
        self.R_proportion = self.store_R[self.ith_day]/self.N
        self.normalized_GDP_min_max_normalized = (fit_line_loaded(self.stringency_index) - MIN_GDP) / (MAX_GDP - MIN_GDP)
        self.r_eff = (self.optimal_beta / self.optimal_gamma) *  (1 - (self.stringency_index/100)) * (self.store_S[self.ith_day] / self.N)
        self.store_r_eff[self.ith_day] = self.r_eff
        self.store_normalized_gdp_min_max_normalized[self.ith_day] = self.normalized_GDP_min_max_normalized
        
        # REMEMBER: to change this definition of the reward in the render as well!!!
        # self.reward = self.normalized_GDP_min_max_normalized - (2 * self.r_eff)

        reward_inertia= abs(diff_action)*-1*8
        reward_I_percentage = -2000 if self.I_proportion >= 0.003 else 50
        
        # gdp_reward = ((self.normalized_GDP_min_max_normalized**5)*300) - 80
        # r_eff_reward = ((self.r_eff**2)*-100) + 100
        # reward_weighted = gdp_reward + r_eff_reward
        if self.r_eff > 1.5:
            reward_weighted = -20 * self.r_eff
        elif self.r_eff >= 1.25 and self.r_eff <= 1.5:
            reward_weighted = 100 * self.normalized_GDP_min_max_normalized
        else:
            reward_weighted = 200 * self.normalized_GDP_min_max_normalized
        
        self.reward = reward_weighted + reward_inertia + reward_I_percentage
        # print(self.ith_day, self.reward)
        self.store_reward[self.ith_day] = self.reward
        if RL_LEARNING_TYPE == "normal":
            observation = [self.S_proportion, self.I_proportion, 
                        self.R_proportion, self.normalized_GDP_min_max_normalized, 
                        self.r_eff] + list(self.prev_actions)
            observation = np.array(observation)
        else:
            observation = {}
            observation["stringency"] = np.array(self.prev_actions)
            observation["normalized_gdp"] = np.array(self.store_normalized_gdp_min_max_normalized)
            observation["r_eff"] = np.array(self.store_r_eff)
            observation["other_stats"] = np.array([self.S_proportion, self.I_proportion, self.R_proportion])

        # after doing the action add to ith_day
        # and then if ith_day is the last day then end the environment
        self.ith_day += 1
        if self.ith_day > TOTAL_DAYS:
            self.terminated = True
        info = {
            "action": action,
            "reward_inertia": reward_inertia,
            # "reward_r_eff": reward_r_eff,
            "reward_I_percentage": reward_I_percentage,
            "reward_weigthed": reward_weighted,
            "reward": self.reward,
            "stringency_index": self.stringency_index
            }
        return observation, self.reward, self.terminated, self.truncated, info
    
    def render(self, score=0.0, stringency=np.array([]), learning=False):
        N = self.df.loc[min(df.index), ["N"]].item()
        y0 = self.df.loc[min(df.index), ["S"]].item(), self.df.loc[min(df.index), ["I"]].item(), self.df.loc[min(df.index), ["R"]].item()
        days_difference = (max(self.df["date"]) - min(self.df["date"])).days
        t = np.linspace(0, days_difference, days_difference + 1)
        nu_varying = list(self.df['nu_varying_with_time'])

        moves_lockdown = stringency / 100
        moves_ret = odeint(deriv, y0, t, args=(N, optimal_beta, optimal_gamma, nu_varying, moves_lockdown))
        moves_S, moves_I, moves_R = moves_ret.T
        
        self.df["S_moves"] = moves_S
        self.df["I_moves"] = moves_I
        self.df["R_moves"] = moves_R
        
        modelling_type = "with_lockdown_with_vaccination_time_varying_nu"
        r0 = self.optimal_beta / self.optimal_gamma * (1 - moves_lockdown)
        self.df["r_eff_moves_" + modelling_type] = r0 * df["S_moves"] / N
        self.df["gdp_normalized_moves"] = fit_line_loaded(stringency)
        self.df["gdp_normalized_moves_min_max_normalized"] = ((fit_line_loaded(stringency) - MIN_GDP) / (MAX_GDP - MIN_GDP))

        self.t = np.linspace(0, TOTAL_DAYS, TOTAL_DAYS + 1)
        
        hospital_capacity = 0.003
        hospital_capacity_punishment = -2000
        hospital_capacity_reward = 20
        I_reward_actual = [hospital_capacity_punishment if I_percentage >= hospital_capacity else hospital_capacity_reward for I_percentage in self.df["I"] / self.df["N"]]
        I_reward_modelled = [hospital_capacity_punishment if I_percentage >= hospital_capacity else hospital_capacity_reward for I_percentage in self.df["I_modelled_" + modelling_type] / self.N]
        I_reward_moves = [hospital_capacity_punishment if I_percentage >= hospital_capacity else hospital_capacity_reward for I_percentage in self.df["I_moves"] / N]

        
        inertia_rewards_actual = np.array([0] + [abs(diff)*8*-1 for diff in (self.df['stringency_index'][i] - self.df['stringency_index'][i - 1] for i in range(1, len(self.df)))])
        # modelled reward for intertia is same as actual
        inertia_rewards_modelled = np.array([0] + [abs(diff)*8*-1 for diff in (self.df['stringency_index'][i] - self.df['stringency_index'][i - 1] for i in range(1, len(self.df)))])
        inertia_rewards_moves = np.array([0] + [abs(diff)*8*-1 for diff in (stringency[i] - stringency[i - 1] for i in range(1, len(stringency)))])

        reward_actual = np.array(calculate_reward_weighted(self.df["gdp_min_max_normalized"], self.df["r_eff_actual_" + modelling_type])) + I_reward_actual + inertia_rewards_actual
        reward_modelled = np.array(calculate_reward_weighted(self.df["gdp_normalized_modelled_min_max_normalized"], self.df["r_eff_modelled_" + modelling_type])) + I_reward_modelled + inertia_rewards_modelled
        reward_moves = np.array(calculate_reward_weighted(self.df["gdp_normalized_moves_min_max_normalized"], self.df["r_eff_moves_" + modelling_type])) + I_reward_moves + inertia_rewards_moves

        # output_path_img = os.path.join(OUTPUT_RL, str(int(reward_moves.sum())))
        output_path_img = os.path.join(OUTPUT_RL, str(score))
        try:
            os.makedirs(output_path_img)
        except:
            print("path exists")

        plt.figure(figsize=(12, 8))
        plt.plot(self.df['date'], self.df['S']/self.df['N'], color="#006EAE", alpha=0.5, lw=2, label='Susceptible (actual)')
        plt.plot(self.df['date'], self.df['I']/self.df['N'], color="#C5373D", alpha=0.5, lw=2, label='Infected (actual)')
        plt.plot(self.df['date'], self.df['R']/self.df['N'], color="#429130", alpha=0.5, lw=2, label='Recovered (actual)')
        plt.plot(self.df['date'], self.df['S_modelled_' + modelling_type]/self.N, color="#006EAE", linestyle=':', alpha=0.5, lw=2, label='Susceptible (modelled)')
        plt.plot(self.df['date'], self.df['I_modelled_' + modelling_type]/self.N, color="#C5373D", linestyle=':', alpha=0.5, lw=2, label='Infected (modelled)')
        plt.plot(self.df['date'], self.df['R_modelled_' + modelling_type]/self.N, color="#429130", linestyle=':', alpha=0.5, lw=2, label='Recovered (modelled)')
        plt.plot(self.df['date'], self.df['S_moves']/self.N, color="#006EAE", linestyle='--', alpha=0.5, lw=2, label='Susceptible (rl)')
        plt.plot(self.df['date'], self.df['I_moves']/self.N, color="#C5373D", linestyle='--', alpha=0.5, lw=2, label='Infected (rl)')
        plt.plot(self.df['date'], self.df['R_moves']/self.N, color="#429130", linestyle='--', alpha=0.5, lw=2, label='Recovered (rl)')
        plt.xlabel('Date')
        plt.ylabel('Percentage of Population')
        plt.title('SIR Epidemic Trajectory')
        plt.tick_params(length=0)
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_path_img, "rl_sir.png"))
        plt.savefig(os.path.join(output_path_img, "rl_sir.eps"))
        plt.close()

        plt.figure(figsize=(12, 8))
        plt.plot(self.df['date'], self.df['I']/self.df['N'], color="#C5373D", label='Infected (actual)')
        plt.plot(self.df['date'], self.df['I_modelled_' + modelling_type]/self.N, color="#006EAE", label='Infected (modelled)')
        plt.plot(self.df['date'], self.df['I_moves']/self.N, color="#429130", label='Infected (rl)')
        plt.xlabel("Date")
        plt.ylabel("Percentage of Infected Population")
        plt.title("SIR Epidemic Trajectory (Infected)")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_path_img, "rl_i.png"))
        plt.savefig(os.path.join(output_path_img, "rl_i.eps"))
        plt.close()

        plt.figure(figsize=(12, 8))
        plt.plot(self.df['date'], self.df['stringency_index'], color="#006EAE", label="Stringency (actual)")
        plt.plot(self.df['date'], stringency, color="#429130", label="Stringency (rl)")
        plt.xlabel("Date")
        plt.ylabel("Stringency Index")
        plt.title("Stringency over Time")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_path_img, "rl_stringency.png"))
        plt.savefig(os.path.join(output_path_img, "rl_stringency.eps"))
        plt.close()

        plt.figure(figsize=(12, 8))
        plt.plot(self.df['date'], self.df['gdp_normalized'], color="#C5373D", label="GDP normalized (actual)")
        plt.plot(self.df['date'], self.df['gdp_normalized_modelled'], color="#006EAE", label="GDP normalized (modelled)")
        plt.plot(self.df['date'], self.df['gdp_normalized_moves'], color="#429130", label="GDP normalized (rl)")
        plt.xlabel("Date")
        plt.ylabel("GDP")
        plt.title("GDP over Time")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_path_img, "rl_gdp.png"))
        plt.savefig(os.path.join(output_path_img, "rl_gdp.eps"))
        plt.close()

        first_time_r_eff_actual_1 = next((t for t, r_eff in zip(self.df['date'], self.df['r_eff_actual_' + modelling_type]) if r_eff <= 1), None)
        first_time_r_eff_modelled_1 = next((t for t, r_eff in zip(self.df['date'], self.df['r_eff_modelled_' + modelling_type]) if r_eff <= 1), None)
        first_time_r_eff_1 = next((t for t, r_eff in zip(self.df['date'], self.df["r_eff_moves_" + modelling_type]) if r_eff <= 1), None)

        plt.figure(figsize=(12, 8))
        plt.plot(self.df['date'], self.df['r_eff_actual_' + modelling_type], color="#C5373D", label="R_eff (actual)")
        plt.plot(self.df['date'], self.df['r_eff_modelled_' + modelling_type], color="#006EAE", label="R_eff (modelled)")
        plt.plot(self.df['date'], self.df["r_eff_moves_" + modelling_type], color="#429130", label="R_eff (rl)")
        plt.xlabel("Date")
        plt.ylabel("R_eff")
        plt.title("R_eff over Time")
        plt.grid(True)
        legend = plt.legend()
        legend.get_texts()[0].set_text(f'R_eff (actual); R_eff=1 at {first_time_r_eff_actual_1}')
        legend.get_texts()[1].set_text(f'R_eff (modelled); R_eff=1 at {first_time_r_eff_modelled_1}')
        legend.get_texts()[2].set_text(f'R_eff (rl); R_eff=1 at {first_time_r_eff_1}')
        plt.savefig(os.path.join(output_path_img, "rl_r_eff.png"))
        plt.savefig(os.path.join(output_path_img, "rl_r_eff.eps"))
        plt.close()

        # print("len df", len(calculate_reward_weighted(self.df["gdp_min_max_normalized"], self.df["r_eff_actual_" + modelling_type])))

        plt.figure(figsize=(12, 8))
        plt.plot(self.df['date'], reward_actual, color="#C5373D", label=f"Reward (actual) Total: {reward_actual.sum():.2f}")
        plt.plot(self.df['date'], reward_modelled, color="#006EAE", label=f"Reward (modelled) Total: {reward_modelled.sum():.2f}")
        plt.plot(self.df['date'], reward_moves, color="#429130", label=f"Reward (rl) Total: {reward_moves.sum():.2f}")
        plt.xlabel("Time /days")
        plt.ylabel("Reward")
        plt.title("Reward over Time")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_path_img, "rl_reward.png"))
        plt.savefig(os.path.join(output_path_img, "rl_reward.eps"))
        plt.close()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.stringency_index = START_STRINGENCY
        self.N = N
        self.y0 = y0
        self.optimal_beta = optimal_beta
        self.optimal_gamma = optimal_gamma
        self.df = df
        self.days_difference = (max(self.df['date']) - min(self.df['date'])).days
        self.t = np.linspace(0, self.days_difference, self.days_difference + 1)
        self.ith_day = 0
        self.nu_varying = self.df['nu_varying_with_time']
        
        self.store_S = np.zeros(TOTAL_DAYS + 1)
        self.store_I = np.zeros(TOTAL_DAYS + 1)
        self.store_R = np.zeros(TOTAL_DAYS + 1)
        
        self.store_stringency = np.zeros(TOTAL_DAYS + 1)
        self.stringency_index_list = [START_STRINGENCY]
        self.store_gdp = np.zeros(TOTAL_DAYS + 1)
        self.store_normalized_gdp_min_max_normalized = np.zeros(TOTAL_DAYS + 1)
        self.store_r_eff = np.zeros(TOTAL_DAYS + 1)
        self.store_reward = np.zeros(TOTAL_DAYS + 1)
        
        # self.prev_reward = 0
        self.terminated = False
        self.truncated = False
        
        self.S_proportion = self.y0[0]/self.N
        self.I_proportion = self.y0[1]/self.N
        self.R_proportion = self.y0[2]/self.N
        self.normalized_GDP_min_max_normalized = (fit_line_loaded(self.stringency_index) - MIN_GDP) / (MAX_GDP - MIN_GDP)
        self.r_eff = (self.optimal_beta / self.optimal_gamma) * (1 - (self.stringency_index/100)) * (self.store_S[self.ith_day] / self.N)
        
        self.prev_actions = deque(maxlen = TOTAL_DAYS + 1)
        for i in range(TOTAL_DAYS + 1):
            self.prev_actions.append(-1) 
        
        # create the observation
        if RL_LEARNING_TYPE == "normal":
            observation = [self.S_proportion, self.I_proportion, 
                       self.R_proportion, self.normalized_GDP_min_max_normalized, 
                       self.r_eff] + list(self.prev_actions)
            observation = np.array(observation)
        elif RL_LEARNING_TYPE == "deep":
            observation = {}
            observation["stringency"] = np.array(self.prev_actions)
            observation["normalized_gdp"] = np.array(self.store_normalized_gdp_min_max_normalized)
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