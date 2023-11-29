import gymnasium as gym
from gymnasium import spaces
import numpy as np

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
    beta_array = time_varying_beta(beta_optimal, stringency_weight, current_stringency)
    predictions = odeint(deriv, y0, t, args=(N, beta_array, gamma))
    S, I, R = predictions.T
    cost = compute_cost(df['S'], S) + compute_cost(df['I'], I) + compute_cost(df['R'], R)
    return cost

class SIR_model:
    def __init__(self, N, y0, beta_optimal, gamma_optimal, s_weight_optimal, df, stringency_index=None):
        self.N = N
        self.y0 = y0
        self.beta_optimal = beta_optimal
        self.gamma_optimal = gamma_optimal
        self.s_weight_optimal = s_weight_optimal
        self.df = df
        
        self.days_difference = (max(self.df['date']) - min(self.df['date'])).days
        self.t = np.linspace(0, self.days_difference, self.days_difference + 1)
        self.store_S = []
        self.store_I = []
        self.store_R = []
        
        for ith_day in range(0, self.days_difference + 1):
            t = np.linspace(0, ith_day, ith_day + 1)
            stringency_index = df['stringency_index'].iloc[ith_day]
            beta_for_stringency = time_varying_beta(self.beta_optimal, self.s_weight_optimal, stringency_index)
            predictions = odeint(deriv, y0, t, args=(N, beta_for_stringency, self.gamma_optimal))
            S, I, R = predictions.T
            self.store_S.append(S[-1])
            self.store_I.append(I[-1])
            self.store_R.append(R[-1])
            
    def plot_against_data(self):
        plt.figure()
        plt.plot(self.t, self.df['S']/self.N, 'b', alpha=0.5, lw=2, label='Susceptible Data')
        plt.plot(self.t, self.df['I']/self.N, 'r', alpha=0.5, lw=2, label='Infected Data')
        plt.plot(self.t, self.df['R']/self.N, 'g', alpha=0.5, lw=2, label='Recovered Data')

        plt.plot(self.t, self.store_S/self.N, 'b--', alpha=0.5, lw=2, label='Susceptible (Model)')
        plt.plot(self.t, self.store_I/self.N, 'r--', alpha=0.5, lw=2, label='Infected (Model)')
        plt.plot(self.t, self.store_R/self.N, 'g--', alpha=0.5, lw=2, label='Recovered (Model)')

        plt.xlabel('Time /days')
        plt.ylabel('Percentage of Population')
        plt.tick_params(length=0)
        plt.grid(True)
        legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        legend.get_frame().set_alpha(0.5)
        # plt.savefig(os.path.join(OUTPUT_DIR, "SIR_model_beta_varying_fit_" + LOCATION_CHOOSEN + ".png"))
        plt.show()

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
        self.store_gdp[self.ith_day] = fit_line(self.stringency_index)
        
        self.S_proportion = self.store_S[self.ith_day]/self.N
        self.I_proportion = self.store_I[self.ith_day]/self.N
        self.R_proportion = self.store_R[self.ith_day]/self.N
        self.normalized_GDP = (fit_line(self.stringency_index) - MIN_GDP) / (MAX_GDP - MIN_GDP)
        self.r_eff = (beta_for_stringency / self.gamma_optimal) * (self.store_S[self.ith_day] / self.N)
        self.store_r_eff[self.ith_day] = self.r_eff
        
        # self.reward = self.normalized_GDP / self.r_eff
        self.reward = self.normalized_GDP - (4 * self.r_eff)

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
        # print("TERMINATED VALUE", self.terminated)
        return observation, self.reward, self.terminated, self.truncated, info
    
    def render(self, score=0.0):
        self.t = np.linspace(0, TOTAL_DAYS, TOTAL_DAYS + 1)
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
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
        axes[1, 1].set_xlabel("Time /days")
        axes[1, 1].set_ylabel("R_eff")
        axes[1, 1].set_title("Time vs. R_eff")
        axes[1, 1].set_ylim(bottom=0, top=2.0)
        axes[1, 1].grid(True)

        formatted_score = "{:.2f}".format(score)
        fig.suptitle(f"Episode Score: {formatted_score}", y=0.02)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_RL, str(score) + ".png"))
        plt.close()
        # plt.show()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.stringency_index = START_STRINGENCY
        self.N = N
        self.y0 = y0
        self.beta_optimal = beta_optimal
        self.gamma_optimal = gamma_optimal
        self.s_weight_optimal = s_weight_optimal
        self.df = df
        self.days_difference = (max(self.df['date']) - min(self.df['date'])).days
        self.t = np.linspace(0, self.days_difference, self.days_difference + 1)
        self.ith_day = 0
        
        self.store_S = np.zeros(TOTAL_DAYS+1)
        self.store_I = np.zeros(TOTAL_DAYS+1)
        self.store_R = np.zeros(TOTAL_DAYS+1)
        
        self.store_stringency = np.zeros(TOTAL_DAYS+1)
        self.store_gdp = np.zeros(TOTAL_DAYS+1)
        self.store_r_eff = np.zeros(TOTAL_DAYS+1)
        
        self.prev_reward = 0
        self.terminated = False
        self.truncated = False
        
        self.S_proportion = self.y0[0]/self.N
        self.I_proportion = self.y0[1]/self.N
        self.R_proportion = self.y0[2]/self.N
        self.normalized_GDP = (fit_line(self.stringency_index) - MIN_GDP) / (MAX_GDP - MIN_GDP)
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