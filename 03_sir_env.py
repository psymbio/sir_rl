import pandas as pd
import numpy as np
import os
import numpy as np
from scipy.integrate import odeint
import json
import matplotlib.pyplot as plt
from constants import LOCATION_CHOOSEN, OUTPUT_DIR, DATA_CACHE_DIR, STRINGENCY_BASED_GDP, OPTIMAL_VALUES_FILE

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
N = df.loc[min(df.index), ['N']].item()
y0 = df.loc[min(df.index), ['S']].item(), df.loc[min(df.index), ['I']].item(), df.loc[min(df.index), ['R']].item()

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
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "SIR_model_beta_varying_fit_" + LOCATION_CHOOSEN + ".png"))
        plt.show()

SIR_instance = SIR_model(N=N, y0=y0, beta_optimal=optimal_beta, 
                         gamma_optimal=optimal_gamma, 
                         s_weight_optimal=optimal_stringency_weight, df=df)

SIR_instance.plot_against_data()
