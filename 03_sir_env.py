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
print(df)
print(f'Total Days: {TOTAL_DAYS}')

START_STRINGENCY = df.loc[min(df.index), ['stringency_index']].item()
N = df.loc[min(df.index), ['N']].item()
y0 = df.loc[min(df.index), ['S']].item(), df.loc[min(df.index), ['I']].item(), df.loc[min(df.index), ['R']].item()

def deriv(y, t, N, beta, gamma, lockdown):
    S, I, R = y
    t = min(int(t), len(lockdown) - 1)  # Ensure t is an integer and within the range of 'lockdown'
    dSdt = -beta * (1 - lockdown[int(t)]) * S * I / N
    dIdt = beta * (1 - lockdown[int(t)]) * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

class SIR_model:
    def __init__(self, df, optimal_beta, optimal_gamma):
        self.df = df
        self.N = self.df.loc[min(self.df.index), ['N']].item()
        self.y0 = self.df.loc[min(self.df.index), ['S']].item(), self.df.loc[min(self.df.index), ['I']].item(), self.df.loc[min(self.df.index), ['R']].item()
        self.days_difference = (max(self.df['date']) - min(self.df['date'])).days
        self.t = np.linspace(0, self.days_difference, self.days_difference + 1)
        self.lockdown = list(df['stringency_index'].values / 100)
        self.optimal_beta = optimal_beta
        self.optimal_gamma = optimal_gamma

        ret = odeint(deriv, self.y0, self.t, args=(self.N, self.optimal_beta, self.optimal_gamma, self.lockdown))
        self.store_S, self.store_I, self.store_R = ret.T

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
        plt.savefig(os.path.join(OUTPUT_DIR, "SIR_env_with_lockdown_" + LOCATION_CHOOSEN + ".png"))
        plt.show()

SIR_instance = SIR_model(df=df, optimal_beta=optimal_beta, optimal_gamma=optimal_gamma)

SIR_instance.plot_against_data()
