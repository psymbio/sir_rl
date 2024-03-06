import numpy as np

from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.stats import poisson
from scipy import spatial

from scipy import stats

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation
from matplotlib.ticker import FuncFormatter

import pandas as pd
import os
import seaborn as sns
import json
from datetime import date, timedelta
import math
np.random.seed(seed=42)

# {'dejavusans':'ok', 'dejavuserif':'not_tried', 'cm':'no', 'stix':'no', 'stixsans':'no', 'custom':'best'}
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# plt.rcParams['text.usetex'] = True
# plt.rcParams["font.family"] = "Times New Roman"
# sns.set_style("whitegrid", rc={'font.family': 'Times New Roman', 'font.size': 16})
plt.rc('legend',fontsize=16)
plt.rcParams.update({'font.size': 16, 'axes.labelsize': 14, 'axes.titlesize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14})
sns.set_style("whitegrid", rc={'font.size': 16})

from constants import LOCATION_CHOOSEN, LOCATION_CHOOSEN_2, OUTPUT_DIR, DATA_CACHE_DIR, OPTIMAL_VALUES_FILE, STRINGENCY_BASED_GDP



data_path = os.path.join(DATA_CACHE_DIR, LOCATION_CHOOSEN + "_merged_data.csv")
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    df = pd.read_excel("owid-covid-data.xlsx")
    df = df.loc[df['iso_code'] == LOCATION_CHOOSEN]
    df.to_csv(data_path, index=False)



start_date = date(2020, 5, 1) # -> works, but not for GDP
# start_date = date(2020, 2, 10) -> doesn't work
# start_date = date(2020, 3, 1) -> doesn't work for SIR with lockdown
# start_date = date(2020, 3, 15) -> doesn't work for SIR with lockdown
# start_date = date(2020, 4, 1) -> doesn't work for SIR with lockdown
# start_date = date(2020, 4, 15) -> doesn't work for SIR with lockdown
# start_date = date(2020, 6, 1) -> works but is after 2020-05-01
end_date = date(2022, 11, 1)
# end_date = date(2023, 7, 1)
delta = timedelta(days=1)

worldometers_dates = []
worldometers_total_cases = []
worldometers_total_recovered = []
# worldometers_active_cases = []

while start_date <= end_date:
    date_str = start_date.strftime("%Y%m%d")
    data_path_worldometer = os.path.join(DATA_CACHE_DIR, "worldometer", date_str + ".csv")
    if os.path.exists(data_path_worldometer):
        worldometer_df = pd.read_csv(data_path_worldometer)
        worldometer_df.columns = ['country' if col.startswith('Country') else col for col in worldometer_df.columns]
        worldometer_df.columns = worldometer_df.columns.str.lower().str.replace(' ', '')
        date_data = ""
        total_cases = 0
        total_recovered = 0
        worldometer_df_specified_location = worldometer_df.loc[(worldometer_df["country"].str.lower() == LOCATION_CHOOSEN_2.lower())]
        date_data = date_str[:4] + "-" + date_str[4:6] + "-" + date_str[6:]
        total_cases = 0.0 if math.isnan(worldometer_df_specified_location["totalcases"].item()) else worldometer_df_specified_location["totalcases"].item()
        # active_cases = 0.0 if math.isnan(worldometer_df_specified_location["activecases"].item()) else worldometer_df_specified_location["activecases"].item()
        total_recovered = 0.0 if math.isnan(worldometer_df_specified_location["totalrecovered"].item()) else worldometer_df_specified_location["totalrecovered"].item()
        # print(date_data, total_cases, total_recovered, type(total_recovered), math.isnan(total_recovered))
        worldometers_dates.append(date_data)
        worldometers_total_cases.append(total_cases)
        worldometers_total_recovered.append(total_recovered)
        # worldometers_active_cases.append(active_cases)
    else:
        print(data_path_worldometer, "DOES NOT EXIST -- COLLECT DATA MANUALLY")
    start_date += delta



worldometer_df = pd.DataFrame({"date": worldometers_dates, "total_cases_worldometer": worldometers_total_cases, "total_recovered_worldometer": worldometers_total_recovered, 
                               # "active_cases_worldometer": worldometers_active_cases
                              })
result = pd.merge(df, worldometer_df, on="date")
df = result

df['date'] = pd.to_datetime(df['date'])



df



# deaths are considered recovered: https://www.kaggle.com/code/lisphilar/covid-19-data-with-sir-model/notebook?scriptVersionId=28560520
# https://lisphilar.github.io/covid19-sir/02_data_engineering.html#1.-Data-cleaning

# df['N'] = df['population']
# df['S'] = df['population'] - df['total_cases_worldometer'] - df['people_fully_vaccinated']
# df['I'] = df['total_cases_worldometer'] - df['total_recovered_worldometer'] - df['total_deaths']
# df['R'] = df['total_recovered_worldometer'] + df['people_fully_vaccinated'] + df['total_deaths']

# without deaths model
df['N'] = df['population']
df['S'] = df['population'] - df['total_cases_worldometer'] - df['people_fully_vaccinated'] - df['total_recovered_worldometer']
df['I'] = df['total_cases_worldometer'] - df['total_recovered_worldometer']
df['R'] = df['total_recovered_worldometer'] + df['people_fully_vaccinated']



plt.plot( df['people_fully_vaccinated'].diff())
print(min(df['people_fully_vaccinated'].diff()[290:]))



plt.figure(figsize=(12, 8))
plt.plot(df['date'], df['S'], color="#006EAE", label='Susceptible')
plt.plot(df['date'], df['I'], color="#C5373D", label='Infected')
plt.plot(df['date'], df['R'], color="#429130", label='Recovered')
plt.title('COVID-19 Metrics Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "metrics_over_time_" + LOCATION_CHOOSEN + ".pdf"))
# plt.show()



plt.figure(figsize=(12, 8))
plt.plot(df['date'], df['I'], label='I')
plt.title('COVID-19 Metrics Over Time')
plt.xlabel('Date')
plt.ylabel('Count')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "infected_over_time_" + LOCATION_CHOOSEN + ".pdf"))
# plt.show()


# ## SIR Model


# def compute_cost(data, predictions):
#     # mse
#     return np.square(data - predictions).mean()

# def compute_cost(data, predictions):
#     # mae
#     return np.abs(data - predictions).mean()

# def compute_cost(data, predictions):
#     # Relative Root Mean Squared Error
#     residual = data - predictions
#     root_mean_squared_error = np.sqrt(np.mean(np.square(residual)))
#     mean_data = np.mean(data)
#     return root_mean_squared_error / mean_data

def compute_cost(data, predictions, delta=1.0):
    # Huber loss
    residual = np.abs(data - predictions)
    condition = residual < delta
    squared_loss = 0.5 * np.square(residual)
    linear_loss = delta * (residual - 0.5 * delta)
    return np.where(condition, squared_loss, linear_loss).mean()

def descriptive_statistics(data, method=2):
    mean = np.mean(data)
    median = np.median(data)
    mode_result = stats.mode(data)
    
    mode = 0
    # mode = mode_result.mode[0] if len(mode_result.mode) == 1 else mode_result.mode  # Handling multimodal data
    if isinstance(mode_result.mode, np.float64):
        mode = mode_result.mode
    else:
        mode = mode_result.mode[0] if len(mode_result.mode) == 1 else mode_result.mode
    std_dev = np.std(data)
    min_data = np.min(data)
    max_data = np.max(data)
    data_range = max_data - min_data
    
    if method == 1:
        desc_stat_string = f"""
    ?overbar{{R_0}} = {mean:.3f} \quad ?text{{(Mean)}} ;
    ?widetilde{{R_0}} = {median:.3f} \quad ?text{{(Median)}} ;
    ?text{{Mode}}(R_0) = {mode:.3f} \quad ?text{{(Mode)}} ;
    ?sigma_{{R_0}} = {std_dev:.3f} \quad ?text{{(Standard Deviation)}} ;
    R_0 \in [{min_data:.3f}, {max_data:.3f}] \quad ?text{{(Range)}}
    """
    else:
        desc_stat_string = f"""
    ?overbar{{R_0}} = {mean:.3f} ;
    ?widetilde{{R_0}} = {median:.3f} ;
    ?text{{Mode}}(R_0) = {mode:.3f} ;
    ?sigma_{{R_0}} = {std_dev:.3f} ;
    R_0 \in [{min_data:.3f}, {max_data:.3f}]
    """
    desc_stat_string = desc_stat_string.replace(";", "\\\\")
    desc_stat_string = desc_stat_string.replace("?", "\\")
    print(desc_stat_string)
    # return mean, median, mode, std_dev, data_range



def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def integrate_system(params, y0, t, N):
    beta, gamma = params
    result = odeint(deriv, y0, t, args=(N, beta, gamma))
    return result

def objective_function(params, y0, t, N):
    predictions = integrate_system(params, y0, t, N)
    S, I, R = predictions.T
    cost = compute_cost(df['S'], S) + compute_cost(df['I'], I) +  compute_cost(df['R'], R)
    return cost



N = df.loc[min(df.index), ['N']].item()
y0 = df.loc[min(df.index), ['S']].item(), df.loc[min(df.index), ['I']].item() + 1, df.loc[min(df.index), ['R']].item()
initial_guess_for_beta, initial_guess_for_gamma = 0.4, 1/15 
days_difference = (max(df['date']) - min(df['date'])).days
t = np.linspace(0, days_difference, days_difference + 1)



initial_guesses = [initial_guess_for_beta, initial_guess_for_gamma]
result = minimize(
    objective_function,
    initial_guesses,
    args=(y0, t, N),
    method='Nelder-Mead',
)
optimal_beta, optimal_gamma = result.x
print(f"optimal_beta: {optimal_beta:.3f} optimal_gamma: {optimal_gamma:.3f}")
print(f"optimal_beta/optimal_gamma: {optimal_beta/optimal_gamma:.3f}")

ret = odeint(deriv, y0, t, args=(N, optimal_beta, optimal_gamma))
S, I, R = ret.T

SIR_cost = compute_cost(df['S'], S) + compute_cost(df['I'], I) + compute_cost(df['R'], R)
print(f"cost: {SIR_cost:.3f}")

plt.figure(figsize=(12, 8))
plt.plot(df['date'], df['S']/df['N']*100, color="#006EAE", label='Susceptible (Data)')
plt.plot(df['date'], df['I']/df['N']*100, color="#C5373D", label='Infected (Data)')
plt.plot(df['date'], df['R']/df['N']*100, color="#429130", label='Recovered (Data)')

plt.plot(df['date'], S/N*100, color="#006EAE", linestyle="--", label='Susceptible (Model)')
plt.plot(df['date'], I/N*100, color="#C5373D", linestyle="--", label='Infected (Model)')
plt.plot(df['date'], R/N*100, color="#429130", linestyle="--", label='Recovered (Model)')

exponent = 4
loss_scaled = SIR_cost * (10 ** -exponent)
plt.plot([], [], ' ', label=rf"""Loss = ${loss_scaled:.3f} \times 10^{exponent}$""")

plt.xlabel('Date')
plt.ylabel('Percentage of Population')
plt.tick_params(length=0)
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "SIR_model_" + LOCATION_CHOOSEN + ".pdf"), bbox_inches="tight")
# plt.show()



SIR_cost_I_cost = compute_cost(df['I'], I)
print(f"cost: {SIR_cost_I_cost:.3f}")

plt.figure(figsize=(12, 8))
plt.plot(df['date'], df['I']/df['N']*100, color="#C5373D", label='Infected (Data)')
plt.plot(df['date'], I/N*100, color="#C5373D", linestyle="--", label='Infected (Model)')

exponent = 4
loss_scaled = SIR_cost_I_cost * (10 ** -exponent)
plt.plot([], [], color="none", label=rf"""Loss = ${loss_scaled:.3f} \times 10^{exponent}$""")

plt.xlabel('Date')
plt.ylabel('Percentage of Population')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "SIR_model_infections_" + LOCATION_CHOOSEN + ".pdf"), bbox_inches="tight")
# plt.show()



r0 = optimal_beta/optimal_gamma

df["S_modelled"] = S
df["I_modelled"] = I
df["R_modelled"] = R
df["r_eff_modelled"] = r0 * df["S_modelled"]/N
df["r_eff_actual"] = r0 * df["S"]/df["N"]



r_eff_cosine_similarity = 1 - spatial.distance.cosine(df["r_eff_actual"], df["r_eff_modelled"])

plt.figure(figsize=(12, 8))
plt.plot(df['date'], df["r_eff_actual"], color="#0096A0", label="$R_e$ (Model)")
plt.plot(df['date'], df["r_eff_modelled"], color="#0096A0", linestyle="--", label="$R_e$ (Data)")
plt.plot([], [], color="none", label=f"Cosine Similarity of $R_e$ (Model)\nand $R_e$ (Data) = {r_eff_cosine_similarity:.3f}")

plt.xlabel('Date')
plt.ylabel('$R_e$')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "SIR_model_reff_" + LOCATION_CHOOSEN + ".pdf"), bbox_inches="tight")
# plt.show()


# ## SIR Model with Lockdown


def deriv(y, t, N, beta, gamma, lockdown):
    S, I, R = y
    t = min(int(t), len(lockdown) - 1)
    dSdt = -beta * (1 - lockdown[int(t)]) * S * I / N
    dIdt = beta * (1 - lockdown[int(t)]) * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def integrate_system(params, y0, t, N, lockdown):
    beta, gamma = params
    result = odeint(deriv, y0, t, args=(N, beta, gamma, lockdown))
    return result

def objective_function(params, y0, t, N, lockdown):
    predictions = integrate_system(params, y0, t, N, lockdown)
    S, I, R = predictions.T
    cost = (compute_cost(df['S'], S) + compute_cost(df['I'], I) + compute_cost(df['R'], R))
    return cost



N = df.loc[min(df.index), ['N']].item()
y0 = df.loc[min(df.index), ['S']].item(), df.loc[min(df.index), ['I']].item(), df.loc[min(df.index), ['R']].item()
initial_guess_for_beta, initial_guess_for_gamma = 0.2, 1./10 
days_difference = (max(df['date']) - min(df['date'])).days
t = np.linspace(0, days_difference, days_difference + 1)



initial_guesses = [initial_guess_for_beta, initial_guess_for_gamma]
lockdown = list(df['stringency_index'].values / 100)
result = minimize(
    objective_function,
    initial_guesses,
    args=(y0, t, N, lockdown),
    method='Nelder-Mead',
)
optimal_beta, optimal_gamma = result.x
print(f"optimal_beta: {optimal_beta:.3f} optimal_gamma: {optimal_gamma:.3f}")
print(f"optimal_beta/optimal_gamma: {optimal_beta/optimal_gamma:.3f}")

ret = odeint(deriv, y0, t, args=(N, optimal_beta, optimal_gamma, lockdown))
S, I, R = ret.T

SIR_with_lockdown_cost = compute_cost(df['S'], S) + compute_cost(df['I'], I) + compute_cost(df['R'], R)
print(f"cost: {SIR_with_lockdown_cost:.3f}")

plt.figure(figsize=(12, 8))
plt.plot(df['date'], df['S']/df['N']*100, color="#006EAE", label='Susceptible (Data)')
plt.plot(df['date'], df['I']/df['N']*100, color="#C5373D", label='Infected (Data)')
plt.plot(df['date'], df['R']/df['N']*100, color="#429130", label='Recovered (Data)')

plt.plot(df['date'], S/N*100, color="#006EAE", linestyle="--", label='Susceptible (Model)')
plt.plot(df['date'], I/N*100, color="#C5373D", linestyle="--", label='Infected (Model)')
plt.plot(df['date'], R/N*100, color="#429130", linestyle="--", label='Recovered (Model)')

exponent = 4
loss_scaled = SIR_with_lockdown_cost * (10 ** -exponent)
plt.plot([], [], color="none", label=rf"""Loss = ${loss_scaled:.3f} \times 10^{exponent}$""")

plt.xlabel('Date')
plt.ylabel('Percentage of Population')
plt.tick_params(length=0)
plt.grid(True)
# legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# legend.get_frame().set_alpha(0.5)
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "SIR_model_with_lockdown_" + LOCATION_CHOOSEN + ".pdf"), bbox_inches="tight")
# plt.show()

plt.figure(figsize=(12, 8))
plt.plot(t, df['stringency_index'], 'c')
plt.xlabel('Time (days)')
plt.ylabel('Stringency Index')
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "stringency_varying_with_time_" + LOCATION_CHOOSEN + ".pdf"), bbox_inches="tight")
# plt.show()



SIR_with_lockdown_cost_I_cost = compute_cost(df['I'], I)
print(f"cost: {SIR_with_lockdown_cost_I_cost:.3f}")

plt.figure(figsize=(12, 8))
plt.plot(df['date'], df['I']/df['N']*100, color="#C5373D", label='Infected (Data)')
plt.plot(df['date'], I/N*100, color="#C5373D", linestyle="--", label='Infected (Model)')

exponent = 4
loss_scaled = SIR_with_lockdown_cost_I_cost * (10 ** -exponent)
plt.plot([], [], color="none", label=rf"""Loss = ${loss_scaled:.3f} \times 10^{exponent}$""")

plt.xlabel('Date')
plt.ylabel('Percentage of Population')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "SIR_model_with_lockdown_infections_" + LOCATION_CHOOSEN + ".pdf"), bbox_inches="tight")
# plt.show()



r0 = optimal_beta/optimal_gamma * (1 - np.array(lockdown))

df["S_modelled_with_lockdown"] = S
df["I_modelled_with_lockdown"] = I
df["R_modelled_with_lockdown"] = R

df["r_eff_modelled_with_lockdown"] = r0 * np.array((df["S_modelled_with_lockdown"]/N))
df["r_eff_actual_with_lockdown"] = r0 * np.array((df["S"]/df["N"]))

plt.figure(figsize=(12, 8))
plt.plot(df['date'], r0, color="#734E3D")
plt.xlabel('Date')
plt.ylabel('$R_0$')
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "SIR_model_with_lockdown_r0_" + LOCATION_CHOOSEN + ".pdf"), bbox_inches="tight")
# plt.show()

descriptive_statistics(r0, method=1)



r_eff_cosine_similarity = 1 - spatial.distance.cosine(df["r_eff_actual_with_lockdown"], df["r_eff_modelled_with_lockdown"])

plt.figure(figsize=(12, 8))
plt.plot(df['date'], df["r_eff_actual_with_lockdown"], color="#0096A0", label="$R_e$ (Model)")
plt.plot(df['date'], df["r_eff_modelled_with_lockdown"], color="#0096A0", linestyle="--", label="$R_e$ (Data)")
plt.plot([], [], color="none", label=f"Cosine Similarity of $R_e$ (Model)\nand $R_e$ (Data) = {r_eff_cosine_similarity:.3f}")

plt.xlabel('Date')
plt.ylabel('$R_e$')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "SIR_model_with_lockdown_reff_" + LOCATION_CHOOSEN + ".pdf"), bbox_inches="tight")
# plt.show()



plt.figure(figsize=(12, 8))
plt.plot(df["r_eff_modelled_with_lockdown"], label="modelled")
plt.plot(df["r_eff_actual_with_lockdown"], label="actual")
plt.plot(t, df['stringency_index']/50, 'c')
plt.legend()


# ## SIR Model with Lockdown and Vaccination


def deriv(y, t, N, beta, gamma, nu, lockdown):
    S, I, R = y
    t = min(int(t), len(lockdown) - 1)  # Ensure t is an integer and within the range of 'lockdown'
    dSdt = -beta * (1 - lockdown[int(t)]) * S * I / N - nu * S
    dIdt = beta * (1 - lockdown[int(t)]) * S * I / N - gamma * I
    dRdt = gamma * I + nu * S
    return dSdt, dIdt, dRdt

def integrate_system(params, y0, t, N, lockdown):
    beta, gamma, nu = params
    result = odeint(deriv, y0, t, args=(N, beta, gamma, nu, lockdown))
    return result

def objective_function(params, y0, t, N, lockdown):
    predictions = integrate_system(params, y0, t, N, lockdown)
    S, I, R = predictions.T
    cost = (compute_cost(df['S'], S) + compute_cost(df['I'], I) + compute_cost(df['R'], R))
    return cost



N = df.loc[min(df.index), ['N']].item()
y0 = df.loc[min(df.index), ['S']].item(), df.loc[min(df.index), ['I']].item(), df.loc[min(df.index), ['R']].item()
# initial_guess_for_beta, initial_guess_for_gamma, initial_guess_for_nu = 0.2, 1./10, 0.0001
# initial_guess_for_beta, initial_guess_for_gamma, initial_guess_for_nu = 0.2, 1./10, 0.000001
initial_guess_for_beta, initial_guess_for_gamma, initial_guess_for_nu = 0.2, 1./10, 0.000001
days_difference = (max(df['date']) - min(df['date'])).days
t = np.linspace(0, days_difference, days_difference + 1)



initial_guesses = [initial_guess_for_beta, initial_guess_for_gamma, initial_guess_for_nu]
lockdown = list(df['stringency_index'].values / 100)
result = minimize(
    objective_function,
    initial_guesses,
    args=(y0, t, N, lockdown),
    method='Nelder-Mead',
)
optimal_beta, optimal_gamma, optimal_nu = result.x
print(f"optimal_beta: {optimal_beta:.3f} optimal_gamma: {optimal_gamma:.3f} optimal_nu: {optimal_nu}")
print(f"optimal_beta/optimal_gamma: {optimal_beta/optimal_gamma:.3f}")

ret = odeint(deriv, y0, t, args=(N, optimal_beta, optimal_gamma, optimal_nu, lockdown))
S, I, R = ret.T

SIRV_with_lockdown_cost = compute_cost(df['S'], S) + compute_cost(df['I'], I) + compute_cost(df['R'], R)
print(f"cost: {SIRV_with_lockdown_cost:.3f}")

plt.figure(figsize=(12, 8))
plt.plot(df['date'], df['S']/df['N']*100, color="#006EAE", label='Susceptible (Data)')
plt.plot(df['date'], df['I']/df['N']*100, color="#C5373D", label='Infected (Data)')
plt.plot(df['date'], df['R']/df['N']*100, color="#429130", label='Recovered (Data)')

plt.plot(df['date'], S/N*100, color="#006EAE", linestyle="--", label='Susceptible (Model)')
plt.plot(df['date'], I/N*100, color="#C5373D", linestyle="--", label='Infected (Model)')
plt.plot(df['date'], R/N*100, color="#429130", linestyle="--", label='Recovered (Model)')

exponent = 4
loss_scaled = SIRV_with_lockdown_cost * (10 ** -exponent)
plt.plot([], [], color="none", label=rf"""Loss = ${loss_scaled:.3f} \times 10^{exponent}$""")

plt.xlabel('Date')
plt.ylabel('Percentage of Population')
plt.tick_params(length=0)
plt.grid(True)
# legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# legend.get_frame().set_alpha(0.5)
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "SIR_model_with_lockdown_with_vaccination_" + LOCATION_CHOOSEN + ".pdf"), bbox_inches="tight")
# plt.show()

plt.figure(figsize=(12, 8))
plt.plot(df['date'], df['S']/df['N']*100, color="#006EAE", label='Susceptible (Data)')
plt.plot(df['date'], df['I']/df['N']*100, color="#C5373D", label='Infected (Data)')
plt.plot(df['date'], df['R']/df['N']*100, color="#429130", label='Recovered (Data)')

plt.plot(df['date'], S/N*100, color="#006EAE", linestyle="--", label='Susceptible (Model)')
plt.plot(df['date'], I/N*100, color="#C5373D", linestyle="--", label='Infected (Model)')
plt.plot(df['date'], R/N*100, color="#429130", linestyle="--", label='Recovered (Model)')

plt.plot(df['date'], df['stringency_index'], color='#0096A0', label=r"Normalized Stringency ($s(t)/100$)")

plt.xlabel('Date')
plt.ylabel('Percentage of Population')
plt.tick_params(length=0)
plt.grid(True)
legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
legend.get_frame().set_alpha(0.5)
plt.savefig(os.path.join(OUTPUT_DIR, "SIR_model_with_lockdown_with_vaccination_with_stringency_" + LOCATION_CHOOSEN + ".pdf"), bbox_inches="tight")
# plt.show()

plt.figure(figsize=(12, 8))
plt.plot(df['date'], df['stringency_index'], color='#0096A0')
plt.xlabel('Date')
plt.ylabel('Stringency Index')
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "stringency_varying_with_time_" + LOCATION_CHOOSEN + ".pdf"), bbox_inches="tight")
# plt.show()



SIRV_with_lockdown_cost_I_cost = compute_cost(df['I'], I)
print(f"cost: {SIRV_with_lockdown_cost_I_cost:.3f}")

plt.figure(figsize=(12, 8))
plt.plot(df['date'], df['I']/N*100, color="#C5373D", label='Infected (Data)')
plt.plot(df['date'], I/N*100, color="#C5373D", linestyle="--", label='Infected (Model)')

exponent = 4
loss_scaled = SIRV_with_lockdown_cost_I_cost * (10 ** -exponent)
plt.plot([], [], color="none", label=rf"""Loss = ${loss_scaled:.3f} \times 10^{exponent}$""")

plt.xlabel('Date')
plt.ylabel('Percentage of Population')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "SIR_model_with_lockdown_with_vaccination_infections_" + LOCATION_CHOOSEN + ".pdf"), bbox_inches="tight")
# plt.show()



r0 = optimal_beta/optimal_gamma * (1 - np.array(lockdown))

df["S_modelled_with_lockdown_with_vaccination"] = S
df["I_modelled_with_lockdown_with_vaccination"] = I
df["R_modelled_with_lockdown_with_vaccination"] = R

df["r_eff_modelled_with_lockdown_with_vaccination"] = r0 * np.array((df["S_modelled_with_lockdown_with_vaccination"]/N))
df["r_eff_actual_with_lockdown_with_vaccination"] = r0 * np.array((df["S"]/df["N"]))

plt.figure(figsize=(12, 8))
plt.plot(df['date'], r0, color="#734E3D")
plt.xlabel('Date')
plt.ylabel('$R_0$')
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "SIR_model_with_lockdown_with_vaccination_r0_" + LOCATION_CHOOSEN + ".pdf"), bbox_inches="tight")
# plt.show()

descriptive_statistics(r0, method=2)



r_eff_cosine_similarity = 1 - spatial.distance.cosine(df["r_eff_actual_with_lockdown_with_vaccination"], df["r_eff_modelled_with_lockdown_with_vaccination"])

plt.figure(figsize=(12, 8))
plt.plot(df['date'], df["r_eff_actual_with_lockdown_with_vaccination"], color="#0096A0", label="$R_e$ (Model)")
plt.plot(df['date'], df["r_eff_modelled_with_lockdown_with_vaccination"], color="#0096A0", linestyle="--", label="$R_e$ (Data)")
plt.plot([], [], color="none", label=f"Cosine Similarity of $R_e$ (Model)\nand $R_e$ (Data) = {r_eff_cosine_similarity:.3f}")

plt.xlabel('Date')
plt.ylabel('$R_e$')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "SIR_model_with_lockdown_with_vaccination_reff_" + LOCATION_CHOOSEN + ".pdf"), bbox_inches="tight")
# plt.show()


# ## Vaccination dependent on Time - Optimizing Window Length


window_lengths = list(range(5, 55, 5))
print(window_lengths)

cost_with_different_windows = {}

def deriv_before(y, t, N, nu, lockdown):
    S, I, R = y
    t = min(int(t), len(lockdown) - 1)
    dSdt = -optimal_beta * (1 - lockdown[int(t)]) * S * I / N - nu * S
    dIdt = optimal_beta * (1 - lockdown[int(t)]) * S * I / N - optimal_gamma * I
    dRdt = optimal_gamma * I + nu * S
    return dSdt[0], dIdt, dRdt[0]

def integrate_system_before(params, y0, t, N, lockdown):
    nu = params
    result = odeint(deriv_before, y0, t, args=(N, nu, lockdown), hmax=1.0)
    return result

def objective_function_before(params, y0, t, N, lockdown, days_window):
    predictions = integrate_system_before(params, y0, t, N, lockdown)
    S, I, R = predictions.T
    cost = (compute_cost(df['S'][days_window-window_length:days_window], S) + 
            compute_cost(df['I'][days_window-window_length:days_window], I) + 
            compute_cost(df['R'][days_window-window_length:days_window], R))
    return cost

def deriv_after(y, t, N, beta, gamma, nu_varying, lockdown):
    S, I, R = y
    t = min(int(t), len(lockdown) - 1)  # Ensure t is an integer and within the range of 'lockdown'
    dSdt = -beta * (1 - lockdown[int(t)]) * S * I / N - nu_varying[int(t)] * S
    dIdt = beta * (1 - lockdown[int(t)]) * S * I / N - gamma * I
    dRdt = gamma * I + nu_varying[int(t)] * S
    return dSdt, dIdt, dRdt

def integrate_system_after(params, y0, t, N, nu_varying, lockdown):
    beta, gamma = params
    result = odeint(deriv_after, y0, t, args=(N, beta, gamma, nu_varying, lockdown))
    return result

def objective_function_after(params, y0, t, N, nu_varying, lockdown):
    predictions = integrate_system_after(params, y0, t, N, nu_varying, lockdown)
    S, I, R = predictions.T
    cost = (compute_cost(df['S'], S) + compute_cost(df['I'], I) + compute_cost(df['R'], R))
    return cost
        
for window_length in window_lengths:
    N = df.loc[min(df.index), ['N']].item()
    y0 = df.loc[min(df.index), ['S']].item(), df.loc[min(df.index), ['I']].item(), df.loc[min(df.index), ['R']].item()
    initial_guess_for_beta, initial_guess_for_gamma, initial_guess_for_nu = 0.2, 1./10, 0.001
    
    nu_changing_with_time = []
    time_points_for_nu = []
    for days_window in range(window_length, days_difference, window_length):
        lockdown = list(df['stringency_index'].values / 100)[days_window - window_length:days_window]
        t = np.linspace(days_window - window_length, days_window, window_length)
        initial_guesses = [initial_guess_for_nu]
        result = minimize(
            objective_function_before,
            initial_guesses,
            args=(y0, t, N, lockdown, days_window),
            method='Nelder-Mead',
        )
        optimal_nu = result.x
        # print(f"optimal_nu: {optimal_nu}")
        if optimal_nu < 0.0:
            optimal_nu = [0.0]
        nu_changing_with_time.append(optimal_nu[0])
        time_points_for_nu.append(days_window - window_length)
        y0 = df.loc[days_window, ['S']].item(), df.loc[days_window, ['I']].item(), df.loc[days_window, ['R']].item()
        
    plt.figure(figsize=(12, 8))
    plt.plot(time_points_for_nu, nu_changing_with_time, color="#E96900")
    plt.xlabel('Time (days)')
    plt.ylabel(r'$\nu$ (nu)')
    plt.grid(True)
    # plt.show()

    new_time_points = np.arange(0, days_difference+1, 1)
    interpolated_nu = np.interp(new_time_points, time_points_for_nu, nu_changing_with_time)
    df['nu_varying_with_time'] = interpolated_nu

    N = df.loc[min(df.index), ['N']].item()
    y0 = df.loc[min(df.index), ['S']].item(), df.loc[min(df.index), ['I']].item(), df.loc[min(df.index), ['R']].item()
    initial_guess_for_beta, initial_guess_for_gamma = 0.2, 1./10
    days_difference = (max(df['date']) - min(df['date'])).days
    t = np.linspace(0, days_difference, days_difference + 1)

    initial_guesses = [initial_guess_for_beta, initial_guess_for_gamma]
    lockdown = list(df['stringency_index'].values / 100)
    nu_varying = list(df['nu_varying_with_time'].values)
    result = minimize(
        objective_function_after,
        initial_guesses,
        args=(y0, t, N, nu_varying, lockdown),
        method='Nelder-Mead',
    )
    optimal_beta, optimal_gamma = result.x
    print(f"optimal_beta: {optimal_beta} optimal_gamma: {optimal_gamma}")
    print(f"optimal_beta/optimal_gamma: {optimal_beta/optimal_gamma}")

    ret = odeint(deriv_after, y0, t, args=(N, optimal_beta, optimal_gamma, nu_varying, lockdown))
    S, I, R = ret.T

    SIRV_with_lockdown_time_varying_nu_cost = compute_cost(df['S'], S) + compute_cost(df['I'], I) + compute_cost(df['R'], R)
    SIRV_with_lockdown_time_varying_nu_cost_just_I = compute_cost(df['I'], I)
    cost_with_different_windows[window_length] = {}
    cost_with_different_windows[window_length]['SIR'] = SIRV_with_lockdown_time_varying_nu_cost
    cost_with_different_windows[window_length]['I'] = SIRV_with_lockdown_time_varying_nu_cost_just_I



cost_with_different_windows



cost_with_different_windows_df = pd.DataFrame(cost_with_different_windows)
# display(cost_with_different_windows_df)



# https://stackoverflow.com/a/76189723/10527357
plt.figure(figsize=(12, 8))
plt.plot(cost_with_different_windows_df.iloc[0].index.to_list(), cost_with_different_windows_df.iloc[0].values, marker='o', linestyle='-', color='#792373', label="SIR_loss")
plt.xlabel('Window Length')
plt.ylabel('Loss')
plt.gca().yaxis.set_major_formatter(LogFormatterSciNotation(base=10,minor_thresholds=(10,10)))

plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "window_length_loss_SIR_" + LOCATION_CHOOSEN + ".pdf"), bbox_inches="tight")
# plt.show()



plt.figure(figsize=(12, 8))
plt.plot(cost_with_different_windows_df.iloc[1].index.to_list(), cost_with_different_windows_df.iloc[1].values, marker='o', linestyle='-', color='#96A00A', label="SIR_loss")
plt.xlabel('Window Length')
plt.ylabel('Loss')
plt.gca().yaxis.set_major_formatter(LogFormatterSciNotation(base=10,minor_thresholds=(10,10)))

plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "window_length_loss_I_" + LOCATION_CHOOSEN + ".pdf"), bbox_inches="tight")
# plt.show()


# ## Vaccination dependent on Time


window_length = 15

def deriv(y, t, N, nu, lockdown):
    S, I, R = y
    t = min(int(t), len(lockdown) - 1)  # Ensure t is an integer and within the range of 'lockdown'
    dSdt = -optimal_beta * (1 - lockdown[int(t)]) * S * I / N - nu * S
    dIdt = optimal_beta * (1 - lockdown[int(t)]) * S * I / N - optimal_gamma * I
    dRdt = optimal_gamma * I + nu * S
    return dSdt[0], dIdt, dRdt[0]

def integrate_system(params, y0, t, N, lockdown):
    nu = params
    result = odeint(deriv, y0, t, args=(N, nu, lockdown), hmax=1.0)
    return result

def objective_function(params, y0, t, N, lockdown, days_window):
    predictions = integrate_system(params, y0, t, N, lockdown)
    S, I, R = predictions.T
    cost = (compute_cost(df['S'][days_window-window_length:days_window], S) + 
            compute_cost(df['I'][days_window-window_length:days_window], I) + 
            compute_cost(df['R'][days_window-window_length:days_window], R))
    return cost



N = df.loc[min(df.index), ['N']].item()
y0 = df.loc[min(df.index), ['S']].item(), df.loc[min(df.index), ['I']].item(), df.loc[min(df.index), ['R']].item()
initial_guess_for_beta, initial_guess_for_gamma, initial_guess_for_nu = 0.2, 1./10, 0.001



nu_changing_with_time = []
time_points_for_nu = []
for days_window in range(window_length, days_difference, window_length):
    print(days_window - window_length, ":", days_window)
    lockdown = list(df['stringency_index'].values / 100)[days_window - window_length:days_window]
    t = np.linspace(days_window - window_length, days_window, window_length)
    initial_guesses = [initial_guess_for_nu]
    result = minimize(
        objective_function,
        initial_guesses,
        args=(y0, t, N, lockdown, days_window),
        method='Nelder-Mead',
    )
    optimal_nu = result.x
    print(f"optimal_nu: {optimal_nu}")
    if optimal_nu < 0.0:
        optimal_nu = [0.0]
    nu_changing_with_time.append(optimal_nu[0])
    time_points_for_nu.append(days_window - window_length)
    y0 = df.loc[days_window, ['S']].item(), df.loc[days_window, ['I']].item(), df.loc[days_window, ['R']].item()


# ## SIRV model with time varying nu


plt.figure(figsize=(12, 8))
plt.plot(time_points_for_nu, nu_changing_with_time, color="#E96900")
plt.xlabel('Time (days)')
plt.ylabel(r'$\nu$ (nu)')
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "nu_varying_with_time_" + LOCATION_CHOOSEN + ".pdf"), bbox_inches="tight")
# plt.show()



new_time_points = np.arange(0, days_difference+1, 1)
interpolated_nu = np.interp(new_time_points, time_points_for_nu, nu_changing_with_time)
print(len(interpolated_nu))



plt.figure(figsize=(12, 8))
plt.plot(df['date'], interpolated_nu, color="#E96900")
plt.xlabel('Time (days)')
plt.ylabel(r'$\nu$ (nu)')
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "interpolated_nu_varying_with_time_" + LOCATION_CHOOSEN + ".pdf"), bbox_inches="tight")
# plt.show()



highlight_date = pd.to_datetime("2021-01-16")
highlight_index = df[df['date'] == highlight_date].index[0]

plt.figure(figsize=(12, 8))
plt.plot(df['date'], interpolated_nu, color="#E96900")
plt.axvline(x=df['date'][highlight_index], color="#C5373D", linestyle='--', label='16 January 2021 (Vaccine Drive Launched)')
plt.xlabel('Time (days)')
plt.ylabel(r'$\nu$ (nu)')
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "interpolated_nu_varying_with_time_" + LOCATION_CHOOSEN + ".pdf"), bbox_inches="tight")
# plt.show()



df['nu_varying_with_time'] = interpolated_nu



def deriv(y, t, N, beta, gamma, nu_varying, lockdown):
    S, I, R = y
    t = min(int(t), len(lockdown) - 1)  # Ensure t is an integer and within the range of 'lockdown'
    dSdt = -beta * (1 - lockdown[int(t)]) * S * I / N - nu_varying[int(t)] * S
    dIdt = beta * (1 - lockdown[int(t)]) * S * I / N - gamma * I
    dRdt = gamma * I + nu_varying[int(t)] * S
    return dSdt, dIdt, dRdt

def integrate_system(params, y0, t, N, nu_varying, lockdown):
    beta, gamma = params
    result = odeint(deriv, y0, t, args=(N, beta, gamma, nu_varying, lockdown))
    return result

def objective_function(params, y0, t, N, nu_varying, lockdown):
    predictions = integrate_system(params, y0, t, N, nu_varying, lockdown)
    S, I, R = predictions.T
    cost = (compute_cost(df['S'], S) + compute_cost(df['I'], I) + compute_cost(df['R'], R))
    return cost



N = df.loc[min(df.index), ['N']].item()
y0 = df.loc[min(df.index), ['S']].item(), df.loc[min(df.index), ['I']].item(), df.loc[min(df.index), ['R']].item()
initial_guess_for_beta, initial_guess_for_gamma = 0.2, 1./10
days_difference = (max(df['date']) - min(df['date'])).days
t = np.linspace(0, days_difference, days_difference + 1)



initial_guesses = [initial_guess_for_beta, initial_guess_for_gamma]
lockdown = list(df['stringency_index'].values / 100)
nu_varying = list(df['nu_varying_with_time'].values)
result = minimize(
    objective_function,
    initial_guesses,
    args=(y0, t, N, nu_varying, lockdown),
    method='Nelder-Mead',
)
optimal_beta, optimal_gamma = result.x
print(f"optimal_beta: {optimal_beta:.3f} optimal_gamma: {optimal_gamma:.3f}")
print(f"optimal_beta/optimal_gamma: {optimal_beta/optimal_gamma:.3f}")

ret = odeint(deriv, y0, t, args=(N, optimal_beta, optimal_gamma, nu_varying, lockdown))
S, I, R = ret.T

SIRV_with_lockdown_time_varying_nu_cost = compute_cost(df['S'], S) + compute_cost(df['I'], I) + compute_cost(df['R'], R)
print(f"cost: {SIRV_with_lockdown_time_varying_nu_cost:.3f}")

plt.figure(figsize=(12, 8))
plt.plot(df['date'], df['S']/df['N']*100, color="#006EAE", label='Susceptible (Data)')
plt.plot(df['date'], df['I']/df['N']*100, color="#C5373D", label='Infected (Data)')
plt.plot(df['date'], df['R']/df['N']*100, color="#429130", label='Recovered (Data)')

plt.plot(df['date'], S/N*100, color="#006EAE", linestyle="--", label='Susceptible (Model)')
plt.plot(df['date'], I/N*100, color="#C5373D", linestyle="--", label='Infected (Model)')
plt.plot(df['date'], R/N*100, color="#429130", linestyle="--", label='Recovered (Model)')

exponent = 4
loss_scaled = SIRV_with_lockdown_time_varying_nu_cost * (10 ** -exponent)
plt.plot([], [], color="none", label=rf"""Loss = ${loss_scaled:.3f} \times 10^{exponent}$""")

plt.xlabel('Date')
plt.ylabel('Percentage of Population')
plt.tick_params(length=0)
plt.grid(True)
# legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# legend.get_frame().set_alpha(0.5)
plt.legend()
plt.savefig(os.path.join(OUTPUT_DIR, "SIR_model_with_lockdown_with_vaccination_time_varying_nu_" + LOCATION_CHOOSEN + ".pdf"), bbox_inches="tight")
# plt.show()

plt.figure(figsize=(12, 8))
plt.plot(df['date'], df['stringency_index'], color='#0096A0')
plt.xlabel('Time (days)')
plt.ylabel('Stringency Index')
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "stringency_varying_with_time_" + LOCATION_CHOOSEN + ".pdf"), bbox_inches="tight")
# plt.show()



SIRV_with_lockdown_time_varying_nu_cost_I_cost = compute_cost(df['I'], I)
print(f"cost: {SIRV_with_lockdown_time_varying_nu_cost_I_cost:.3f}")

plt.figure(figsize=(12, 8))
plt.plot(df['date'], df['I']/N*100, color="#C5373D", label='Infected (Data)')
plt.plot(df['date'], I/N*100, color="#C5373D", linestyle="--", label='Infected (Model)')

exponent = 4
loss_scaled = SIRV_with_lockdown_time_varying_nu_cost_I_cost * (10 ** -exponent)
plt.plot([], [], color="none", label=rf"""Loss = ${loss_scaled:.3f} \times 10^{exponent}$""")

plt.xlabel('Date')
plt.ylabel('Percentage of Population')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "SIR_model_with_lockdown_with_vaccination_time_varying_nu_infections_" + LOCATION_CHOOSEN + ".pdf"), bbox_inches="tight")
# plt.show()



r0 = optimal_beta/optimal_gamma * (1 - np.array(lockdown))

df["S_modelled_with_lockdown_with_vaccination_time_varying_nu"] = S
df["I_modelled_with_lockdown_with_vaccination_time_varying_nu"] = I
df["R_modelled_with_lockdown_with_vaccination_time_varying_nu"] = R

df["r_eff_modelled_with_lockdown_with_vaccination_time_varying_nu"] = r0 * df["S_modelled_with_lockdown_with_vaccination_time_varying_nu"]/N
df["r_eff_actual_with_lockdown_with_vaccination_time_varying_nu"] = r0 * df["S"]/df["N"]

plt.figure(figsize=(12, 8))
plt.plot(df['date'], r0, color="#734E3D")
plt.xlabel('Date')
plt.ylabel('$R_0$')
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "SIR_model_with_lockdown_with_vaccination_time_varying_nu_r0_" + LOCATION_CHOOSEN + ".pdf"), bbox_inches="tight")
# plt.show()

descriptive_statistics(r0, method=2)



r_eff_cosine_similarity = 1 - spatial.distance.cosine(df["r_eff_actual_with_lockdown_with_vaccination_time_varying_nu"], df["r_eff_modelled_with_lockdown_with_vaccination_time_varying_nu"])

plt.figure(figsize=(12, 8))
plt.plot(df['date'], df["r_eff_actual_with_lockdown_with_vaccination_time_varying_nu"], color="#0096A0", label="$R_e$ (Model)")
plt.plot(df['date'], df["r_eff_modelled_with_lockdown_with_vaccination_time_varying_nu"], color="#0096A0", linestyle="--", label="$R_e$ (Data)")
plt.plot([], [], color="none", label=f"Cosine Similarity of $R_e$ (Model)\nand $R_e$ (Data) = {r_eff_cosine_similarity:.3f}")

plt.xlabel('Date')
plt.ylabel('$R_e$')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "SIR_model_with_lockdown_with_vaccination_time_varying_nu_reff_" + LOCATION_CHOOSEN + ".pdf"), bbox_inches="tight")
# plt.show()



print(max(df["I_modelled_with_lockdown_with_vaccination_time_varying_nu"]/N))



optimal_values = {
    'optimal_beta': optimal_beta,
    'optimal_gamma': optimal_gamma,
}

with open(OPTIMAL_VALUES_FILE, "w") as outfile: 
    json.dump(optimal_values, outfile)



#breakcode


# ## Comparing costs


print(SIR_cost, SIR_with_lockdown_cost, SIRV_with_lockdown_cost, SIRV_with_lockdown_time_varying_nu_cost)



# Function to format tick labels
def scientific_notation(tick_val, pos):
    if tick_val != 0:
        # exponent = int(np.floor(np.log10(tick_val)))
        exponent = 4
        loss_scaled = tick_val * (10 ** -exponent)
        return rf'${int(loss_scaled)} \times 10^{exponent}$'
    else:
        return rf' '



models = ['SIR', 'SIR with lockdown', r"""SIR with lockdown 
and constant $\nu$""", r"""SIRV with lockdown
and time-varying $\nu$"""]

# Corresponding costs
costs = [SIR_cost, SIR_with_lockdown_cost, SIRV_with_lockdown_cost, SIRV_with_lockdown_time_varying_nu_cost]

plt.figure(figsize=(12, 8))
bars = plt.bar(models, costs, color =["#5496CE", "#48BCBC", "#5EB342", "#C5C500"])
plt.gca().yaxis.set_major_formatter(FuncFormatter(scientific_notation))
plt.xlabel('Models')
plt.ylabel('Loss')
plt.grid(True)

for bar, cost in zip(bars, costs):
    # exponent = int(np.floor(np.log10(cost)))
    exponent = 4
    loss_scaled = cost * (10 ** -exponent)
    eqn_string = rf"""${loss_scaled:.3f} \times 10^{{{exponent}}}$"""
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{eqn_string}', 
             ha='center', va='bottom', fontsize=14)
    
plt.savefig(os.path.join(OUTPUT_DIR, "comparing_costs_SIR_" + LOCATION_CHOOSEN + ".pdf"), bbox_inches="tight")
# plt.show()
# plt.show()



print(SIR_cost_I_cost, SIR_with_lockdown_cost_I_cost, SIRV_with_lockdown_cost_I_cost, SIRV_with_lockdown_time_varying_nu_cost_I_cost)



models = ['SIR', 'SIR with lockdown', r"""SIR with lockdown 
and constant $\nu$""", r"""SIRV with lockdown
and time-varying $\nu$"""]

# Corresponding costs
costs = [SIR_cost_I_cost, SIR_with_lockdown_cost_I_cost, SIRV_with_lockdown_cost_I_cost, SIRV_with_lockdown_time_varying_nu_cost_I_cost]

plt.figure(figsize=(12, 8))
bars = plt.bar(models, costs, color =["#5496CE", "#48BCBC", "#5EB342", "#C5C500"])
plt.gca().yaxis.set_major_formatter(FuncFormatter(scientific_notation))
plt.xlabel('Models')
plt.ylabel('Loss')
for bar, cost in zip(bars, costs):
    # exponent = int(np.floor(np.log10(cost)))
    exponent = 4
    loss_scaled = cost * (10 ** -exponent)
    eqn_string = rf"""${loss_scaled:.3f} \times 10^{{{exponent}}}$"""
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{eqn_string}', 
             ha='center', va='bottom', fontsize=14)
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, "comparing_costs_I_" + LOCATION_CHOOSEN + ".pdf"), bbox_inches="tight")
# plt.show()
# plt.show()



stringency_data_points = np.arange(0, 100, 0.5)
fit_line_loaded = np.poly1d(np.load(STRINGENCY_BASED_GDP))
predicted_gdp = fit_line_loaded(stringency_data_points)
MIN_GDP = min(predicted_gdp)
MAX_GDP = max(predicted_gdp)



df['gdp_min_max_normalized'] = (df['gdp_normalized'] - MIN_GDP) / (MAX_GDP - MIN_GDP)
df['gdp_normalized_modelled_min_max_normalized'] =  (df['gdp_normalized_modelled'] - MIN_GDP) / (MAX_GDP - MIN_GDP)



df.to_csv(os.path.join(DATA_CACHE_DIR, LOCATION_CHOOSEN + "_merged_data.csv"))



# ## BREAKPOINT
df = pd.read_csv(os.path.join(DATA_CACHE_DIR, LOCATION_CHOOSEN + "_merged_data.csv"))
df['date'] = pd.to_datetime(df['date'])

with open(OPTIMAL_VALUES_FILE, 'r') as f:
    optimal_values_read = f.read()
    optimal_values = json.loads(optimal_values_read)
optimal_beta = optimal_values['optimal_beta']
optimal_gamma = optimal_values['optimal_gamma']

r0 = optimal_beta/optimal_gamma

stringency_data_points = np.arange(0, 100, 0.5)
fit_line_loaded = np.poly1d(np.load(STRINGENCY_BASED_GDP))
predicted_gdp = fit_line_loaded(stringency_data_points)
MIN_GDP = min(predicted_gdp)
MAX_GDP = max(predicted_gdp)

print(df.diff()['stringency_index'][1:].describe())



print(df['stringency_index'].describe())



print(df['nu_varying_with_time'])



from scipy.signal import medfilt
def recursive_median_filter(signal, window_size):
    # Ensure window size is odd
    if window_size % 2 == 0:
        window_size += 1
    
    # Recursive median filter implementation
    filtered_signal = np.zeros_like(signal, dtype=float)
    
    # Initialize the first value
    filtered_signal[0] = signal[0]
    
    # Apply the recursive median filter
    for i in range(1, len(signal)):
        window_start = max(0, i - window_size // 2)
        window_end = min(len(signal), i + window_size // 2 + 1)
        
        window_values = signal[window_start:window_end]
        median_value = np.median(window_values)
        
        # Update the filtered signal
        filtered_signal[i] = median_value
    
    return filtered_signal

info_saved_df = pd.read_csv("output/info_save_old/84987.16.csv")
stringency_index_from_actions_taken_raw = info_saved_df['stringency_index']



stringency_index_from_actions_taken_smoothed = medfilt(stringency_index_from_actions_taken_raw, 171) 
stringency_index_from_actions_taken_smoothed[:20] = np.mean(stringency_index_from_actions_taken_raw[:20])
stringency_index_from_actions_taken_smoothed_2 = medfilt(stringency_index_from_actions_taken_smoothed, 51)



plt.figure()
plt.plot(stringency_index_from_actions_taken_smoothed, 'r')
plt.plot(stringency_index_from_actions_taken_raw, 'g')
plt.plot(stringency_index_from_actions_taken_smoothed_2, 'b')
plt.plot(df['stringency_index'])
# plt.show()

stringency_index_from_actions_taken = stringency_index_from_actions_taken_smoothed_2

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



OUTPUT_RL = os.path.join("output", "rl_testing")

def deriv(y, t, N, beta, gamma, nu_varying, lockdown):
    S, I, R = y
    t = min(int(t), len(lockdown) - 1)  # Ensure t is an integer and within the range of 'lockdown'
    dSdt = -beta * (1 - lockdown[int(t)]) * S * I / N - nu_varying[int(t)] * S
    dIdt = beta * (1 - lockdown[int(t)]) * S * I / N - gamma * I
    dRdt = gamma * I + nu_varying[int(t)] * S
    return dSdt, dIdt, dRdt

def reward_strategy(stringency_moves, reward_strategy_choosen, sir_technique, show, stringency_file_name="nothing.csv"):   
    N = df.loc[min(df.index), ["N"]].item()
    y0 = df.loc[min(df.index), ["S"]].item(), df.loc[min(df.index), ["I"]].item(), df.loc[min(df.index), ["R"]].item()
    days_difference = (max(df["date"]) - min(df["date"])).days
    t = np.linspace(0, days_difference, days_difference + 1)
    nu_varying = list(df['nu_varying_with_time'])
    
    store_S = np.zeros(days_difference + 1)
    store_I = np.zeros(days_difference + 1)
    store_R = np.zeros(days_difference + 1)

    # sir_technique 1 is faster
    # 2 is just to check whether the results match with 1
    moves_lockdown = stringency_moves / 100
    if sir_technique == 1:
        moves_ret = odeint(deriv, y0, t, args=(N, optimal_beta, optimal_gamma, nu_varying, moves_lockdown))
        moves_S, moves_I, moves_R = moves_ret.T

        df["S_moves"] = moves_S
        df["I_moves"] = moves_I
        df["R_moves"] = moves_R
    elif sir_technique == 2:
        for ith_day in range(days_difference + 1):
            stringency_index_random_choice.append(stringency_moves[ith_day])
            t = np.linspace(0, ith_day, ith_day + 1)
            moves_ret = odeint(deriv, y0, t, args=(N, optimal_beta, optimal_gamma, nu_varying, np.array(stringency_index_random_choice) / 100))
            moves_S, moves_I, moves_R = moves_ret.T
            store_S[ith_day] = moves_S[-1]
            store_I[ith_day] = moves_I[-1]
            store_R[ith_day] = moves_R[-1]
        df["S_moves"] = store_S
        df["I_moves"] = store_I
        df["R_moves"] = store_R
    
    modelling_type = "with_lockdown_with_vaccination_time_varying_nu"
    r0 = optimal_beta/optimal_gamma * (1 - np.array(moves_lockdown))
    df["r_eff_moves_" + modelling_type] = r0 * df["S_moves"] / N
    df["gdp_normalized_moves"] = fit_line_loaded(stringency_moves)
    df["gdp_normalized_moves_min_max_normalized"] = ((fit_line_loaded(stringency_moves) - MIN_GDP) / (MAX_GDP - MIN_GDP))
    
    modelled_ret = odeint(deriv, y0, t, args=(N, optimal_beta, optimal_gamma, nu_varying, (df["stringency_index"]) / 100))
    modelled_S, modelled_I, modelled_R = modelled_ret.T
    
    df["S_modelled_" + modelling_type + "_inside_plot"] = modelled_S
    df["I_modelled_" + modelling_type + "_inside_plot"] = modelled_I
    df["R_modelled_" + modelling_type + "_inside_plot"] = modelled_R
    
        
    if reward_strategy_choosen == 1:
        plt.figure(figsize=(10, 6))
        plt.plot(df["gdp_min_max_normalized"] / df["r_eff_actual_" + modelling_type], color="b", label="reward(actual) = {reward}".format(reward = np.sum(df["gdp_min_max_normalized"] / df["r_eff_actual_" + modelling_type])))
        plt.plot(df["gdp_normalized_modelled_min_max_normalized"] / df["r_eff_modelled_" + modelling_type], color="r", label="reward(modelled) = {reward}".format(reward = np.sum(df["gdp_normalized_modelled_min_max_normalized"] / df["r_eff_modelled_" + modelling_type])))
        plt.plot(df["gdp_normalized_moves_min_max_normalized"] / df["r_eff_moves_" + modelling_type], color="g", label="reward(modelled) = {reward}".format(reward = np.sum(df["gdp_normalized_moves_min_max_normalized"] / df["r_eff_moves_" + modelling_type])))
        plt.xlabel("days")
        plt.ylabel("reward")
        plt.title("reward")
        plt.legend()
        # plt.show()
        
        plt.figure(figsize=(10, 6))
        plt.plot(df["gdp_min_max_normalized"] / df["r_eff_actual_" + modelling_type], color="b", label="reward(actual) = {reward}".format(reward = np.sum(df["gdp_min_max_normalized"] / df["r_eff_actual_" + modelling_type])))
        plt.plot(df["gdp_normalized_modelled_min_max_normalized"] / df["r_eff_modelled_" + modelling_type], color="r", label="reward(modelled) = {reward}".format(reward = np.sum(df["gdp_normalized_modelled_min_max_normalized"] / df["r_eff_modelled_" + modelling_type])))
        # plt.plot(df["gdp_normalized_moves_min_max_normalized"] / df["r_eff_moves_with_lockdown"], color="g", label="reward(modelled) = {reward}".format(reward = np.sum(df["gdp_normalized_moves_min_max_normalized"] / df["r_eff_moves_with_lockdown"])))
        plt.xlabel("days")
        plt.ylabel("reward")
        plt.title("reward")
        plt.legend()
        # plt.show()
        
    if reward_strategy_choosen == 2:
        plt.figure(figsize=(10, 6))
        
        index_to_the_power_of = 0.0025
        plt.plot(df["gdp_min_max_normalized"] / df["r_eff_actual_" + modelling_type] * np.exp(df.index.to_numpy() * index_to_the_power_of), color="b", label="reward(actual) = {reward}".format(reward = np.sum(df["gdp_min_max_normalized"] / df["r_eff_actual_" + modelling_type] * np.exp(df.index.to_numpy() * index_to_the_power_of))))
        plt.plot(df["gdp_normalized_modelled_min_max_normalized"] / df["r_eff_modelled_" + modelling_type] * np.exp(df.index.to_numpy() * index_to_the_power_of), color="r", label="reward(modelled) = {reward}".format(reward = np.sum(df["gdp_normalized_modelled_min_max_normalized"] / df["r_eff_modelled_" + modelling_type] * np.exp(df.index.to_numpy() * index_to_the_power_of))))
        plt.plot(df["gdp_normalized_moves_min_max_normalized"] / df["r_eff_moves_" + modelling_type] * np.exp(df.index.to_numpy() * index_to_the_power_of), color="g", label="reward(moves) = {reward}".format(reward = np.sum(df["gdp_normalized_moves_min_max_normalized"] / df["r_eff_moves_" + modelling_type] * np.exp(df.index.to_numpy() * index_to_the_power_of))))
        plt.xlabel("days")
        plt.ylabel("reward")
        plt.title("reward")
        plt.legend()
        # plt.show()
        
        plt.figure(figsize=(10, 6))
        plt.plot(df["gdp_min_max_normalized"] / df["r_eff_actual_" + modelling_type] * np.exp(df.index.to_numpy() * index_to_the_power_of), color="b", label="reward(actual) = {reward}".format(reward = np.sum(df["gdp_min_max_normalized"] / df["r_eff_actual_" + modelling_type] * np.exp(df.index.to_numpy() * index_to_the_power_of))))
        plt.plot(df["gdp_normalized_modelled_min_max_normalized"] / df["r_eff_modelled_" + modelling_type] * np.exp(df.index.to_numpy() * index_to_the_power_of), color="r", label="reward(modelled) = {reward}".format(reward = np.sum(df["gdp_normalized_modelled_min_max_normalized"] / df["r_eff_modelled_" + modelling_type] * np.exp(df.index.to_numpy() * index_to_the_power_of))))
        # plt.plot(df["gdp_normalized_moves_min_max_normalized"] / df["r_eff_moves_with_lockdown"] * np.exp(df.index.to_numpy() * index_to_the_power_of), color="g", label="reward(moves) = {reward}".format(reward = np.sum(df["gdp_normalized_moves_min_max_normalized"] / df["r_eff_moves_with_lockdown"] * np.exp(df.index.to_numpy() * index_to_the_power_of))))
        plt.xlabel("days")
        plt.ylabel("reward")
        plt.title("reward")
        plt.legend()
        # plt.show()
        
    if reward_strategy_choosen == 3:
        
        hospital_capacity = 0.003
        hospital_capacity_punishment = -2000
        hospital_capacity_reward = 50
        I_reward_actual = [hospital_capacity_punishment if I_percentage >= hospital_capacity else hospital_capacity_reward for I_percentage in df["I"] / df["N"]]
        I_reward_modelled = [hospital_capacity_punishment if I_percentage >= hospital_capacity else hospital_capacity_reward for I_percentage in df["I_modelled_" + modelling_type] / N]
        I_reward_moves = [hospital_capacity_punishment if I_percentage >= hospital_capacity else hospital_capacity_reward for I_percentage in df["I_moves"] / N]
        
        inertia_rewards_actual = np.array([0] + [abs(diff)*5*-1 for diff in (df['stringency_index'][i] - df['stringency_index'][i - 1] for i in range(1, len(df)))])
        # modelled reward for inertia is same as actual
        inertia_rewards_modelled = np.array([0] + [abs(diff)*5*-1 for diff in (df['stringency_index'][i] - df['stringency_index'][i - 1] for i in range(1, len(df)))])
        inertia_rewards_moves = np.array([0] + [abs(diff)*5*-1 for diff in (stringency_moves[i] - stringency_moves[i - 1] for i in range(1, len(stringency_moves)))])
        
        reward_actual = np.array(calculate_reward_weighted(df["gdp_min_max_normalized"], df["r_eff_actual_" + modelling_type])) + I_reward_actual + inertia_rewards_actual
        reward_modelled = np.array(calculate_reward_weighted(df["gdp_normalized_modelled_min_max_normalized"], df["r_eff_modelled_" + modelling_type])) + I_reward_modelled + inertia_rewards_modelled
        reward_moves = np.array(calculate_reward_weighted(df["gdp_normalized_moves_min_max_normalized"], df["r_eff_moves_" + modelling_type])) + I_reward_moves + inertia_rewards_moves
        reward_moves_weighted = np.array(calculate_reward_weighted(df["gdp_normalized_moves_min_max_normalized"], df["r_eff_moves_" + modelling_type]))
        if int(reward_moves.sum()) < 0:
            print(reward_moves.sum(), " negative reward")
            return reward_moves, reward_moves_weighted, I_reward_moves, inertia_rewards_moves
        else:
            output_path_img = os.path.join(OUTPUT_RL, str(int(reward_moves.sum())))
            try:
                os.makedirs(output_path_img)
            except:
                print("path exists")
            
            f = open(os.path.join(output_path_img, stringency_file_name), "w")
            f.write("")
            f.close()
            
            plt.figure(figsize=(12, 8))
            plt.plot(df['date'], stringency_index_from_actions_taken_raw, color="#C5373D", label="Stringency from Model")
            plt.plot(df['date'], stringency_index_from_actions_taken, color="#429130", label="Median Filtered Stringency")
            plt.xlabel("Date")
            plt.ylabel("Stringency Index")
            # plt.title("stringency smoothing")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(output_path_img, "rl_stringency_smoothing.png"))
            plt.savefig(os.path.join(output_path_img, "rl_stringency_smoothing.pdf"))
            # if show == True:
                # plt.show()
            plt.close()

            plt.figure(figsize=(12, 8))
            plt.plot(df['date'], df['S']/df['N'], color="#006EAE", label='Susceptible (actual)')
            plt.plot(df['date'], df['I']/df['N'], color="#C5373D", label='Infected (actual)')
            plt.plot(df['date'], df['R']/df['N'], color="#429130", label='Recovered (actual)')
            plt.plot(df['date'], df['S_modelled_' + modelling_type]/N, color="#006EAE", linestyle=":", label='Susceptible (modelled)')
            plt.plot(df['date'], df['I_modelled_' + modelling_type]/N, color="#C5373D", linestyle=":", label='Infected (modelled)')
            plt.plot(df['date'], df['R_modelled_' + modelling_type]/N, color="#429130", linestyle=":", label='Recovered (modelled)')
            plt.plot(df['date'], df['S_moves']/N, color="#006EAE", linestyle="--", label='Susceptible (rl)')
            plt.plot(df['date'], df['I_moves']/N, color="#C5373D", linestyle="--", label='Infected (rl)')
            plt.plot(df['date'], df['R_moves']/N, color="#429130", linestyle="--", label='Recovered (rl)')
            plt.xlabel('Date')
            plt.ylabel('Percentage of Population')
            # plt.title('SIR Epidemic Trajectory')
            plt.tick_params(length=0)
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(output_path_img, "rl_sir.png"))
            plt.savefig(os.path.join(output_path_img, "rl_sir.pdf"))
            # if show == True:
                # plt.show()
            plt.close()

            plt.figure(figsize=(12, 8))
            plt.plot(df['date'], df['I']/df['N'], color="#C5373D", label='Infected (actual)')
            plt.plot(df['date'], df['I_modelled_' + modelling_type]/N, color="#006EAE", label='Infected (modelled)')
            plt.plot(df['date'], df['I_moves']/N, color="#429130", label='Infected (rl)')
            plt.xlabel("Date")
            plt.ylabel("Percentage of Infected Population")
            # plt.title("SIR Epidemic Trajectory (Infected)")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(output_path_img, "rl_i.png"))
            plt.savefig(os.path.join(output_path_img, "rl_i.pdf"))
            # if show == True:
                # plt.show()
            plt.close()

            plt.figure(figsize=(12, 8))
            plt.plot(df['date'], df['I_moves']/N, color="#429130", label='Infected (rl)')
            plt.xlabel("Date")
            plt.ylabel("Percentage of Infected Population")
            plt.title("SIR Epidemic Trajectory (Infected)")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(output_path_img, "rl_i_moves.png"))
            plt.savefig(os.path.join(output_path_img, "rl_i_moves.pdf"))
            # if show == True:
                # plt.show()
            plt.close()

            plt.figure(figsize=(12, 8))
            plt.plot(df['date'], df['stringency_index'], color="#006EAE", label="Stringency (actual)")
            # plt.plot(df['date'], stringency_index_from_actions_taken_raw, color="#C5373D", alpha=0.4, label="Stringency (rl)")
            plt.plot(df['date'], stringency_moves, color="#429130", label="Stringency (rl)")
            plt.xlabel("Date")
            plt.ylabel("Stringency Index")
            # plt.title("Stringency over Time")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(output_path_img, "rl_stringency.pdf"))
            plt.savefig(os.path.join(output_path_img, "rl_stringency.png"))
            plt.savefig(os.path.join(output_path_img, "rl_stringency.pdf"))
            # if show == True:
                # plt.show()
            plt.close()

            plt.figure(figsize=(12, 8))
            plt.plot(df['date'], df['gdp_normalized'], color="#C5373D", label="GDP normalized (actual)")
            plt.plot(df['date'], df['gdp_normalized_modelled'], color="#006EAE", label="GDP normalized (modelled)")
            plt.plot(df['date'], df['gdp_normalized_moves'], color="#429130", label="GDP normalized (rl)")
            plt.xlabel("Date")
            plt.ylabel("GDP")
            # plt.title("GDP over Time")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(output_path_img, "rl_gdp.png"))
            plt.savefig(os.path.join(output_path_img, "rl_gdp.pdf"))
            # if show == True:
                # plt.show()
            plt.close()

            r_eff_target_value = 1.0
            first_time_r_eff_actual_1 = next((t for t, r_eff in zip(df['date'].dt.strftime('%Y-%m-%d'), df['r_eff_actual_' + modelling_type]) if r_eff <= r_eff_target_value), None)
            first_time_r_eff_modelled_1 = next((t for t, r_eff in zip(df['date'].dt.strftime('%Y-%m-%d'), df['r_eff_modelled_' + modelling_type]) if r_eff <= r_eff_target_value), None)
            first_time_r_eff_1 = next((t for t, r_eff in zip(df['date'].dt.strftime('%Y-%m-%d'), df["r_eff_moves_" + modelling_type]) if r_eff <= r_eff_target_value), None)

            plt.figure(figsize=(12, 8))
            plt.plot(df['date'], df['r_eff_actual_' + modelling_type], color="#C5373D", label="$R_e$ (actual)")
            plt.plot(df['date'], df['r_eff_modelled_' + modelling_type], color="#006EAE", label="$R_e$ (modelled)")
            plt.plot(df['date'], df["r_eff_moves_" + modelling_type], color="#429130", label="$R_e$ (rl)")
            plt.xlabel("Date")
            plt.ylabel("R_e")
            # plt.title("R_e over Time")
            plt.grid(True)
            legend = plt.legend()
            plt.savefig(os.path.join(output_path_img, "rl_r_eff.png"))
            plt.savefig(os.path.join(output_path_img, "rl_r_eff.pdf"))
            # if show == True:
                # plt.show()
            plt.close()

            # print("len df", len(calculate_reward_weighted(df["gdp_min_max_normalized"], df["r_eff_actual_" + modelling_type])))

            plt.figure(figsize=(12, 8))
            plt.plot(df['date'], reward_actual, color="#C5373D", label=f"Reward (actual) Total: {reward_actual.sum():.2f}")
            plt.plot(df['date'], reward_modelled, color="#006EAE", label=f"Reward (modelled) Total: {reward_modelled.sum():.2f}")
            plt.plot(df['date'], reward_moves, color="#429130", label=f"Reward (rl) Total: {reward_moves.sum():.2f}")
            plt.xlabel("Date")
            plt.ylabel("Reward")
            # plt.title("Reward over Time")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(output_path_img, "rl_reward.png"))
            plt.savefig(os.path.join(output_path_img, "rl_reward.pdf"))
            # if show == True:
                # plt.show()
            plt.close()

            return reward_moves, reward_moves_weighted, I_reward_moves, inertia_rewards_moves



rl_reward_moves, rl_reward_weighted, rl_reward_I_percentage, rl_reward_inertia = reward_strategy(stringency_index_from_actions_taken, reward_strategy_choosen=3, sir_technique=1, show=True)


# ### plt.plot(rl_reward_I_percentage)


info_save_dir = "output/info_save_old/"
for info_file in os.listdir(info_save_dir):
    info_file_path = os.path.join(info_save_dir, info_file)
    info_saved_df = pd.read_csv(info_file_path)
    stringency_index_from_actions_taken_raw = info_saved_df['stringency_index']
    stringency_index_from_actions_taken_smoothed = medfilt(stringency_index_from_actions_taken_raw, 171) 
    stringency_index_from_actions_taken_smoothed[:30] = np.mean(stringency_index_from_actions_taken_raw[:30])
    stringency_index_from_actions_taken_smoothed_2 = medfilt(stringency_index_from_actions_taken_smoothed, 51)
    
    stringency_index_from_actions_taken = stringency_index_from_actions_taken_smoothed_2
    reward_moves, reward_moves_weighted, I_reward_moves, inertia_rewards_moves = reward_strategy(stringency_index_from_actions_taken, reward_strategy_choosen=3, sir_technique=1, show=False, stringency_file_name=info_file)