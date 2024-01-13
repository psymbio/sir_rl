modelling_type = "with_lockdown_with_vaccination_time_varying_nu"

output_path_img = os.path.join(OUTPUT_RL, str(score))
try:
    os.makedirs(output_path_img)
except:
    print("path exists")

plt.figure(figsize=(12, 8))
plt.plot(self.df['date'], self.df['S']/self.df['N'], 'b', alpha=0.5, lw=2, label='Susceptible (actual)')
plt.plot(self.df['date'], self.df['I']/self.df['N'], 'r', alpha=0.5, lw=2, label='Infected (actual)')
plt.plot(self.df['date'], self.df['R']/self.df['N'], 'g', alpha=0.5, lw=2, label='Recovered (actual)')
plt.plot(self.df['date'], self.df['S_modelled_' + modelling_type]/self.N, 'b:', alpha=0.5, lw=2, label='Susceptible (modelled)')
plt.plot(self.df['date'], self.df['I_modelled_' + modelling_type]/self.N, 'r:', alpha=0.5, lw=2, label='Infected (modelled)')
plt.plot(self.df['date'], self.df['R_modelled_' + modelling_type]/self.N, 'g:', alpha=0.5, lw=2, label='Recovered (modelled)')
plt.plot(self.df['date'], self.df['S_moves']/self.N, 'b--', alpha=0.5, lw=2, label='Susceptible (rl)')
plt.plot(self.df['date'], self.df['I_moves']/self.N, 'r--', alpha=0.5, lw=2, label='Infected (rl)')
plt.plot(self.df['date'], self.df['R_moves']/self.N, 'g--', alpha=0.5, lw=2, label='Recovered (rl)')
plt.xlabel('Date')
plt.ylabel('Percentage of Population')
plt.title('SIR Epidemic Trajectory')
plt.tick_params(length=0)
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(output_path_img, "rl_sir.png"))
plt.savefig(os.path.join(output_path_img, "rl_sir.eps"))

plt.figure(figsize=(12, 8))
plt.plot(self.df['date'], self.df['I']/self.df['N'], 'r', alpha=0.5, lw=2, label='Infected (actual)')
plt.plot(self.df['date'], self.df['I_modelled_' + modelling_type]/self.N, 'r:', alpha=0.5, lw=2, label='Infected (modelled)')
plt.plot(self.df['date'], self.df['I_moves']/self.N, 'r--', alpha=0.5, lw=2, label='Infected (rl)')
plt.xlabel("Date")
plt.ylabel("Percentage of Infected Population")
plt.title("SIR Epidemic Trajectory (Infected)")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(output_path_img, "rl_i.png"))
plt.savefig(os.path.join(output_path_img, "rl_i.eps"))

plt.figure(figsize=(12, 8))
plt.plot(self.df['date'], self.df['stringency_index'], 'b', label="Stringency (actual)")
plt.plot(self.df['date'], self.store_stringency, 'g', label="Stringency (rl)")
plt.xlabel("Date")
plt.ylabel("Stringency Index")
plt.title("Stringency over Time")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(output_path_img, "rl_stringency.png"))
plt.savefig(os.path.join(output_path_img, "rl_stringency.eps"))

plt.figure(figsize=(12, 8))
plt.plot(self.df['date'], self.df['gdp_normalized'], 'r', label="GDP normalized (actual)")
plt.plot(self.df['date'], self.df['gdp_normalized_modelled'], 'b', label="GDP normalized (modelled)")
plt.plot(self.df['date'], self.df['gdp_normalized_moves'], 'g', label="GDP normalized (rl)")
plt.xlabel("Date")
plt.ylabel("GDP")
plt.title("GDP over Time")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(output_path_img, "rl_gdp.png"))
plt.savefig(os.path.join(output_path_img, "rl_gdp.eps"))

first_time_r_eff_actual_1 = next((t for t, r_eff in zip(self.df['date'], self.df['r_eff_actual_' + modelling_type]) if r_eff <= 1), None)
first_time_r_eff_modelled_1 = next((t for t, r_eff in zip(self.df['date'], self.df['r_eff_modelled_' + modelling_type]) if r_eff <= 1), None)
first_time_r_eff_1 = next((t for t, r_eff in zip(self.df['date'], self.store_r_eff) if r_eff <= 1), None)

plt.figure(figsize=(12, 8))
plt.plot(self.df['date'], self.df['r_eff_actual_' + modelling_type], 'r', label="R_eff (actual)")
plt.plot(self.df['date'], self.df['r_eff_modelled_' + modelling_type], 'b', label="R_eff (modelled)")
plt.plot(self.df['date'], self.df["r_eff_moves_" + modelling_type], 'g', label="R_eff (rl)")
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

hospital_capacity = 0.003
hospital_capacity_punishment = -5000
hospital_capacity_reward = 20
I_reward_actual = [hospital_capacity_punishment if I_percentage >= hospital_capacity else hospital_capacity_reward for I_percentage in self.df["I"] / self.df["N"]]
I_reward_modelled = [hospital_capacity_punishment if I_percentage >= hospital_capacity else hospital_capacity_reward for I_percentage in self.df["I_modelled_" + modelling_type] / self.N]
I_reward_moves = [hospital_capacity_punishment if I_percentage >= hospital_capacity else hospital_capacity_reward for I_percentage in self.df["I_moves"] / N]

r_eff_reward_choosen = 10
r_eff_punishment_choosen = -10
r_eff_level = 1.9
r_eff_reward_actual = np.array([r_eff_reward_choosen if r_eff <= r_eff_level else r_eff_punishment_choosen for r_eff in self.df["r_eff_actual_" + modelling_type]])
r_eff_reward_modelled = np.array([r_eff_reward_choosen if r_eff <= r_eff_level else r_eff_punishment_choosen for r_eff in self.df["r_eff_modelled_" + modelling_type]])
r_eff_reward_moves = np.array([r_eff_reward_choosen if r_eff <= r_eff_level else r_eff_punishment_choosen for r_eff in self.df["r_eff_moves_" + modelling_type]])

inertia_rewards_actual = np.array([0] + [abs(diff)*5*-1 for diff in (self.df['stringency_index'][i] - self.df['stringency_index'][i - 1] for i in range(1, len(self.df)))])
# modelled reward for intertia is same as actual
inertia_rewards_modelled = np.array([0] + [abs(diff)*5*-1 for diff in (self.df['stringency_index'][i] - self.df['stringency_index'][i - 1] for i in range(1, len(self.df)))])
inertia_rewards_moves = np.array([0] + [abs(diff)*5*-1 for diff in (stringency[i] - stringency[i - 1] for i in range(1, len(stringency)))])

reward_actual = np.array(calculate_reward_weighted(self.df["gdp_min_max_normalized"], self.df["r_eff_actual_" + modelling_type])) + I_reward_actual + r_eff_reward_actual + inertia_rewards_actual
reward_modelled = np.array(calculate_reward_weighted(self.df["gdp_normalized_modelled_min_max_normalized"], self.df["r_eff_modelled_" + modelling_type])) + I_reward_modelled + r_eff_reward_modelled + inertia_rewards_modelled
reward_moves = np.array(calculate_reward_weighted(self.df["gdp_normalized_moves_min_max_normalized"], self.df["r_eff_moves_" + modelling_type])) + I_reward_moves + r_eff_reward_moves + inertia_rewards_moves

print("len df", len(calculate_reward_weighted(self.df["gdp_min_max_normalized"], self.df["r_eff_actual_" + modelling_type])))

plt.figure(figsize=(12, 8))
plt.plot(self.df['date'], reward_actual, 'r', label=f"Reward (actual) Total: {reward_actual.sum():.2f}")
plt.plot(self.df['date'], reward_modelled, 'b', label=f"Reward (modelled) Total: {reward_modelled.sum():.2f}")
plt.plot(self.df['date'], reward_moves, 'g', label=f"Reward (rl) Total: {reward_moves.sum():.2f}")
plt.xlabel("Time /days")
plt.ylabel("Reward")
plt.title("Reward over Time")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(output_path_img, "rl_reward.png"))
plt.savefig(os.path.join(output_path_img, "rl_reward.eps"))