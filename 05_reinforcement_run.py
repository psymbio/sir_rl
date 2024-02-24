from rl_utils_new_complex_reward_sirvlockdown_nu_varying import *
    
env = SIREnvironment()
env.reset()

# models/hpc_run/2667966.zip
# models/hpc_run/4483170.zip
# models/hpc_run_2/1225674.zip
model_path = os.path.join(MODELS_DIR, "hpc_run_2", "786954.zip")
model = PPO.load(model_path, env=env)

episodes = 30
for episode in range(1, episodes+1):
    obs, info = env.reset()
    terminated = False
    score = 0
    actions_taken = []
    info_save = []
    while not terminated:
        action, _states = model.predict(obs)
        actions_taken.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        info_save.append(info)
        # score += reward
        # score += info['reward']
        # with open(os.path.join(OUTPUT_DIR, "score_check.txt"), "a") as f:
        #     for action in actions_taken:
        #         f.write(str(reward) +"\n")
    info_save_df = pd.DataFrame.from_dict(info_save)
    score = sum(info_save_df['reward'])
    info_save_df.to_csv(os.path.join(OUTPUT_DIR, "info_save", f"{score:.2f}.csv"), index=False)
    env.render(score=score, stringency=np.array(info_save_df['stringency_index']))
    print(f'Episode: {episode}, Score: {score}')
    with open(os.path.join(OUTPUT_DIR, "actions_taken", f"{score:.2f}.txt"), "w") as f:
        for action in actions_taken:
            f.write(str(action) +"\n")
    