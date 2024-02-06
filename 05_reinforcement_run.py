from rl_utils_new_complex_reward_sirvlockdown_nu_varying import *
    
env = SIREnvironment()
env.reset()

# models/1704106458/10000.zip
# models/1704189286/1033.zip
# models/1704190312/4132.zip
# models/1704192521/72310.zip
# models/1704205452_ppo_100iter/103300.zip
# models/1704256387_ppo_100iter_hospital-100/103300.zip
# models/1704275312/1033.zip
# models/1704351335/542325.zip
# models/1704382102/346055.zip

# models/1704534388/1229270.zip
# models/1704534388/356385.zip
# models/1704534388/139455.zip

# models/1704873042/5165.zip
# models/1704874281/374740.zip
# models/1704994098/2742.zip
# models/1705049761/79518.zip
# models/1705085840/16452.zip
# models/1705154668/548400.zi
# models/1705305871/578562.zip
# model_path = os.path.join(MODELS_DIR, "1705305871", "578562.zip")\
# models/1705330479/839052.zip
# model_path = os.path.join(MODELS_DIR, "1705330479", "839052.zip")
# model_path = os.path.join(MODELS_DIR, "1705049761", "79518.zip")
# models/1705427702_herd/274200.zip
# model_path = os.path.join(MODELS_DIR, "1705427702_herd", "274200.zip")
model_path = os.path.join(MODELS_DIR, "hpc_run", "1746654.zip")
model = PPO.load(model_path, env=env)

episodes = 10
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
    