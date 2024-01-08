from rl_utils_new_complex_reward import *
    
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
model_path = os.path.join(MODELS_DIR, "1704534388", "139455.zip")

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
        score += reward
    print(f'Episode: {episode}, Score: {score}')
    env.render(score=score)
    with open(os.path.join(OUTPUT_DIR, "actions_taken", f"{score:.2f}.txt"), "w") as f:
        for action in actions_taken:
            f.write(str(action) +"\n")
    info_save_df = pd.DataFrame.from_dict(info_save)
    info_save_df.to_csv(os.path.join(OUTPUT_DIR, "info_save", f"{score:.2f}.csv"), index=False)