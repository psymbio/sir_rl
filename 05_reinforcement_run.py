from rl_utils_new import *
    
env = SIREnvironment()
env.reset()

# models/1704106458/10000.zip
model_path = os.path.join(MODELS_DIR, "1704106458", "10000.zip")

model = PPO.load(model_path, env=env)

episodes = 1
for episode in range(1, episodes+1):
    obs, info = env.reset()
    terminated = False
    score = 0
    actions_taken = []
    while not terminated:
        action, _states = model.predict(obs)
        actions_taken.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
    print(f'Episode: {episode}, Score: {score}')
    env.render(score=score)
    with open("actions_taken.txt", "w") as f:
        for action in actions_taken:
            f.write(str(action) +"\n")
