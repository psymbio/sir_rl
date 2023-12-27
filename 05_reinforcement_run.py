from rl_utils import *
    
env = SIREnvironment()
env.reset()

# model_path = os.path.join(MODELS_DIR, "1701361856", "50000.zip")
# model_path = os.path.join(MODELS_DIR, "1702301430", "280000.zip")
# models/1702708503/610000.zip
# models/1702805721/390000.zip
# models/1702811576/1040000.zip
# models/1702896590/220000.zip
model_path = os.path.join(MODELS_DIR, "1702896590", "220000.zip")

model = PPO.load(model_path, env=env)

episodes = 1
for episode in range(1, episodes+1):
    obs, info = env.reset()
    terminated = False
    score = 0
    while not terminated:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
    print(f'Episode: {episode}, Score: {score}')
    env.render(score=score)
