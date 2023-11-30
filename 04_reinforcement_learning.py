from rl_utils import *
import time

# env = SIREnvironment()
# episodes = 10
# for episode in range(1, episodes+1):
#     state, info = env.reset()
#     terminated = False
#     score = 0
#     while not terminated:
#         action = env.action_space.sample()
#         state, reward, terminated, truncated, info = env.step(action)
#         score += reward
#     print(f'Episode: {episode}, Score: {score}')
#     env.render()

env = SIREnvironment()
check_env(env)

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = SIREnvironment()
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS*iters}")