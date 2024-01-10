from rl_utils_new_complex_reward_sirvlockdown import *
import time

# base testing code
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

# https://stackoverflow.com/questions/56700948/understanding-the-total-timesteps-parameter-in-stable-baselines-models
TIMESTEPS = 4570

if RL_LEARNING_TYPE == "normal":
	env = SIREnvironment()
	env.reset()
	model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
	iters = 0
	while True:
		iters += 1
		model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
		model.save(f"{models_dir}/{TIMESTEPS*iters}")
elif RL_LEARNING_TYPE == "deep":
	env = SIREnvironment()
	env.reset()
	policy_kwargs = dict(
		features_extractor_class=CustomCombinedExtractor,
	)
	model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=0)
	iters = 0
	while True:
		iters += 1
		print("ITER", iters)
		model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
		model.save(f"{models_dir}/{TIMESTEPS*iters}")