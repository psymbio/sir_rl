from rl_utils_new_complex_reward_sirvlockdown_nu_varying import *
import time

TIMESTEPS = 2742
models_dir = f"models/{int(time.time())}/"
if not os.path.exists(models_dir):
	os.makedirs(models_dir)
env = SIREnvironment()
# model_path = os.path.join(MODELS_DIR, "1705154668", "548400.zip")
# models/1705318802/586788.zip
model_path = os.path.join(MODELS_DIR, "1705318802", "586788.zip")
iters = int(586788 / TIMESTEPS)
model = PPO.load(model_path)
env.reset()
model.set_env(env)
while True:
    iters += 1
    print("ITER", iters)
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
    model.save(f"{models_dir}/{TIMESTEPS*iters}")