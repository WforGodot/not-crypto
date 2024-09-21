from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from solver_env import CryptarithmeticEnv


PROBLEM = "EVEN + HERE = TENT"


if __name__ == "__main__":
    problem = PROBLEM

    # Create and wrap the environment
    def env_creator():
        return CryptarithmeticEnv(problem)

    env = DummyVecEnv([env_creator])

    # Create a separate environment for evaluation
    eval_env = DummyVecEnv([env_creator])

    # Set up the evaluation callback
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=2000,
                                 deterministic=True, render=False)

    model = PPO('MlpPolicy', env, verbose=1)
    
    # Train the model with the evaluation callback
    model.learn(total_timesteps=20000, callback=eval_callback)
    
    model.save("ppo_cryptarithmetic")

    # Optionally, evaluate the trained model
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")