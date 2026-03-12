# rl_agent.py — run AFTER nosql_injection_ml_project.py
import numpy as np, pandas as pd
import gymnasium as gym
from gymnasium import spaces
import joblib, warnings

warnings.filterwarnings('ignore')
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback

svm = joblib.load('models/svm_model.pkl')


class BlockchainWAFEnv(gym.Env):
    '''
    RL Environment for Blockchain WAF response optimisation.
    State  : PCA-transformed feature vector of one request
    Actions: 0=Allow  1=Alert  2=Block
    Reward : +10 correct block, -10 missed attack,
             -5 false positive, +1 correct allow
    '''

    def __init__(self, X, y):
        super().__init__()
        self.X = np.array(X, dtype=np.float32)
        self.y = np.array(y, dtype=np.int32)
        self.n = len(X)
        self.idx = 0
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(X.shape[1],), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = self.np_random.integers(0, self.n)
        return self.X[self.idx], {}

    def step(self, action):
        label = self.y[self.idx]
        if label == 1:  # Real attack
            reward = {2: 10.0, 1: 3.0, 0: -10.0}[action]
        else:  # Normal traffic
            reward = {0: 1.0, 1: -1.0, 2: -5.0}[action]
        self.idx = (self.idx + 1) % self.n
        return self.X[self.idx], reward, False, False, {}


if __name__ == '__main__':
    import os;

    os.makedirs('models', exist_ok=True)
    # X_tr_pca, X_te_pca, y_tr, y_te from main pipeline
    env = BlockchainWAFEnv(X_tr_pca, y_tr.values)
    eval_env = BlockchainWAFEnv(X_te_pca, y_te.values)
    check_env(env)

    agent = DQN('MlpPolicy', env,
                learning_rate=1e-3, buffer_size=10000,
                batch_size=32, gamma=0.95,
                exploration_fraction=0.3,
                exploration_final_eps=0.05, verbose=1)

    agent.learn(total_timesteps=50000,
                callback=EvalCallback(eval_env,
                                      best_model_save_path='models/',
                                      eval_freq=500, verbose=1))
    agent.save('models/dqn_blockchain_agent')
    print('[✓] RL agent saved → models/dqn_blockchain_agent')