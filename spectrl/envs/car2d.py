from scipy.stats import truncnorm

import numpy as np
import gym

class envGoal():
    def __init__(self):
        self.startX = 0.
        self.startY = 0.
        self.endX = 0.
        self.endY = 0.
        self.round_num = 0
        self.edg_num = 0
        self.target_num = 0

class VC_Env(gym.Env):
    def __init__(self, time_limit, set_goal, std=0.2, initx=-1, inity=-1):
        # self.state = np.array([5.0, 0.]) + truncnorm.rvs(-1, 1, 0, 0.3, 2)
        # self.state = np.array([0., 0., 0., 0., 0., 0.]) # [x, y, startX, startY, endX, endY]

        self.set_goal = set_goal
        self.state = np.array([set_goal.startX, set_goal.startY]) + truncnorm.rvs(-1, 1, 0, 0.3, 2)
        # self.state = np.append(self.state, [0., 0., 0., 0.])

        self.time_limit = time_limit
        self.time = 0
        self.std = std

        shape = self.state.shape[0]

        # self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(2,))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(shape,))

        self.action_space = gym.spaces.Box(-1, 1, shape=(2,))

        #just for gtop to initial spec xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        # self.initx = initx
        # self.inity = inity

    def reset(self):
        # self.state = np.array([5.0, 0.]) + truncnorm.rvs(-1, 1, 0, 0.3, 2)
        self.state = np.array([self.set_goal.startX, self.set_goal.startY]) + truncnorm.rvs(-1, 1, 0, 0.3, 2)
        
        # temp = self.state[2:]
        # self.state = np.array([temp[0], temp[1]]) + truncnorm.rvs(-1, 1, 0, 0.3, 2)
        # self.state = np.array([temp[0], temp[1]])
        # if self.initx != -1:
        #     self.state = np.array([self.initx, self.inity]) + truncnorm.rvs(-1, 1, 0, 0.3, 2)

        # self.state = np.append(self.state, temp)

        self.time = 0
        return self.state

    def step(self, action):
        action = action * np.array([0.5, np.pi]) + np.array([0.5, 0.])
        velocity = action[0] * np.array([np.cos(action[1]), np.sin(action[1])])
        # temp = self.state[2:]
        # next_state = self.state[:2] + velocity + truncnorm.rvs(-1, 1, 0, self.std, 2)
        # next_state = self.state[:2] + velocity
        # next_state = np.append(next_state, temp)
        next_state = self.state + velocity + truncnorm.rvs(-1, 1, 0, self.std, 2)

        self.state = next_state
        self.time = self.time + 1
        return next_state, 0, self.time > self.time_limit, None

    def render(self):
        print("System state : ",self.state)
        print(f"Obs space : {self.observation_space.shape}, Action space : {self.action_space.shape}")

    def get_sim_state(self):
        return self.state

    def set_sim_state(self, state):
        self.state = state
        return self.state

    def close(self):
        pass
