# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 11:27:36 2022

@author: mariana
API: OpenAI Gym
"""
import pygame
import gym

class Agent:
    def __init__(self):
        self.r = 0.0
        
    def step(self, env: gym.Env):
        # choose a random action from environment (world)
        a = env.action_space.sample()
        s, r, end, _ = env.step(a)
        self.r += r
        return end
    
# program: the world contains the set of actions, from which we choose randomly with step() function
agent = Agent()
# create environment
env = gym.make('CartPole-v0')
# the world must always begin from initial state
obs = env.reset()
end = False
steps = 0

# 200 states
while steps <= 200:
    env.render()
    end = agent.step(env)
    steps += 1
    print(end)
env.close()
r = agent.r