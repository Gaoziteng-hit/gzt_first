import gym
import gym_sumo
from a2c import A2CAgent
from dqn_original import DQNAgent

#创建一个gym_sumo环境
env = gym.make('gym_sumo-v0')

#调用dqn中的类DQNAent
agent = DQNAgent()

agent.train(env)# = DQNAgent.train(env) = DQNAgent.train(gym.make('gym_sumo-v0'))
