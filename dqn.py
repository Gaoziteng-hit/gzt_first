import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import tensorflow.keras.layers as kl
import numpy as np
import logging
import datetime
import os
import random
import math
from replayMemory import ReplayMemory
from time import gmtime, strftime, localtime

import pandas as pd
import openpyxl

def save_eps(data_name,file_name):
    # list转dataframe
    df = pd.DataFrame(data_name, columns=['Epsilon(s)'])
	# 保存到本地excel
    df.to_excel(file_name, index=False)

# return model = tf.keras.models.Model(input, X, name='CarModel') 封装NN model 输入为input，输出X
def CarModel(num_actions, input_len):
    input = kl.Input(shape=(input_len)) # input()这个方法是用来初始化一个keras tensor的，tensor说白了就是个数组。他强大到之通过输入和输出就能建立一个keras模型。
    hidden1 = kl.Dense(64, activation='relu')(input) # 添加一个全连接层
    hidden2 = kl.Dense(128, activation='relu')(hidden1)
    hidden3 = kl.Dense(64, activation='relu')(hidden2)
    state_value = kl.Dense(1)(hidden3) # 输入为hidden3的输出，输出1个值
    # kl.Lambda()用以对上一层的输出施以任何Theano/TensorFlow表达式
    # f.keras.backend.expand_dims( ,-1) 增加维度
    state_value = kl.Lambda(lambda s: tf.keras.backend.expand_dims(s[:, 0], -1), output_shape=(num_actions,))(state_value) # ???

    action_advantage = kl.Dense(num_actions)(hidden3)
    action_advantage = kl.Lambda(lambda a: a[:, :] - tf.keras.backend.mean(a[:, :], keepdims=True), output_shape=(num_actions,))(
        action_advantage)

    X = kl.Add()([state_value, action_advantage]) # add（）：直接对张量求和

    model = tf.keras.models.Model(input, X, name='CarModel') # 封装NN model 输入为input，输出X
    return model

# return combined 为v和norm_adv两张量和
class DQModel(tf.keras.Model):
    def __init__(self, hidden_size=128, num_actions=3):
        super(DQModel, self).__init__()
        self.dense1 = kl.Dense(hidden_size, activation='relu') # hidden_size=128
        self.dense2 = kl.Dense(hidden_size, activation='relu')
        self.adv_dense = kl.Dense(hidden_size, activation='relu')
        self.adv_out = kl.Dense(num_actions) # num_actions=3
        self.v_dense = kl.Dense(hidden_size, activation='relu')
        self.v_out = kl.Dense(1)
        '''
        tf.reduce_mean(): 根据给出的axis在input_tensor上求平均值。除非keep_dims为真，axis中的每个的张量秩会减少1。
        如果keep_dims为真，求平均值的维度的长度都会保持为1.如果不设置axis，所有维度上的元素都会被求平均值，并且只会返回一个只有一个元素的张量。
        '''
        self.lambda_layer = kl.Lambda(lambda x: x - tf.reduce_mean(x))
        self.combine = kl.Add() # add（）：直接对张量求和

    def call(self, input):
        x = self.dense1(input)
        x = self.dense2(x)
        adv = self.adv_dense(x)
        adv = self.adv_out(adv)
        v = self.v_dense(x)
        v = self.v_out(v)
        norm_adv = self.lambda_layer(adv)
        combined = self.combine([v, norm_adv]) # 实际为对v和norm_adv两张量求和
        return combined


class DQNAgent:
    # 定义超参数；设定model/train_log保存路径；初始化step/EPS；Huber loss function; Adam optimizer; main/target network
    def __init__(self, fn=None, lr=0.001, gamma=0.95, batch_size=32):
        # Coefficients are used for the loss terms.
        self.gamma = gamma # （时间）折扣因子discount factor
        self.lr = lr # 学习率 learning rate
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # 形如'20220331-123407'，得到具体的年月日-时分秒

        self.log_name = strftime("%Y%m%d_%H%M%S", localtime())

        self.checkpoint_dir = 'checkpoints/'
        self.model_name = 'DQN'
        self.model_dir = self.checkpoint_dir + self.model_name # 'checkpoints/DQN' - model保存路径
        self.log_dir = 'logs/'
        self.train_log_dir = self.log_dir + self.log_name # 'logs/DQN' - train_log（训练日志)保存路径
        self.create_log_dir() # 创建文件夹（文件路径）*4
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir) # 使用tf.summary记录数据 eg loss/step...
        self.fn = fn # ??
        self.EPS_START = 0.9 # 不知道代表episode还是epsilon (epsilon 可能是动态的)
        self.EPS_END = 0.5 #0.5
        self.steps_done = 0
        self.EPS_DECAY = 100 # ？？？
        self.steps_done = 0 # train的步数
        self.batch_size = batch_size # batch_size=32
        self.TAU = 0.08 # not known yet
        # Parameter updates
        self.loss = tf.keras.losses.Huber() # Huber损失函数
        self.optimizer = tf.optimizers.Adam(learning_rate=self.lr) # 优化器 Adam
        self.main_network = CarModel(num_actions=3, input_len=37) # 预测Q估计的网络，使用最新参数
        self.target_network = CarModel(num_actions=3, input_len=37) # 预测Q现实的网络，参数更新有延迟
        self.epsilon = 1

    # 创建存储model和train_log(训练日志)的文件路径
    def create_log_dir(self):
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir) # 创建 'logs/'
        if not os.path.exists(self.train_log_dir):
            os.mkdir(self.train_log_dir) # 创建 'logs/DQN'
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir) # 创建 'checkpoints/'
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir) # 创建 'checkpoints/DQN'

    # return action 返回动作决策（0/1/2）（0-保持当前车道，1-向右变道，2-向左变道）
    def act(self, state, main_network):
        # we need to do exploration vs exploitation
        # eps_threshold = 0.5 + (0.9 - 0.5) * math.exp(-1. * self.steps_done / 100)
        # Eg: steps_done = 1000 => eps_threshold = 0.500018159971905
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)

        # eps_threshold =  1 - 0.0001 * epoch


        # q_val = main_network.predict(np.expand_dims(state, axis=0)) # 获得一系列Q值
        # index = np.argmax(q_val)
        # q_val = q_val[0][index] # 预测输入state下最大的Q(s,a)值

        # np.random.rand()返回一个服从“0~1”均匀分布的随机样本值,随机样本取值范围是[0,1)，不包括1。
        if np.random.rand() < eps_threshold:
            action = random.randint(0, 2) # 随机选择动作（0-保持当前车道，1-向右变道，2-向左变道）
        else:
            action = main_network.predict(np.expand_dims(state, axis=0))
            action = np.argmax(action) # numpy中获取array的某一个维度中数值最大的那个元素的索引(索引值即为0，1，2)
        #return action, q_val
        return action

    # 获得minibatch（32组，包括：s,  a, r, s', terminal_flags），求得target_q（Q估计），return loss
    def train_step_(self, replay_memory):
        # 从replay_memory中取一个minibatch的数据 - 32组（包括：s,  a, r, s', terminal_flags-暂时不知道是什么））
        states, actions, rewards, new_states, terminal_flags = replay_memory.get_minibatch()
        # self.main_network = CarModel(num_actions=3, input_len=37)
        q_vals = self.main_network(new_states)
        actions = np.argmax(q_vals, axis=1)
        # The target network estimates the Q-values (in the next state s', new_states is passed!)
        # for every transition in the minibatch
        q_vals = self.target_network(new_states)
        # Bellman equation. Multiplication with (1-terminal_flags) makes sure that
        # if the game is over, targetQ=rewards
        # terminal_flags = True-1 or False-0
        q_vals = np.array([q_vals[num, action] for num, action in enumerate(actions)])
        # TD（0）
        target_q = rewards + (self.gamma * q_vals * (1 - terminal_flags)) # 更新target_q - Q估计
        loss = self.main_network.train_on_batch(states, target_q) # 在训练集数据的一批数据上进行训练，train_on_batch()返回loss
        return loss

    # 更新main_network参数（更新后main参数 = 0.092原main参数 + 0.008target参数）
    def update_network(self):
        # update target network parameters slowly from primary network
        # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的对象
        for t, e in zip(self.main_network.trainable_variables, self.target_network.trainable_variables):
            t.assign(t * (1 - self.TAU) + e * self.TAU) #

    def train(self, env, steps_per_epoch=128, epochs=2000):
        # Every four actions a gradient descend step is performed 每四个动作执行一个梯度下降步骤 不知何用，待定
        UPDATE_FREQ = 4
        # Number of chosen actions between updating the target network. 暂时未知，待定
        NETW_UPDATE_FREQ = 1000
        # Replay mem 暂时未知，待定
        REPLAY_MEMORY_START_SIZE = 33
        # Create network model
        self.main_network.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
        # Early stopping function

        # Replay memory 调用ReplayMemory()
        my_replay_memory = ReplayMemory()
        # Metrics
        loss_avg = tf.keras.metrics.Mean() # 求得loss平均值
        train_reward_tot = tf.keras.metrics.Sum() # reward均值（总reward，舒适度/效率/安全性 三种细分reward）
        train_rew_comf_tot = tf.keras.metrics.Sum()
        train_rew_eff_tot = tf.keras.metrics.Sum()
        train_rew_safe_tot = tf.keras.metrics.Sum()
        train_coll_rate = tf.keras.metrics.Mean() # 训练中碰撞率
        train_speed_rate = tf.keras.metrics.Mean() # 训练中速度

        # Training loop: collect samples, send to optimizer, repeat updates times.
        # 训练循环：收集样本，发送给优化器，重复更新次数
        next_obs = env.reset(gui=True, numVehicles=40) # 重置环境状态
        first_epoch = 0
        eps_list = []
        try:
            for epoch in range(first_epoch, epochs): # epoch = 0 - 9999
                ep_rewards = 0
                for step in range(steps_per_epoch): # step = 0 - 127
                    # curr state 当前状态，此时 obs(observation)即为state
                    state = next_obs.copy()
                    # 上一步的state
                    # q_val_last = self.act(state, self.main_network)[1]

                    # get action 得到动作决策
                    action = self.act(state, self.main_network)
                    # action = self.act(state, self.main_network)[0]
                    # do step 进行一步仿真，得到更新后状态/reward/是否结束仿真/是否碰撞
                    next_obs, rewards_info, done, collision = env.step(action)
                    # process obs and get rewards ego车速/ego当前车道允许的最大车速
                    avg_speed_perc = env.speed / env.target_speed
                    rewards_tot, R_comf, R_eff, R_safe = rewards_info

                    # calculate epsilon(s) under VEBD method
                    # q_val_curr = self.act(next_obs.copy(), self.main_network)[1]
                    # TD_ERROR = rewards_tot + self.gamma * q_val_curr - q_val_last
                    # TD_ERROR = 0.04 * TD_ERROR
                    # self.epsilon = 1 / 3 * ((1 - math.exp(-abs(TD_ERROR) / (1 + epoch))) / (1 + math.exp(-abs(TD_ERROR) / (1 + epoch)))) + 2 / 3 * self.epsilon
                    # eps_list.append(self.epsilon)

                    # Add experience 记忆库中增加一份experience(a,s',R_total,是否结束仿真)
                    my_replay_memory.add_experience(action=action,
                                                    frame=next_obs,
                                                    reward=rewards_tot,
                                                    terminal=done)
                    # Update metrics 包括：loss平均值，reward均值（总reward，舒适度/效率/安全性 三种细分reward），训练中碰撞率，训练中速度
                    train_reward_tot.update_state(rewards_tot)
                    train_rew_comf_tot.update_state(R_comf)
                    train_rew_eff_tot.update_state(R_eff)
                    train_rew_safe_tot.update_state(R_safe)
                    train_coll_rate.update_state(collision)
                    train_speed_rate.update_state(avg_speed_perc)

                    # Train every UPDATE_FREQ times 完成的steps > 33
                    if self.steps_done > REPLAY_MEMORY_START_SIZE:
                        loss_value = self.train_step_(my_replay_memory) # 得到loss值
                        loss_avg.update_state(loss_value) # 求出loss评价值
                        self.update_network() # 更新一次main_network参数
                    else:
                        loss_avg.update_state(-1)
                    # Copy network from main to target every NETW_UPDATE_FREQ
                    # 每经过 NETW_UPDATE_FREQ = 1000，将main复制到target
                    if step % NETW_UPDATE_FREQ == 0 and step > REPLAY_MEMORY_START_SIZE:
                        self.target_network.set_weights(self.main_network.get_weights())

                    self.steps_done += 1

                # Write train_log训练日志中存储训练结果，包括：loss平均值，reward均值（总reward，舒适度/效率/安全性 三种细分reward），训练中碰撞率，训练中速度
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss_avg.result(), step=epoch)
                    tf.summary.scalar('reward_tot', train_reward_tot.result(), step=epoch)
                    tf.summary.scalar('rewards_comf', train_rew_comf_tot.result(), step=epoch)
                    tf.summary.scalar('rewards_eff', train_rew_eff_tot.result(), step=epoch)
                    tf.summary.scalar('rewards_safe', train_rew_safe_tot.result(), step=epoch)
                    tf.summary.scalar('collission_rate', train_coll_rate.result(), step=epoch)
                    tf.summary.scalar('avg speed wrt maximum', train_speed_rate.result(), step=epoch)

                # Reset
                train_reward_tot.reset_states()
                train_rew_comf_tot.reset_states()
                train_rew_eff_tot.reset_states()
                train_rew_safe_tot.reset_states()
                train_coll_rate.reset_states()
                train_speed_rate.reset_states()

                # Save model 每100个epoch存一次model
                if epoch % 100 == 0:
                    tf.keras.models.save_model(self.main_network, self.model_dir + "/" + str(epoch) + "_main_network.hp5", save_format="h5")
                    tf.keras.models.save_model(self.target_network, self.model_dir + "/" + str(epoch) + "_target_network.hp5", save_format="h5")
        except KeyboardInterrupt:
            # self.model.save_weights(self.model_dir+"/model.ckpt")
            tf.keras.models.save_model(self.main_network, self.model_dir + "/" + str(epoch) + "_main_network.hp5", save_format="h5")
            tf.keras.models.save_model(self.target_network, self.model_dir + "/" + str(epoch) + "_target_network.hp5", save_format="h5")

        env.close()
        # store results under VDBE method
        save_eps(eps_list, 'Epsilon(s)_value_from_VDBE_α=0.04_epoch=2000.xlsx')

        return 0