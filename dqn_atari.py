#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random
import gym
os.environ["CUDA_VISIBLE_DEVICES"]="6"

import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import load_model
from vae import ConvVAE
from dqn import DQNAgent
from core_ import Memory, Preprocessor
import core_
from cnn import CNN 

IMG_SIZE = (64, 64)
core_.IMG_SIZE = IMG_SIZE


import keras

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
config.gpu_options.per_process_gpu_memory_fraction = 0.1
config.gpu_options.allow_growth = True
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

'''
Logic: First we sample lots of paths. For each path, we keep the information about every step's scene, reward, done. 
Then we want to encode the scene with its reward information (discount reward), using VAE
with VAE trained, we then train the RNN to spread good/bad information to (s,a) pair and history
Use controller to get action from RNN. 
'''

# Hyperparameters for ConvVAE


def main():
    z_size=128
    batch_size=32
    learning_rate=0.0001
    vae = CNN(z_size=z_size,
              batch_size=batch_size,
              learning_rate=learning_rate,
              is_training=True,
              reuse=False,
              gpu_mode=False)
    env = 'Breakout-v0'
    done_reward = -1
    gamma = 0.95
    epsilon = 0.05
    save_freq = 10000
    batch_size = 32
    memory_size = 2000
    vae = vae
    rnn = None 
    controller = None
    memory = Memory(memory_size)
    preprocessor = Preprocessor()
    foresee_steps = 1
    DQA = DQNAgent(env = env,
        done_reward = done_reward,
        gamma = gamma,
        epsilon = epsilon,
        save_freq = save_freq,
        batch_size = batch_size,
        memory = memory,
        cnn = vae, 
        controller = controller, 
        preprocessor = preprocessor,
        foresee_steps = foresee_steps,
        #vae_path = 'tf_cnn/cnn.json',
        #rnn_path = 'tf_rnn/rnn.json',
        )
    '''
    tmp_reward = []
    for _ in range(10):
        _, reward = DQA.generate_path(is_random = False)
        tmp_reward.append(reward)
    print(np.mean(tmp_reward))
    '''
    '''
    path, _ = DQA.generate_path(is_random = False)
    DQA.test_vae(path)
    #DQA.test_rnn(path)
    #DQA.test_preprocess(path)
    '''
    o = open('results.txt','w')
    o.write('Start!')
    o.write('\n')
    S = []
    res = DQA.fill_memory(2000, is_random = True)
    S.append(res)
    DQA.train_vae()
    DQA.train_rnn()
    o.write(str(res))
    o.write('\n')
    o.close()
    for i in range(100):
        o = open('results.txt','a')
        DQA.memory.clear()
        res = DQA.fill_memory(300, is_random = False)
        o.write(str(res))
        o.write('\n')
        o.close()
        S.append(res)
        DQA.train_vae()
        DQA.train_rnn()
        print(S)
    
    '''
    DQA.train_controller()
    DQA.test_run()
    DQA.train(epochs)
    '''
main()
