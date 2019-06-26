"""Main DQN agent."""

import numpy as np
import time
from random import random, randrange
from numpy.random import randint, shuffle
from keras import backend as K
from keras.models import Model
import tensorflow as tf
import copy
import gym
from rnn import HyperParams, MDNRNN

MAX_LEN = 400

def default_hps():
  return HyperParams(num_steps=2000, # train model for 2000 steps.
                     max_seq_len=MAX_LEN, # train on sequences of 100
                     input_seq_width=128+4,    # width of our data (32 + 3 actions)
                     output_seq_width=128,    # width of our data is 32
                     rnn_size=256,    # number of rnn cells
                     batch_size=32,   # minibatch sizes
                     grad_clip=1.0,
                     num_mixture=5,   # number of mixtures in MDN
                     learning_rate=0.001,
                     decay_rate=1.0,
                     min_learning_rate=0.00001,
                     use_layer_norm=0, # set this to 1 to get more stable results (less chance of NaN), but slower
                     use_recurrent_dropout=0,
                     recurrent_dropout_prob=0.90,
                     use_input_dropout=0,
                     input_dropout_prob=0.90,
                     use_output_dropout=0,
                     output_dropout_prob=0.90,
                     is_training=1)

class DQNAgent:
    def __init__(self,env,done_reward,gamma,epsilon,save_freq,batch_size,memory,cnn,controller,preprocessor,foresee_steps=3,vae_path=None,rnn_path=None):
        self.env = gym.make(env)
        self.done_reward = done_reward
        self.gamma = gamma
        self.epsilon = epsilon
        self.save_freq = save_freq
        self.batch_size = batch_size
        self.memory = memory
        self.cnn = cnn
        self.controller = controller
        self.preprocessor = preprocessor
        self.foresee_steps = foresee_steps
        self.rnn_last_state_h, self.rnn_last_state_tuple = None, None
        self.mark = 0
        hps_model = default_hps()
        self.rnn = MDNRNN(hps_model)
        if vae_path != None:
            self.cnn.load_json(vae_path)
        if rnn_path != None:
            self.rnn.load_json(rnn_path)

    def select_action(self):
        N = self.env.action_space.n
        if len(self.path['image_list']) < 3:
            return np.random.randint(0, N)
        z = np.array(self.path['z_list'][-1]) # 32
        '''
        z_shape = np.array(self.path['z_list'][:-1]).shape
        value_list = np.array(self.path['action_list'])
        a_shape = value_list.shape
        raw_z = np.array(self.path['z_list'][:-1]).reshape(1, z_shape[0], z_shape[1])
        raw_a = value_list.reshape(1, a_shape[0], a_shape[1])
        inputs = np.concatenate((raw_z[:, :, :], raw_a[:, :, :]), axis=2)
        state = np.array(self.rnn_last_state_tuple)
        '''
        #S1, S2, Z = [], [], []
        Z = []
        act = np.zeros([N, N]).astype(np.float32)
        for i in range(N):
            act[i][i] = 1
            #S1.append(self.rnn_last_state_tuple[0])
            #S2.append(self.rnn_last_state_tuple[1])
            Z.append(z)
        Z = np.array(Z)
        if self.mark:
            s1, s2 = self.rnn_last_state_tuple.c, self.rnn_last_state_tuple.h
        if not self.mark:
            s1 = np.array(self.rnn_last_state_tuple[0])
            s2 = np.array(self.rnn_last_state_tuple[1])
            self.mark = 1
        S1 = [s1, s1, s1, s1]
        S2 = [s2, s2, s2 ,s2]
        input_x = np.concatenate([Z, act], axis = 1)
        '''
        print(input_x.shape)
        print(S1[0])
        print(np.array(S1))
        print(S1.shape)
        print(S2.shape)
        '''
        feed = {self.rnn.input_single: input_x, self.rnn.state_c: S1, self.rnn.state_h: S2}
        output, state_tuples = self.rnn.sess.run([self.rnn.output_batch, self.rnn.state_tuples], feed)
        '''
        S_c, S_h = [], []
        Z_NN, act_NN = [], []
        statec, stateh = state_tuples.c, state_tuples.h
        for i in range(N):
            state_c, state_h = statec[i], stateh[i]
            for j in range(N):
                tmp = np.zeros(N).astype(np.float32)
                tmp[j] = 1
                S_c.append(state_c)
                S_h.append(state_h)
                Z_NN.append(output[i])
                act_NN.append(tmp)
        input_x = np.concatenate([Z_NN, act_NN], axis = 1)
        feed = {self.rnn.input_single: input_x, self.rnn.state_c: S_c, self.rnn.state_h: S_h}
        output, _ = self.rnn.sess.run([self.rnn.output_batch, self.rnn.state_tuples], feed)
        '''

        # output --- 4 * batch_size
        #print(state_tuples)
        # Input to vae and see the results --- to be implemented. 
        feed = {self.cnn.z_1: output} 
        #print(output)
        tq_pred = self.cnn.sess.run(self.cnn.tq_pred, feed)
        tq_pred = tq_pred.reshape(N)
        #print(tq_pred)
        #print(tq_pred)
        '''
        ind_r, ind_d, rew, die = 0, 0, 0.0, 1.0
        for i in range(N):
            if tq_pred[i][1] > rew:
                rew = max(rew, tq_pred[i][1])
                ind_r = i
            if tq_pred[i][2] < die:
                die = min(die, tq_pred[i][2])
                ind_d = i
        if rew > 0.1:
            ans = ind_r 
        else:
            ans = ind_d
        #print(state_tuples)
        '''
        ans = np.argmax(tq_pred)
        #ans = ans // N
        if np.random.rand() < 0.05:
            ans = np.random.randint(0, N)
        state_c, state_h = state_tuples.c, state_tuples.h
        self.rnn_last_state_tuple = tf.nn.rnn_cell.LSTMStateTuple(state_c[ans], state_h[ans])
        #print(self.rnn_last_state_tuple)
        #self.rnn_last_state_h = output[ans]
        return ans

    def select_action_new(self):
        N = self.env.action_space.n
        if len(self.path['image_list']) < 3:
            return np.random.randint(0, N)
        z = np.array(self.path['z_list'][-1]) # 32
        z_shape = np.array(self.path['z_list'][:-1]).shape
        value_list = np.array(self.path['action_list'])
        a_shape = value_list.shape
        raw_z = np.array(self.path['z_list'][:-1]).reshape(1, z_shape[0], z_shape[1])
        raw_a = value_list.reshape(1, a_shape[0], a_shape[1])
        inputs = np.concatenate((raw_z[:, :, :], raw_a[:, :, :]), axis=2)
        state = np.array(self.rnn_last_state_tuple)
        #S1, S2, Z = [], [], []
        Z = []
        act = np.zeros([N, N]).astype(np.float32)
        for i in range(N):
            act[i][i] = 1
            #S1.append(self.rnn_last_state_tuple[0])
            #S2.append(self.rnn_last_state_tuple[1])
            Z.append(z)
        Z = np.array(Z)
        if self.mark:
            s1, s2 = self.rnn_last_state_tuple.c, self.rnn_last_state_tuple.h
        if not self.mark:
            s1 = np.array(self.rnn_last_state_tuple[0])
            s2 = np.array(self.rnn_last_state_tuple[1])
            self.mark = 1
        S1 = [s1, s1, s1, s1]
        S2 = [s2, s2, s2 ,s2]
        input_x = np.concatenate([Z, act], axis = 1)
        '''
        print(input_x.shape)
        print(S1[0])
        print(np.array(S1))
        print(S1.shape)
        print(S2.shape)
        '''
        feed = {self.rnn.input_single: input_x, self.rnn.state_c: S1, self.rnn.state_h: S2}
        output, state_tuples = self.rnn.sess.run([self.rnn.output_batch, self.rnn.state_tuples], feed)
        #print(state_tuples)
        # Input to vae and see the results --- to be implemented. 
        tq_pred = output.reshape(4)
        tq_pred[1] = -10
        '''
        print(tq_pred)
        ind_r, ind_d, rew, die = 0, 0, 0.0, 1.0
        for i in range(N):
            if tq_pred[i][1] > rew:
                rew = max(rew, tq_pred[i][1])
                ind_r = i
            if tq_pred[i][2] < die:
                die = min(die, tq_pred[i][2])
                ind_d = i
        if rew > 0.1:
            ans = ind_r 
        else:
            ans = ind_d
        #print(state_tuples)
        '''
        #print(tq_pred)
        ans = np.argmax(tq_pred)
        state_c, state_h = state_tuples.c, state_tuples.h
        self.rnn_last_state_tuple = tf.nn.rnn_cell.LSTMStateTuple(state_c[ans], state_h[ans])
        #print(self.rnn_last_state_tuple)
        #self.rnn_last_state_h = output[ans]
        return ans

    def generate_path(self, is_random = True):
        global MAX_LEN
        self.env.reset()
        self.rnn_last_state_tuple = (np.zeros([256]), np.zeros([256])) #self.rnn.cell.zero_state(batch_size=self.env.action_space.n, dtype=tf.float32) 
        self.mark = 0
        done = 0
        score = 0
        reward = 0
        life_cnt = 5
        life_current = 5
        self.path = {'image_list': [], 'value_list': [], 'label_list':[], 'framediff_list': [], 'z_list': [], 'action_list': []}
        obs, reward_, done_, info_ = self.env.step(1)
        done = max(done, done_)
        obs = self.preprocessor.process_state_for_memory(obs)
        reward += reward_
        for _ in range(2):
            action = 0
            obs_, reward_, done_, info = self.env.step(action)
            obs_ = self.preprocessor.process_state_for_memory(obs_)
            obs = np.maximum(obs, obs_)
            reward += reward_
            done = max(done, done_)
        current_state = obs
        framediff = np.zeros(current_state.shape)
        score += reward
        choose_new = False
        step_cnt = 0
        while not done:
            step_cnt += 1
            life_cnt = life_current
            self.path['image_list'].append(current_state)
            self.path['framediff_list'].append(framediff)
            if not is_random:
                tmp_state = np.array([current_state]).astype(np.float)
                tmp_state = tmp_state/255.0
                diff_ = np.array([framediff]).astype(np.float)
                diff_ = diff_/255.0
                feed = {self.cnn.x: tmp_state, self.cnn.diff: diff_}
                z_1 = np.array(self.cnn.sess.run(self.cnn.z_1, feed)[0]) # 1*32
                self.path['z_list'].append(z_1)
                action = self.select_action()
            else:
                action = randint(0, self.env.action_space.n)
            if choose_new:
                self.env.step(1)
                self.env.step(1)
                choose_new = False
            tmp = np.zeros(4)
            tmp[action] = 1
            reward = 0
            obs, reward_, done_, info = self.env.step(action)
            obs = self.preprocessor.process_state_for_memory(obs)
            reward += reward_
            done = max(done, done_)
            life_current = int(info['ale.lives'])
            
            for _ in range(1):
                obs_, reward_, done_, info = self.env.step(0)
                obs_ = self.preprocessor.process_state_for_memory(obs_)
                obs = np.maximum(obs, obs_)
                reward += reward_
                done = max(done, done_)
                life_current = int(info['ale.lives'])
            for _ in range(1):
                obs_, reward_, done_, info = self.env.step(1)
                obs_ = self.preprocessor.process_state_for_memory(obs_)
                obs = np.maximum(obs, obs_)
                reward += reward_
                done = max(done, done_)
                life_current = int(info['ale.lives'])
            
            next_state = obs 
            value = []
            score += reward
            if life_current < life_cnt:
                reward += self.done_reward
                choose_new = True
            value.append(int(reward))
            value.append(int(done))
            value.append(int(action))
            self.path['value_list'].append(value)
            framediff = next_state - current_state
            current_state = next_state
            self.path['action_list'].append(tmp)
        tmp_reward = 0.0
        L = len(self.path['value_list'])
        for i in range(L):
            if self.path['value_list'][i][0] > 0:
                self.path['label_list'].append([0, 1, 0])
                for j in range(1, 13):
                    self.path['label_list'][i-j] = [0, 1, 0]
            if self.path['value_list'][i][0] == 0:
                self.path['label_list'].append([1, 0, 0])
            if self.path['value_list'][i][0] < 0:
                self.path['label_list'].append([0, 0, 1])
                for j in range(1, 5):
                    self.path['label_list'][i-j] = [0, 0, 1]
        for i in range(L-1, -1, -1):
            if self.path['value_list'][i][0] == self.done_reward:
                tmp_reward = 0.0
            tmp_reward = tmp_reward * self.gamma
            self.path['value_list'][i].append(self.path['value_list'][i][0] + tmp_reward)
            tmp_reward += self.path['value_list'][i][0]
        self.path['image_list'], self.path['value_list'], self.path['label_list'], self.path['framediff_list'] = np.array(self.path['image_list']), np.array(self.path['value_list']), np.array(self.path['label_list']), np.array(self.path['framediff_list'])
        # Here we use temporary expected value for label
        self.path['label_list'] = self.path['value_list'][:,-1]
        #print(self.path['label_list'])
        print('Score', score)
        #MAX_LEN = max(MAX_LEN, step_cnt)
        return self.path, score

    def generate_image(self, obs, diff):
        obs = np.array(obs).astype(float)
        obs = obs/255.0
        diff = np.array(diff).astype(float)
        diff = diff/255.0
        feed = {self.cnn.x: np.array([obs]), self.cnn.diff: np.array([diff])}
        tq_pred, latent_1= self.cnn.sess.run([self.cnn.tq_pred, self.cnn.z_1], feed)
        print(latent_1)
        return tq_pred

        '''
        feed = {self.vae.x: np.array([obs]).astype(np.float), self.vae.frame_diff: np.array([diff]).astype(np.float), self.vae.true_tq: [[0, 0, 0]]}
        obs_pred, diff_pred, tq_pred, latent_1, latent_2 = self.vae.sess.run([self.vae.y_1, self.vae.y_2, self.vae.tq_pred, self.vae.z_1, self.vae.z_2], feed)
        obs_pred = obs_pred[0]
        diff_pred = diff_pred[0]
        tq_pred = tq_pred[0]
        print(latent_1)
        print(latent_2)
        return (obs_pred, diff_pred, tq_pred)
        '''

    def test_preprocess(self, path):
        import matplotlib.pyplot as plt 
        image_list, value_list = path['image_list'], path['value_list'][:,-1]
        for i in range(len(image_list)):
            name = str('test/') + str(i) + '_image_' + str(value_list[i]) +'.png'
            plt.imsave(name, image_list[i])

    def test_vae(self, path):
        import matplotlib.pyplot as plt 
        from PIL import Image
        image_list = path['image_list']
        framediff_list = path['framediff_list']
        for i in range(len(image_list)):
            tq_pred = self.generate_image(image_list[i], framediff_list[i])
            name2 = str('test/') + str(i) + '_image_' + str(tq_pred)+'.png'
            plt.imsave(name2, image_list[i])
            '''
            obs_pred, diff_pred, tq_pred = self.generate_image(image_list[i], framediff_list[i])
            name1 = str('test/') + str(i) +'.png'
            plt.imsave(name1, image_list[i])
            name2 = str('test/') + str(i) + '_image_' + str(tq_pred)+'.png'
            plt.imsave(name2, obs_pred)
            name3 = str('test/') + str(i) + '_diff_' + str(tq_pred)+'.png'
            plt.imsave(name3, diff_pred.reshape(16, 16 ,3))
            name4 = str('test/') + str(i) + '_diff_' + '.png'
            plt.imsave(name4, framediff_list[i])
            '''
    def test_rnn(self, path):
        import matplotlib.pyplot as plt 
        image_list = path['image_list']
        for i in range(len(image_list)):
            name2 = str('test/') + str(i) +'.png'
            plt.imsave(name2, image_list[i])


    def train_vae(self):
        batch_size = self.cnn.batch_size
        for _ in range(10):
            for i in range(self.memory.memory_size):
                path = self.memory.memory[i]
                L = len(path['value_list'])
                image_list, framediff_list, label_list = path['image_list'], path['framediff_list'], path['label_list']
                ind = np.linspace(0, L-1, L).astype(np.int)
                shuffle(ind)
                num_epochs = int(L/batch_size)
                for j in range(num_epochs):
                    image = image_list[ind[j*batch_size:(j+1)*batch_size]]
                    #print(image)
                    #print(image.shape)
                    image = image.astype(np.float)
                    image = image/255.0
                    
                    framediff = framediff_list[ind[j*batch_size:(j+1)*batch_size]]
                    framediff = framediff.astype(np.float)
                    framediff = framediff/255.0
                    
                    tq = label_list[ind[j*batch_size:(j+1)*batch_size]]
                    tq = tq.astype(np.float)
                    tq = tq.reshape((len(tq), 1))
                    #tq = tq/100.0
                    #print(tq)
                    #feed = {self.cnn.x: image, self.cnn.true_tq: tq}
                    feed = {self.cnn.x: image, self.cnn.diff: framediff, self.cnn.true_tq: tq}
                    max_loss, tq_loss, train_step, _ = self.cnn.sess.run([self.cnn.max_loss_1, self.cnn.tq_loss, self.cnn.global_step, self.cnn.train_op], feed)
                    if ((train_step+1) % 200 == 0):
                        print("step", (train_step+1), max_loss, tq_loss)
                    if ((train_step+1) % 200 == 0):
                        self.cnn.save_json("tf_cnn/cnn.json")
                    '''
                    train_loss, r_loss, tq_loss, diff_loss, train_step, _ = self.vae.sess.run([self.vae.loss, self.vae.r_loss, self.vae.tq_loss, self.vae.diff_loss, self.vae.global_step, self.vae.train_op], feed)
                    if ((train_step+1) % 500 == 0):
                        print("step", (train_step+1), train_loss, r_loss, tq_loss, diff_loss)
                    if ((train_step+1) % 2000 == 0):
                        self.vae.save_json("tf_vae/vae.json")
                    '''

    def train_rnn(self):
        global MAX_LEN
        start = time.time()
        batch_size = 32
        #MAX_LEN = (MAX_LEN // batch_size + 1) * batch_size
        hps_model = default_hps()
        hps = hps_model
        #self.rnn = MDNRNN(hps_model)
        print('MAX Length: ', MAX_LEN)
        for _ in range(50):
            N = self.memory.memory_size
            ind = np.linspace(0, N-1, N).astype(np.int)
            shuffle(ind)
            num_epochs = int(N/batch_size)
            for j in range(num_epochs):
                step = self.rnn.sess.run(self.rnn.global_step)
                curr_learning_rate = (hps.learning_rate-hps.min_learning_rate) * (hps.decay_rate) ** step + hps.min_learning_rate
                raw_z, raw_a, raw_tq = [], [], []
                for i in range(batch_size):
                    p = self.memory.memory[ind[j*batch_size+i]]
                    L = len(p['image_list'])
                    num = min(L, MAX_LEN) // batch_size
                    I, T = [], []
                    for k in range(num):
                        imgtmp = p['image_list'][k*batch_size:(k+1)*batch_size].astype(np.float)
                        imgtmp = imgtmp/255.0
                        difftmp = p['framediff_list'][k*batch_size:(k+1)*batch_size].astype(np.float)
                        difftmp = difftmp/255.0
                        feed = {self.cnn.x: imgtmp, self.cnn.diff: difftmp}
                        '''
                        imgtmp = imgtmp.reshape(batch_size, -1)
                        difftmp = difftmp.reshape(batch_size, -1)
                        print(imgtmp.shape)
                        I.append(np.concatenate([imgtmp, difftmp], axis = 1))
                        '''
                        tmp, label = self.cnn.sess.run([self.cnn.z_1, self.cnn.tq_pred], feed)
                        #print(tmp.shape)
                        I.append(tmp)
                        T.append(label)
                    '''
                    if L > num*batch_size:
                        feed = {self.cnn.x: p['image_list'][num*batch_size:]}
                        tmp = self.cnn.sess.run(self.cnn.z_1, feed)
                        I.append(tmp)
                    '''
                    l = len(I) * batch_size
                    if l < MAX_LEN:
                        t = np.zeros((MAX_LEN - l, self.cnn.z_size))
                        #t = np.zeros((MAX_LEN-l, 64*64*3*2))
                        tt = np.zeros((MAX_LEN-l, 1))
                        #print(t.shape)
                        I.append(t)
                        T.append(tt)
                        raw_a.append(np.concatenate([p['action_list'][:l], np.zeros((MAX_LEN-l, 4))]))
                    image = np.concatenate(I, axis = 0)
                    label = np.concatenate(T, axis = 0)
                    #print(image.shape)
                    #print(label.shape)
                    raw_z.append(image)
                    raw_tq.append(label)
                #print(len(raw_z), len(raw_z[0]), len(raw_z[0][0]))
                #print(len(raw_a), len(raw_a[0]), len(raw_a[0][0]))
                '''
                min_len = 1000
                for i in range(batch_size):
                    min_len = min(min_len, len(raw_z[i]))
                for i in range(batch_size):
                    raw_z[i], raw_a[i] = raw_z[i][:min_len], raw_a[i][:min_len]
                '''
                raw_z, raw_a, raw_tq = np.array(raw_z), np.array(raw_a), np.array(raw_tq)
                #print(raw_z.shape)
                #print(raw_a.shape)
                try:
                    inputs = np.concatenate((raw_z[:, :-self.foresee_steps, :], raw_a[:, :-self.foresee_steps, :]), axis=2)
                except ValueError:
                    continue
                inputs = np.concatenate((raw_z[:, :-self.foresee_steps, :], raw_a[:, :-self.foresee_steps, :]), axis=2)
                outputs = raw_z[:, self.foresee_steps:, :] # teacher forcing (shift by one predictions)
                outputs_tq = raw_tq[:, self.foresee_steps:, :]
                feed = {self.rnn.input_x: inputs, self.rnn.output_x: outputs, self.rnn.output_tq:outputs_tq, self.rnn.lr: curr_learning_rate}
                (change_loss, state, train_step, _) = self.rnn.sess.run([self.rnn.change_loss, self.rnn.final_state, self.rnn.global_step, self.rnn.train_op], feed)
                if (step%20==0 and step > 0):
                    end = time.time()
                    time_taken = end-start
                    start = time.time()
                    output_log = "step: %d, lr: %.6f, change_cost: %.4f, train_time_taken: %.4f" % (step, curr_learning_rate, change_loss, time_taken)
                    self.rnn.save_json("tf_rnn/rnn.json")
                    print(output_log)


    def fill_memory(self, itrs, is_random = False):
        s_tmp = []
        for _ in range(itrs):
            path, score = self.generate_path(is_random = is_random)
            s_tmp.append(score)
            self.memory.append(path)
            if (self.memory.memory_size + 1) % 50 == 0:
                print(self.memory.memory_size)
        return np.mean(s_tmp)
        #self.memory.save('./path/')
