"""Core classes."""
import numpy as np 
from keras.layers import (Activation, Conv2D, Dense, Flatten, Add, Concatenate, Permute)
from keras.engine.input_layer import Input
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras.layers.core import Dense, Flatten, Lambda
import tensorflow as tf
from random import randint
from PIL import Image
from keras.layers.merge import Add
#import deeprl_hw2.objectives


IMG_SIZE = None

def create_model_linear(output_shape, input_shape, model_name='q_linear_network'):
    model = Sequential()
    model.add(Dense(output_shape, input_shape=(input_shape,)))
    model.compile(optimizer = 'Adam', loss = tf.losses.huber_loss)
    return model

def create_model_conv(output_shape, input_shape, model_name='q_conv_network'):
    model = Sequential()
    model.add(Conv2D(32, 8, strides=(4, 4),
                              padding='valid',
                              activation='relu',
                              input_shape=input_shape,
                              data_format='channels_first'))
    model.add(Conv2D(64, 4, strides=(2, 2),
                              padding='valid',
                              activation='relu',
                              input_shape=input_shape,
                              data_format='channels_first'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(output_shape))
    model.compile(optimizer = 'Adam', loss = tf.losses.huber_loss)
    return model 

def create_model_duel(output_shape, input_shape):
    inputs = Input(shape=input_shape)
    net = Conv2D(32, 8, strides=(4, 4), data_format='channels_first', 
               activation='relu')(inputs)
    net = Conv2D(64, 4, strides=(2, 2), data_format='channels_first', 
               activation='relu')(net)
    net = Flatten()(net)
    advt = Dense(256, activation='relu')(net)
    advt = Dense(output_shape)(advt)
    value = Dense(256, activation='relu')(net)
    value = Dense(1)(value)
    advt = Lambda(lambda advt: advt - tf.reduce_mean(
      advt, axis=-1, keep_dims=True))(advt)
    value = Lambda(lambda value: tf.tile(value, [1, output_shape]))(value)
    final = Add()([value, advt])
    model = Model(
      inputs=inputs,
      outputs=final)
    model.compile(optimizer = 'Adam', loss = 'mse')
    return model

def create_model_framedifference(output_shape, input_shape1, input_shape2):
    # input_shape1 is the original four frame stack 4*84*84
    # input_shape2 should be three difference frame extracted 3*84*84
    frame = Input(shape = input_shape1)
    framediff = Input(shape = input_shape2)
    net_frame = Flatten()(frame)
    net_framediff = Conv2D(32, 8, strides = (4,4))(framediff)
    net_framediff = Conv2D(64, 4, strides = (3,3))(net_framediff)
    net_framediff = Conv2D(64, 4, strides = (3,3))(net_framediff)
    net_framediff = Flatten()(net_framediff)
    comb_frame = Concatenate()([net_frame, net_framediff])
    advt = Dense(256, activation='relu')(comb_frame)
    advt = Dense(output_shape)(advt)
    value = Dense(256, activation='relu')(net_frame)
    value = Dense(1)(value)
    advt = Lambda(lambda advt: advt - tf.reduce_mean(
      advt, axis=-1, keep_dims=True))(advt)
    value = Lambda(lambda value: tf.tile(value, [1, output_shape]))(value)
    final = Add()([value, advt])
    model = Model(
      inputs=[frame, framediff],
      outputs=final)
    model.compile(optimizer = 'Adam', loss = 'mse')
    #print(model.summary())
    return model
'''
def create_model_framedifference(output_shape, input_shape1, input_shape2):
    # input_shape1 is the original four frame stack 4*84*84
    # input_shape2 should be three difference frame extracted 3*84*84
    frameimg = Input(shape = input_shape1)
    framediff = Input(shape = input_shape2)
    rsn = ResNet50(include_top=False, weights='imagenet', input_tensor=frameimg, input_shape=input_shape1, pooling = None)
    #print(rsn.summary())
    for layer in rsn.layers:
        layer.trainable = False
    frame = rsn.layers[-1].output 
    frame = Flatten()(frame)
    net_framediff = Conv2D(32, 8, strides = (4,4))(framediff)
    net_framediff = Conv2D(64, 4, strides = (3,3))(net_framediff)
    net_framediff = Conv2D(64, 4, strides = (3,3))(net_framediff)
    net_framediff = Flatten()(net_framediff)
    advt = Dense(256, activation='relu')(net_framediff)
    advt = Dense(output_shape)(advt)
    value = Dense(256, activation='relu')(frame)
    value = Dense(1)(value)
    advt = Lambda(lambda advt: advt - tf.reduce_mean(
      advt, axis=-1, keep_dims=True))(advt)
    value = Lambda(lambda value: tf.tile(value, [1, output_shape]))(value)
    final = Add()([value, advt])
    model = Model(
      inputs=[frameimg, framediff],
      outputs=final)
    model.compile(optimizer = 'Adam', loss = 'mse')
    #print(model.summary())
    return model
'''

class Sample:
    """Represents a reinforcement learning sample.

    Used to store observed experience from an MDP. Represents a
    standard `(s, a, r, s', terminal)` tuple.

    Note: This is not the most efficient way to store things in the
    replay memory, but it is a convenient class to work with when
    sampling batches, or saving and loading samples while debugging.

    Parameters
    ----------
    state: array-like
      Represents the state of the MDP before taking an action. In most
      cases this will be a numpy array.
    action: int, float, tuple
      For discrete action domains this will be an integer. For
      continuous action domains this will be a floating point
      number. For a parameterized action MDP this will be a tuple
      containing the action and its associated parameters.
    reward: float
      The reward received for executing the given action in the given
      state and transitioning to the resulting state.
    next_state: array-like
      This is the state the agent transitions to after executing the
      `action` in `state`. Expected to be the same type/dimensions as
      the state.
    is_terminal: boolean
      True if this action finished the episode. False otherwise.
    """
    pass


class Preprocessor:
    """Preprocessor base class.

    Preprocessor can be used to perform some fixed operations on the
    raw state from an environment. For example, in ConvNet based
    networks which use image as the raw state, it is often useful to
    convert the image to greyscale or downsample the image.

    Preprocessors are implemented as class so that they can have
    internal state. This can be useful for things like the
    AtariPreproccessor which maxes over k frames.

    If you're using internal states, such as for keeping a sequence of
    inputs like in Atari, you should probably call reset when a new
    episode begins so that state doesn't leak in from episode to
    episode.
    """

    def process_state_for_network(self, state):
        global IMG_SIZE
        image = Image.fromarray(state, 'RGB').resize(IMG_SIZE)
        return np.asarray(image.getdata(), dtype=np.float64).reshape(image.size[0], image.size[1], 3)

    def process_state_for_memory(self, state):
        global IMG_SIZE
        image = Image.fromarray(state, 'RGB').resize(IMG_SIZE)
        return np.asarray(image.getdata(), dtype=int).reshape(image.size[0], image.size[1], 3)

    def process_batch(self, samples):
        """Process batch of samples.

        If your replay memory storage format is different than your
        network input, you may want to apply this function to your
        sampled batch before running it through your update function.

        Parameters
        ----------
        samples: list(tensorflow_rl.core.Sample)
          List of samples to process

        Returns
        -------
        processed_samples: list(tensorflow_rl.core.Sample)
          Samples after processing. Can be modified in anyways, but
          the list length will generally stay the same.
        """
        return samples

    def process_reward(self, reward):
        """Process the reward.

        Useful for things like reward clipping. The Atari environments
        from DQN paper do this. Instead of taking real score, they
        take the sign of the delta of the score.

        Parameters
        ----------
        reward: float
          Reward to process

        Returns
        -------
        processed_reward: float
          The processed reward
        """
        return reward

    def reset(self):
        """Reset any internal state.

        Will be called at the start of every new episode. Makes it
        possible to do history snapshots.
        """
        pass


class Memory:
    """Interface for replay memories.

    We have found this to be a useful interface for the replay
    memory. Feel free to add, modify or delete methods/attributes to
    this class.

    It is expected that the replay memory has implemented the
    __iter__, __getitem__, and __len__ methods.

    If you are storing raw Sample objects in your memory, then you may
    not need the end_episode method, and you may want to tweak the
    append method. This will make the sample method easy to implement
    (just ranomly draw saamples saved in your memory).

    However, the above approach will waste a lot of memory (as states
    will be stored multiple times in s as next state and then s' as
    state, etc.). Depending on your machine resources you may want to
    implement a version that stores samples in a more memory efficient
    manner.

    Methods
    -------
    append(state, action, reward, debug_info=None)
      Add a sample to the replay memory. The sample can be any python
      object, but it is suggested that tensorflow_rl.core.Sample be
      used.
    end_episode(final_state, is_terminal, debug_info=None)
      Set the final state of an episode and mark whether it was a true
      terminal state (i.e. the env returned is_terminal=True), of it
      is is an artificial terminal state (i.e. agent quit the episode
      early, but agent could have kept running episode).
    sample(batch_size, indexes=None)
      Return list of samples from the memory. Each class will
      implement a different method of choosing the
      samples. Optionally, specify the sample indexes manually.
    clear()
      Reset the memory. Deletes all references to the samples.
    """
    def __init__(self, max_size):
        """Setup memory.

        You should specify the maximum size o the memory. Once the
        memory fills up oldest values should be removed. You can try
        the collections.deque class as the underlying storage, but
        your sample method will be very slow.

        We recommend using a list as a ring buffer. Just track the
        index where the next sample should be inserted in the list.
        """
        # Here we didn't specify the state form
        # So when we sample from D, we need to first judge whether D contains empty dicts. 
        self.max_size = max_size
        self.last_element_index = 0
        self.fulfilled = 0
        self.memory_size = 0
        self.memory = list(np.zeros(max_size))

    def append(self, path):
        self.memory[self.last_element_index] = path
        self.last_element_index += 1
        if self.memory_size < self.max_size:
            self.memory_size += 1
        if self.last_element_index >= self.max_size:
            self.last_element_index = 0
            self.fulfilled = 1

    def end_episode(self, final_state, is_terminal):
        raise NotImplementedError('This method should be overridden')

    def sample(self, batch_size):
        batch = []
        if self.fulfilled == 0:
            for i in range(batch_size):
                batch.append(self.memory[randint(0,self.last_element_index-1)])
        else:
            for i in range(batch_size):
                batch.append(self.memory[randint(0,self.max_size-1)])
        return batch

    def save(self, parent_path):
        for i in range(self.memory_size):
            p = self.memory[i]
            o = open(parent_path+str(i)+'_image.txt', 'w')
            p['image_list'].tofile(o)
            o = open(parent_path+str(i)+'_image.txt', 'w')
            for item in p['image_list']:
                for v in item:
                    o.write(str(v))
                    o.write('\n')
                o.write('\n')
            o.close()
            o = open(parent_path+str(i)+'_diff.txt', 'w')
            for item in p['framediff_list']:
                o.write(str(item))
                o.write('\n')
            o.close()
            o = open(parent_path+str(i)+'_value.txt', 'w')
            for item in p['value_list']:
                o.write(str(item))
                o.write('\n')
            o.close()
            o = open(parent_path+str(i)+'_label.txt', 'w')
            for item in p['label_list']:
                o.write(str(item))
                o.write('\n')
            o.close()
            o = open(parent_path+str(i)+'_action.txt', 'w')
            for item in p['action_list']:
                o.write(str(item))
                o.write('\n')
            o.close()


    def clear(self):
        self.last_element_index = 0
        self.fulfilled = 0
        self.memory_size = 0
        self.memory = list(np.zeros(self.max_size))
