# ConvVAE model

import numpy as np
import json
import tensorflow as tf
import os

def reset_graph():
  if 'sess' in globals() and sess:
    sess.close()
  tf.reset_default_graph()

class CNN(object):
  def __init__(self, z_size=32, batch_size=1, learning_rate=0.0001, is_training=False, reuse=False, gpu_mode=False):
    self.z_size = z_size
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.is_training = is_training
    self.reuse = reuse
    with tf.variable_scope('conv_vae', reuse=self.reuse):
      if not gpu_mode:
        with tf.device('/cpu:0'):
          tf.logging.info('Model using cpu.')
          self._build_graph()
      else:
        tf.logging.info('Model using gpu.')
        self._build_graph()
    self._init_session()

  def normalize_with_moments(self, x, axes=[0, 1], epsilon=1e-8):
    mean, variance = tf.nn.moments(x, axes=axes)
    x_normed = (x - mean) / tf.sqrt(variance + epsilon) # epsilon to avoid dividing by zero
    return x_normed


  def _build_graph(self):
    self.g = tf.Graph()
    with self.g.as_default():

      self.x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
      self.diff = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
      self.true_tq = tf.placeholder(tf.float32, shape = [None, 1])

      # Encoder 1
      h_1 = tf.layers.conv2d(self.x, 128, 3, strides=1, activation=tf.nn.relu, name="enc_conv1_1")
      h_1 = tf.layers.max_pooling2d(h_1, 2, 2)
      h_1 = tf.layers.conv2d(h_1, 64, 3, strides=2, activation=tf.nn.relu, name="enc_conv4_1")
      h_1 = tf.layers.Flatten()(h_1)
      h_1 = tf.layers.dense(h_1, 128)

      # CNN for framediff
      h_2 = tf.layers.conv2d(self.diff, 128, 3, strides=1)
      h_2 = tf.layers.average_pooling2d(h_2, 2, 2)
      h_2 = tf.layers.conv2d(h_2, 64, 3, strides = 2)
      h_2 = tf.layers.Flatten()(h_2)
      h_2 = tf.layers.dense(h_2, 128)
      # VAE
      #self.z_nonorm = tf.layers.dense(h_1, self.z_size, activation = tf.nn.relu, name="enc_fc_latent_1")
      #self.z_1 = tf.layers.batch_normalization(self.z_nonorm)
      h = tf.concat([h_1, h_2], axis = 1)
      self.z_1 = tf.layers.dense(h, self.z_size, activation = tf.nn.relu, name="enc_fc_latent_1")
      self.tq_pred = tf.layers.dense(self.z_1, 1)
      '''
      h_1 = tf.nn.dropout(self.z_1, keep_prob = 0.5)
      self.tq = tf.layers.dense(h_1, 3, activation = tf.nn.relu)
      self.tq_pred = tf.nn.softmax(self.tq)
      '''

      if self.is_training:
        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        eps = 1e-6 # avoid taking log of zero
        
        #self.tq_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.true_tq, logits = self.tq))
        self.tq_loss = tf.reduce_mean(tf.square(self.tq_pred - self.true_tq))
        self.max_loss_1 = tf.reduce_mean(tf.reduce_mean(self.z_1, 1))
        max_loss_1 = tf.maximum(self.max_loss_1, 1)
        self.loss = self.tq_loss #+ max_loss_1

        
        # training
        self.lr = tf.Variable(self.learning_rate, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        grads = self.optimizer.compute_gradients(self.loss) # can potentially clip gradients here.

        self.train_op = self.optimizer.apply_gradients(
          grads, global_step=self.global_step, name='train_step')

      # initialize vars
      self.init = tf.global_variables_initializer()
      
      t_vars = tf.trainable_variables()
      self.assign_ops = {}
      for var in t_vars:
        #if var.name.startswith('conv_vae'):
        pshape = var.get_shape()
        pl = tf.placeholder(tf.float32, pshape, var.name[:-2]+'_placeholder')
        assign_op = var.assign(pl)
        self.assign_ops[var] = (assign_op, pl)


  def _init_session(self):
    """Launch TensorFlow session and initialize variables"""
    self.sess = tf.Session(graph=self.g)
    self.sess.run(self.init)
  def close_sess(self):
    """ Close TensorFlow session """
    self.sess.close()
  def encode(self, x):
    return self.sess.run(self.z, feed_dict={self.x: x})
  def encode_mu_logvar(self, x):
    (mu, logvar) = self.sess.run([self.mu, self.logvar], feed_dict={self.x: x})
    return mu, logvar
  def decode(self, z):
    return self.sess.run(self.y, feed_dict={self.z: z})
  def get_model_params(self):
    # get trainable params.
    model_names = []
    model_params = []
    model_shapes = []
    with self.g.as_default():
      t_vars = tf.trainable_variables()
      for var in t_vars:
        #if var.name.startswith('conv_vae'):
        param_name = var.name
        p = self.sess.run(var)
        model_names.append(param_name)
        params = np.round(p*10000).astype(np.int).tolist()
        model_params.append(params)
        model_shapes.append(p.shape)
    return model_params, model_shapes, model_names
  def get_random_model_params(self, stdev=0.5):
    # get random params.
    _, mshape, _ = self.get_model_params()
    rparam = []
    for s in mshape:
      #rparam.append(np.random.randn(*s)*stdev)
      rparam.append(np.random.standard_cauchy(s)*stdev) # spice things up
    return rparam
  def set_model_params(self, params):
    with self.g.as_default():
      t_vars = tf.trainable_variables()
      idx = 0
      for var in t_vars:
        #if var.name.startswith('conv_vae'):
        pshape = tuple(var.get_shape().as_list())
        p = np.array(params[idx])
        assert pshape == p.shape, "inconsistent shape"
        assign_op, pl = self.assign_ops[var]
        self.sess.run(assign_op, feed_dict={pl.name: p/10000.})
        idx += 1
  def load_json(self, jsonfile='vae.json'):
    with open(jsonfile, 'r') as f:
      params = json.load(f)
    self.set_model_params(params)
  def save_json(self, jsonfile='vae.json'):
    model_params, model_shapes, model_names = self.get_model_params()
    qparams = []
    for p in model_params:
      qparams.append(p)
    with open(jsonfile, 'wt') as outfile:
      json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))
  def set_random_params(self, stdev=0.5):
    rparam = self.get_random_model_params(stdev)
    self.set_model_params(rparam)
  def save_model(self, model_save_path):
    sess = self.sess
    with self.g.as_default():
      saver = tf.train.Saver(tf.global_variables())
    checkpoint_path = os.path.join(model_save_path, 'vae')
    tf.logging.info('saving model %s.', checkpoint_path)
    saver.save(sess, checkpoint_path, 0) # just keep one
  def load_checkpoint(self, checkpoint_path):
    sess = self.sess
    with self.g.as_default():
      saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(checkpoint_path)
    print('loading model', ckpt.model_checkpoint_path)
    tf.logging.info('Loading model %s.', ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

