import numpy as np
import tensorflow as tf

from encoder import Encoder
from decoder import Decoder

import bisect
from time import time

class PointerNet(object):
    def __init__(self, vsize, esize, hsize, asize, buckets, **kwargs):
        super(PointerNet, self).__init__()

        self.name  = kwargs.get('name', self.__class__.__name__)
        self.scope = kwargs.get('scope', self.name)

        self.enc_vsize = vsize
        self.enc_esize = esize
        self.enc_hsize = hsize

        self.dec_msize = self.enc_hsize * 2  # concatenation of bidirectional RNN
        self.dec_isize = kwargs.get('dec_isize', self.enc_hsize)
        self.dec_hsize = hsize
        self.dec_asize = asize

        self.buckets = buckets

        self.encoder = Encoder(self.enc_vsize, self.enc_esize, self.enc_hsize)
        self.decoder = Decoder(self.dec_isize, self.dec_hsize, self.dec_msize, self.dec_asize, self.buckets[-1])

    def __call__(self, enc_input, dec_input_indices, child_indices, head_indices, rewards):
        batch_size = tf.shape(enc_input)[0]
        # forward computation graph
        with tf.variable_scope(self.scope):
            # encoder output
            enc_memory, _ = self.encoder(enc_input)

            # padding the memory with a dummy (all-zero) vector at the end
            enc_memory = tf.pad(enc_memory, [[0,0],[0,1],[0,0]])

            # decoder
            dec_hiddens, dec_actions, dec_act_probs = self.decoder(dec_input_indices, enc_memory, child_indices, head_indices, rewards)

            # cost
            costs = []
            for act_prob, reward in zip(dec_act_probs, rewards):
                costs.append(tf.reduce_sum(act_prob * reward))

        # gradient computation graph
        pointer_net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        grad_params_buckets = []
        for limit in self.buckets:
            print '0 ~ %d' % (limit-1)
            grad_params = tf.gradients(costs[:limit], pointer_net_params)
            grad_params_buckets.append(grad_params)

        return dec_hiddens, dec_actions, grad_params_buckets

#### test script
if __name__ == '__main__':
    # hyper-parameters
    vsize = 1000
    esize = 256
    hsize = 256
    asize = 256
    isize = 333

    buckets = [10]#, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    max_len = buckets[-1]

    ####################
    # symbolic section
    ####################
    
    # model initialization
    pointer_net = PointerNet(vsize, esize, hsize, asize, buckets, dec_isize=isize)

    # placeholders
    enc_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='enc_input')

    inputs, rewards = [], []
    cindices, hindices = [], []
    for i in range(max_len):
        inputs.append(tf.placeholder(dtype=tf.int32, shape=[None, 2], name='input_%d'%i))
        rewards.append(tf.placeholder(dtype=tf.float32, name='reward_%d'%i))
        cindices.append(tf.placeholder(dtype=tf.int32, name='child_indice_%d'%i))
        hindices.append(tf.placeholder(dtype=tf.int32, name='head_indice_%d'%i))

    # build computation graph
    dec_hiddens, dec_actions, grad_params_buckets = pointer_net(enc_input, inputs, cindices, hindices, rewards)

    ####################
    # run-time section
    ####################

    lsize = 10
    bsize = 32

    all_feeds = []
    all_feeds.extend(inputs)
    all_feeds.extend(rewards)
    all_feeds.extend(hindices)
    all_feeds.extend(cindices)
    all_feeds.append(enc_input)

    all_fetches = []
    all_fetches.extend(dec_hiddens)
    all_fetches.extend(dec_actions)

    # get from evironment
    def action(output):
        input  = np.repeat(np.array([3,6]).astype(np.int32).reshape(1, -1), bsize, axis=0)
        reward = np.ones(output.shape)
        h_indices = np.repeat(np.arange(lsize).astype(np.int32).reshape(1, -1), bsize, axis=0)
        c_indices = np.repeat(np.arange(lsize).astype(np.int32).reshape(1, -1), bsize, axis=0)
        return input, reward, h_indices, c_indices

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        
        enc_input_np = np.random.randint(0, vsize, size=[bsize, lsize]).astype(np.int32)

        init_input_np = np.repeat(np.arange(2).astype(np.int32).reshape(1, -1), bsize, axis=0)
        init_hidx_np = np.repeat(np.arange(lsize).astype(np.int32).reshape(1, -1), bsize, axis=0)
        init_cidx_np = np.repeat(np.arange(lsize).astype(np.int32).reshape(1, -1), bsize, axis=0)

        bucket_id = bisect.bisect_left(buckets, lsize)
        grad_w = grad_params_buckets[bucket_id]

        ##############################
        hiddens_np, dec_actions_np, inputs_np, rewards_np, hindices_np, cindices_np = [], [], [], [], [], []
        inputs_np.append(init_input_np)
        hindices_np.append(init_hidx_np)
        cindices_np.append(init_cidx_np)
        t = time()
        feed_dict={enc_input:enc_input_np}
        for i in range(lsize):
            # t_i = time()
            feed_dict.update({inputs[i]:inputs_np[i], hindices[i]:hindices_np[i], cindices[i]:cindices_np[i]})
            
            h_i_np, a_i_np = sess.run([dec_hiddens[i], dec_actions[i]], feed_dict=feed_dict)

            hiddens_np.append(h_i_np)
            dec_actions_np.append(a_i_np)
            input_i_np, reward_i_np, hidx_np, cidx_np = action(dec_actions_np[i])
            rewards_np.append(reward_i_np)
            inputs_np.append(input_i_np)
            hindices_np.append(hidx_np)
            cindices_np.append(cidx_np)
            # print i, time() - t_i
        print time() - t
        
        t = time()
        feed_dict.update({go:go_np for go, go_np in zip(rewards, rewards_np)})
        grad_w_np_2 = sess.run(grad_w, feed_dict=feed_dict)
        print time() - t
        ##############################

        ##############################
        hiddens_np, dec_actions_np, inputs_np, rewards_np, hindices_np, cindices_np = [], [], [], [], [], []
        inputs_np.append(init_input_np)
        hindices_np.append(init_hidx_np)
        cindices_np.append(init_cidx_np)
        
        t = time()
        handle = sess.partial_run_setup(all_fetches+grad_w, all_feeds)

        for i in range(lsize):
            # t_i = time()
            feed_dict = {inputs[i]:inputs_np[i], hindices[i]:hindices_np[i], cindices[i]:cindices_np[i]}
            if i == 0:
                feed_dict.update({enc_input:enc_input_np})
            h_i_np, a_i_np = sess.partial_run(handle, [dec_hiddens[i], dec_actions[i]], feed_dict=feed_dict)
            hiddens_np.append(h_i_np)
            dec_actions_np.append(a_i_np)
            input_i_np, reward_i_np, hidx_np, cidx_np = action(dec_actions_np[i])
            rewards_np.append(reward_i_np)
            inputs_np.append(input_i_np)
            hindices_np.append(hidx_np)
            cindices_np.append(cidx_np)
            # print i, time() - t_i
        print time() - t

        t = time()
        grad_w_np_1 = sess.partial_run(handle, grad_w, feed_dict={go:go_np for go, go_np in zip(rewards, rewards_np)})
        print time() - t

        # # print type(grad_w_np_1), type(grad_w_np_2)
        for g1, g2 in zip(grad_w_np_1, grad_w_np_2):
            if type(g1) != type(g2):
                print 'diff in type', type(g1), type(g2)
                continue
            elif not isinstance(g1, np.ndarray):
                print 'not numpy array', type(g1), type(g2)
                continue

            if not np.allclose(g1, g2):
                print 'g1', np.max(g1), np.min(g1)
                print 'g2', np.max(g2), np.min(g2)
            else: 
                print 'Pass: g1 = g2', g1.shape, g2.shape
                if np.allclose(g1, np.zeros_like(g1)):
                    print 'Fail: g1 != 0', np.max(g1), np.min(g1)
                    
                if np.allclose(g2, np.zeros_like(g2)):
                    print 'Fail: g2 != 0', np.max(g2), np.min(g2)