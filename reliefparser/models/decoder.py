import numpy as np
import tensorflow as tf

import bisect
from time import time

class Decoder(object):
    def __init__(self, isize, hsize, msize, asize, max_len, **kwargs):
        super(Decoder, self).__init__()

        self.name  = kwargs.get('name', self.__class__.__name__)
        self.scope = kwargs.get('scope', self.name)
        
        self.isize = isize
        self.hsize = hsize
        self.msize = msize
        self.asize = asize

        self.max_len = max_len

        self.rnn_cell = tf.nn.rnn_cell.GRUCell(self.hsize)

    def __call__(self, inputs, memory, child_indices, head_indices, rewards):
        batch_size = tf.shape(inputs[0])[0]
        init_hidden = self.rnn_cell.zero_state(batch_size, dtype=tf.float32)

        # variables invariant across steps
        mem_shape = tf.shape(memory)
        base_idx = tf.expand_dims(tf.range(mem_shape[0]) * mem_shape[1], [1])

        def step(hid_tm, inp_t, hd_idx_t, cd_idx_t):
            # perform rnn step
            hid_t, _ = self.rnn_cell(inp_t, hid_tm)

            # head memory & child memory
            flat_hd_idx = base_idx + hd_idx_t
            hd_mem = tf.gather(tf.reshape(memory, [-1, self.msize]), flat_hd_idx)
            flat_cd_idx = base_idx + cd_idx_t
            cd_mem = tf.gather(tf.reshape(memory, [-1, self.msize]), flat_cd_idx)
            

            # attention vec left
            weight_hid_left = tf.get_variable(name='weight_hidden_left', shape=[self.hsize, self.asize],
                                              initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            weight_hd_left  = tf.get_variable(name='weight_head_left', shape=[1, self.msize, self.asize],
                                              initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            weight_cd_left  = tf.get_variable(name='weight_child_left', shape=[1, self.msize, self.asize],
                                              initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            weight_left     = tf.get_variable(name='weight_left', shape=[self.asize],
                                              initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            bias_left       = tf.get_variable(name='bias_left', shape=[self.asize],
                                              initializer=tf.constant_initializer(value=0.0))

            weight_hid_right = tf.get_variable(name='weight_hidden_right', shape=[self.hsize, self.asize],
                                              initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            weight_hd_right  = tf.get_variable(name='weight_head_right', shape=[1, self.msize, self.asize],
                                              initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            weight_cd_right  = tf.get_variable(name='weight_child_right', shape=[1, self.msize, self.asize],
                                              initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            weight_right     = tf.get_variable(name='weight_right', shape=[self.asize],
                                              initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))
            bias_right       = tf.get_variable(name='bias_right', shape=[self.asize],
                                              initializer=tf.constant_initializer(value=0.0))

            # left-arc score
            hd_att_left = tf.nn.conv1d(hd_mem, weight_hd_left, 1, 'SAME')
            cd_att_left = tf.nn.conv1d(cd_mem, weight_cd_left, 1, 'SAME')
            att_left = tf.tanh(tf.expand_dims(tf.matmul(hid_t, weight_hid_left), 1) + 
                               hd_att_left + cd_att_left + bias_left)
            score_left = tf.reduce_sum(att_left * weight_left, [2])

            # right-arc score
            hd_att_right = tf.nn.conv1d(hd_mem, weight_hd_right, 1, 'SAME')
            cd_att_right = tf.nn.conv1d(cd_mem, weight_cd_right, 1, 'SAME')
            att_right = tf.tanh(tf.expand_dims(tf.matmul(hid_t, weight_hid_right), 1) + 
                                hd_att_right + cd_att_right + bias_right)
            score_right = tf.reduce_sum(att_right * weight_right, [2])
            
            # concatenate and softmax
            prob_t = tf.nn.softmax(tf.concat(1, [score_left, score_right]))
            act_t = tf.to_int32(tf.argmax(prob_t, dimension=1))

            # probabilty of sampled action
            prob_shape_t = tf.shape(prob_t)
            action_idx = tf.range(prob_shape_t[0]) * prob_shape_t[1] + act_t
            act_prob_t = tf.gather(tf.reshape(prob_t, [-1]), action_idx)
            
            return hid_t, act_t, act_prob_t

        hiddens, actions, act_probs = [], [], []
        # core computational graph
        with tf.variable_scope(self.scope) as dec_scope:
            for step_idx in range(self.max_len):
                # recurrent parameter share
                if step_idx > 0:
                    dec_scope.reuse_variables()
                
                # fetch step func arguments
                hid_tm = hiddens[step_idx-1] if step_idx > 0 else init_hidden
                inp_t  = inputs[step_idx]
                hd_idx_t = head_indices[step_idx]
                cd_idx_t = child_indices[step_idx]
                
                # call step func
                hid_t, act_t, act_prob_t = step(hid_tm, inp_t, hd_idx_t, cd_idx_t)

                # store step func returns
                hiddens.append(hid_t)
                actions.append(act_t)
                act_probs.append(act_prob_t)

        return hiddens, actions, act_probs

#### test script
if __name__ == '__main__':
    lsize = 10
    bsize = 32

    isize = 512
    hsize = 512
    msize = 512
    asize = 128

    buckets = [10]#, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

    dec_model = Decoder(isize, hsize, msize, asize, buckets[-1])

    # placeholders
    inputs, rewards = [], []
    cindices, hindices = [], []
    for i in range(dec_model.max_len):
        inputs.append(tf.placeholder(dtype=tf.float32, shape=[None, isize], name='input_%d'%i))
        rewards.append(tf.placeholder(dtype=tf.float32, name='reward_%d'%i))
        cindices.append(tf.placeholder(dtype=tf.int32, name='child_indice_%d'%i))
        hindices.append(tf.placeholder(dtype=tf.int32, name='head_indice_%d'%i))

    # Comes from the encoder
    memory = tf.Variable(np.random.randn(bsize, lsize, msize), dtype=tf.float32, trainable=False)
    
    hiddens, actions, act_probs = dec_model(inputs, memory, cindices, hindices, rewards)
    print len(hiddens), len(actions), len(act_probs)

    ####################
    # run-time section
    ####################

    all_feeds = []
    all_feeds.extend(inputs)
    all_feeds.extend(rewards)
    all_feeds.extend(hindices)
    all_feeds.extend(cindices)

    all_fetches = []
    all_fetches.extend(hiddens)
    all_fetches.extend(actions)

    # get from evironment
    def take_action(action):
        input  = np.ones((bsize, isize)) * 0.1
        reward = np.ones(action.shape)
        h_indices = np.repeat(np.arange(lsize).astype(np.int32).reshape(1, -1), bsize, axis=0)
        c_indices = np.repeat(np.arange(lsize).astype(np.int32).reshape(1, -1), bsize, axis=0)
        return input, reward, h_indices, c_indices

    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        init_input_np = np.random.randn(bsize, isize).astype(np.float32)
        init_hidx_np = np.repeat(np.arange(lsize).astype(np.int32).reshape(1, -1), bsize, axis=0)
        init_cidx_np = np.repeat(np.arange(lsize).astype(np.int32).reshape(1, -1), bsize, axis=0)

        ##############################
        hiddens_np, actions_np, inputs_np, rewards_np, hindices_np, cindices_np = [], [], [], [], [], []
        inputs_np.append(init_input_np)
        hindices_np.append(init_hidx_np)
        cindices_np.append(init_cidx_np)
        t = time()
        
        feed_dict={}
        for i in range(lsize):
            # t_i = time()
            feed_dict.update({inputs[i]:inputs_np[i], hindices[i]:hindices_np[i], cindices[i]:cindices_np[i]})
            
            h_i_np, a_i_np = sess.run([hiddens[i], actions[i]], feed_dict=feed_dict)
            
            hiddens_np.append(h_i_np)
            actions_np.append(a_i_np)
            input_i_np, reward_i_np, hidx_np, cidx_np = take_action(actions_np[i])
            rewards_np.append(reward_i_np)
            inputs_np.append(input_i_np)
            hindices_np.append(hidx_np)
            cindices_np.append(cidx_np)
            # print i, time() - t_i
        print time() - t