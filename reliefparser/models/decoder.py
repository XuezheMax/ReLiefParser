import numpy as np
import tensorflow as tf

import bisect
from time import time

class Decoder(object):
    def __init__(self, isize, hsize, msize, asize, max_len, rnn_class, **kwargs):
        super(Decoder, self).__init__()

        self.name  = kwargs.get('name', self.__class__.__name__)
        self.scope = kwargs.get('scope', self.name)

        self.epsilon = tf.Variable(kwargs.get('epsilon', 1.0), trainable=False)

        self.isize = isize
        self.hsize = hsize
        self.msize = msize
        self.asize = asize

        self.max_len = max_len

        self.rnn_cell = rnn_class(self.hsize)

        self.weight_intializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)

    def __call__(self, input_indices, memory, valid_indices, left_indices, right_indices, valid_masks, init_state=None):
        batch_size = tf.shape(input_indices[0])[0]
        if init_state is None:
            init_state = self.rnn_cell.zero_state(batch_size, dtype=tf.float32)

        # variables invariant across steps
        mem_shape = tf.shape(memory)
        base_idx = tf.expand_dims(tf.range(mem_shape[0]) * mem_shape[1], [1])

        def step(state_tm, in_idx_t, vd_idx_t, lt_idx_t, rt_idx_t, mask_t):
            # combine previsouly predicted head and child as the current input
            flat_in_idx = base_idx + in_idx_t
            inp_vecs = tf.gather(tf.reshape(memory, [-1, self.msize]), flat_in_idx)

            weight_combine = tf.get_variable(name='weight_combine', shape=[2*self.msize, self.isize],
                                             initializer=self.weight_intializer)
            bias_combine   = tf.get_variable(name='bias_combine', shape=[self.isize],
                                             initializer=tf.constant_initializer(value=0.0))

            # TODO: discuss with Max about the model design here
            inp_vecs = tf.reshape(inp_vecs, [batch_size, 2*self.msize])
            inp_t = tf.tanh(tf.matmul(inp_vecs, weight_combine) + bias_combine)

            # perform rnn step
            hid_t, state_t = self.rnn_cell(inp_t, state_tm)

            # valid memory, left memory & right memory
            flat_vd_idx = base_idx + vd_idx_t
            flat_lt_idx = base_idx + lt_idx_t
            flat_rt_idx = base_idx + rt_idx_t

            vd_mem = tf.gather(tf.reshape(memory, [-1, self.msize]), flat_vd_idx)  # valid memory
            lt_mem = tf.gather(tf.reshape(memory, [-1, self.msize]), flat_lt_idx)  # left  memory
            rt_mem = tf.gather(tf.reshape(memory, [-1, self.msize]), flat_rt_idx)  # right memory

            # attention vec left
            weight_hid_left = tf.get_variable(name='weight_hidden_left', shape=[self.hsize, self.asize],
                                              initializer=self.weight_intializer)
            weight_hd_left  = tf.get_variable(name='weight_head_left', shape=[1, self.msize, self.asize],
                                              initializer=self.weight_intializer)
            weight_cd_left  = tf.get_variable(name='weight_child_left', shape=[1, self.msize, self.asize],
                                              initializer=self.weight_intializer)
            weight_left     = tf.get_variable(name='weight_left', shape=[self.asize],
                                              initializer=self.weight_intializer)
            bias_left       = tf.get_variable(name='bias_left', shape=[self.asize],
                                              initializer=tf.constant_initializer(value=0.0))

            weight_hid_right = tf.get_variable(name='weight_hidden_right', shape=[self.hsize, self.asize],
                                               initializer=self.weight_intializer)
            weight_hd_right  = tf.get_variable(name='weight_head_right', shape=[1, self.msize, self.asize],
                                               initializer=self.weight_intializer)
            weight_cd_right  = tf.get_variable(name='weight_child_right', shape=[1, self.msize, self.asize],
                                               initializer=self.weight_intializer)
            weight_right     = tf.get_variable(name='weight_right', shape=[self.asize],
                                               initializer=self.weight_intializer)
            bias_right       = tf.get_variable(name='bias_right', shape=[self.asize],
                                               initializer=tf.constant_initializer(value=0.0))

            # left-arc score (head = valid memory, child = left memory)
            hd_att_left = tf.nn.conv1d(vd_mem, weight_hd_left, 1, 'SAME')
            cd_att_left = tf.nn.conv1d(lt_mem, weight_cd_left, 1, 'SAME')
            att_left = tf.tanh(tf.expand_dims(tf.matmul(hid_t, weight_hid_left), 1) + 
                               hd_att_left + cd_att_left + bias_left)
            score_left = tf.reduce_sum(att_left * weight_left, [2])

            # right-arc score (head = valid memory, child = right memory)
            hd_att_right = tf.nn.conv1d(vd_mem, weight_hd_right, 1, 'SAME')
            cd_att_right = tf.nn.conv1d(rt_mem, weight_cd_right, 1, 'SAME')
            att_right = tf.tanh(tf.expand_dims(tf.matmul(hid_t, weight_hid_right), 1) + 
                                hd_att_right + cd_att_right + bias_right)
            score_right = tf.reduce_sum(att_right * weight_right, [2])
            
            # concatenate and softmax
            prob_t = tf.nn.softmax(tf.concat(1, [score_left, score_right]))
            if mask_t is not None:
                prob_t = prob_t * mask_t
                prob_t /= tf.reduce_sum(prob_t, reduction_indices=[1], keep_dims=True)
            logp_t = tf.log(prob_t + 1e-8)

            # use epsilon greedy as the exploring policy
            greedy_act_func = lambda: tf.argmax(logp_t, dimension=1)
            sample_act_func = lambda: tf.reshape(tf.multinomial(logp_t, 1), [-1])

            rand_num = tf.random_uniform(shape=[1])[0]
            act_t = tf.cond(rand_num>self.epsilon, greedy_act_func, sample_act_func)
            act_t = tf.to_int32(act_t)

            # probabilty of sampled action
            prob_shape_t = tf.shape(logp_t)
            action_idx = tf.range(prob_shape_t[0]) * prob_shape_t[1] + act_t
            act_logp_t = tf.gather(tf.reshape(logp_t, [-1]), action_idx)
            
            return hid_t, state_t, act_t, act_logp_t

        hiddens, states, actions, act_logps = [], [], [], []
        # core computational graph
        with tf.variable_scope(self.scope) as dec_scope:
            for step_idx in range(self.max_len):
                # recurrent parameter share
                if step_idx > 0:
                    dec_scope.reuse_variables()
                
                # fetch step func arguments
                state_tm = states[step_idx-1] if step_idx > 0 else init_state
                in_idx_t = input_indices[step_idx]
                vd_idx_t = valid_indices[step_idx]
                lt_idx_t = left_indices[step_idx]
                rt_idx_t = right_indices[step_idx]
                mask_t = valid_masks[step_idx] if valid_masks is not None else None

                
                # call step func
                hid_t, state_t, act_t, act_prob_t = step(state_tm, in_idx_t, vd_idx_t, lt_idx_t, rt_idx_t, mask_t)

                # store step func returns
                hiddens.append(hid_t)
                states.append(state_t)
                actions.append(act_t)
                act_logps.append(act_prob_t)

        return hiddens, actions, act_logps

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
    input_indices, valid_indices, left_indices, right_indices = [], [], [], []
    for i in range(dec_model.max_len):
        input_indices.append(tf.placeholder(dtype=tf.int32, shape=[None, 2], name='input_index_%d'%i))
        valid_indices.append(tf.placeholder(dtype=tf.int32, name='valid_index_%d'%i))
        left_indices.append (tf.placeholder(dtype=tf.int32, name='left_index_%d'%i))
        right_indices.append(tf.placeholder(dtype=tf.int32, name='right_index_%d'%i))

    # Comes from the encoder
    memory = tf.Variable(np.random.randn(bsize, lsize, msize), dtype=tf.float32, trainable=False)
    
    hiddens, actions, act_probs = dec_model(input_indices, memory, valid_indices, left_indices, right_indices)
    print len(hiddens), len(actions), len(act_probs)

    ####################
    # run-time section
    ####################

    all_feeds = []
    all_feeds.extend(input_indices)
    all_feeds.extend(valid_indices)
    all_feeds.extend(left_indices)
    all_feeds.extend(right_indices)

    all_fetches = []
    all_fetches.extend(hiddens)
    all_fetches.extend(actions)

    # get from evironment
    def take_action(action):
        input_idx  = np.repeat(np.arange(2).astype(np.int32).reshape(1, -1), bsize, axis=0)
        valid_idx = np.repeat(np.arange(lsize).astype(np.int32).reshape(1, -1), bsize, axis=0)
        left_idx = np.repeat(np.arange(lsize).astype(np.int32).reshape(1, -1), bsize, axis=0)
        right_idx = np.repeat(np.arange(lsize).astype(np.int32).reshape(1, -1), bsize, axis=0)

        if action is None:
            reward = None
        else:
            reward = np.ones(action.shape)
        return reward, input_idx, valid_idx, left_idx, right_idx

    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        _, init_inidx_np, init_vdidx_np, init_ltidx_np, init_rtidx_np = take_action(None)

        ##############################
        input_indices_np, valid_indices_np, left_indices_np, right_indices_np = [], [], [], []
        hiddens_np, actions_np = [], []

        input_indices_np.append(init_inidx_np)
        valid_indices_np.append(init_vdidx_np)
        left_indices_np.append(init_ltidx_np)
        right_indices_np.append(init_rtidx_np)
        
        t = time()
        
        feed_dict={}
        for i in range(lsize):
            # t_i = time()
            feed_dict.update({input_indices[i]:input_indices_np[i], 
                              valid_indices[i]:valid_indices_np[i], 
                              left_indices[i]:left_indices_np[i],
                              right_indices[i]:right_indices_np[i]})
            
            h_i_np, a_i_np = sess.run([hiddens[i], actions[i]], feed_dict=feed_dict)
            
            hiddens_np.append(h_i_np)
            actions_np.append(a_i_np)
            
            reward_np, input_idx_np, valid_idx_np, left_idx_np, right_idx_np = take_action(actions_np[i])
            
            input_indices_np.append(input_idx_np)
            valid_indices_np.append(valid_idx_np)
            left_indices_np.append(left_idx_np)
            right_indices_np.append(right_idx_np)
            
            # print i, time() - t_i
        print time() - t