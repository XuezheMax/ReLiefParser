import numpy as np
import tensorflow as tf

import bisect
from time import time

class TreeDecoder(object):
    def __init__(self, isize, hsize, msize, asize, max_len, rnn_class, **kwargs):
        super(TreeDecoder, self).__init__()

        self.name  = kwargs.get('name', self.__class__.__name__)
        self.scope = kwargs.get('scope', self.name)

        self.epsilon = tf.Variable(kwargs.get('epsilon', 1.0), trainable=False)

        self.isize = isize
        self.hsize = hsize
        self.msize = msize
        self.asize = asize

        self.max_len = max_len

        self.num_layer = kwargs.get('num_layer', 1)
        self.rnn_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_class(num_units=self.hsize)] * self.num_layer)

        self.weight_intializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)

    # memory        : a [batch_size X seq_len X msize] tensor (float32)
    # subtree_masks : a list of [batch_size X seq_len] tensors (float32)
    # valid_indices : a list of [batch_size X seq_len] tensors (int32)
    # left_indices  : a list of [batch_size X seq_len] tensors (int32)
    # right_indices : a list of [batch_size X seq_len] tensors (int32)
    # valid_masks   : a list of [batch_size X 2*seq_len] tensors (float32)
    def __call__(self, memory, subtree_masks, valid_indices, left_indices, right_indices, valid_masks, init_state=None):
        # initial states and variables across steps
        batch_size = tf.shape(memory)[0]
        if init_state is None:
            init_state = self.rnn_cell.zero_state(batch_size, dtype=tf.float32)
        
        # padding the memory with a dummy (all-zero) vector at the end of the 2nd dimension
        pad_memory = tf.pad(memory, [[0,0],[0,1],[0,0]])
        base_idx = tf.expand_dims(tf.range(batch_size) * tf.shape(pad_memory)[1], [1])

        def step(state_tm, subtree_t, vd_idx_t, lt_idx_t, rt_idx_t, mask_t):
            # attention vec left
            weight_hid_input = tf.get_variable(name='weight_hidden_input', shape=[self.hsize, self.asize],
                                              initializer=self.weight_intializer)
            weight_mem_input = tf.get_variable(name='weight_head_input', shape=[1, self.msize, self.asize],
                                              initializer=self.weight_intializer)
            weight_input     = tf.get_variable(name='weight_input', shape=[self.asize],
                                              initializer=self.weight_intializer)
            bias_input       = tf.get_variable(name='bias_input', shape=[self.asize],
                                              initializer=tf.constant_initializer(value=0.0))

            # attention over the subtree memory
            hid_tm = state_tm[-1] if isinstance(state_tm[-1], tf.Tensor) else state_tm[-1].h
            att_input = tf.tanh(tf.expand_dims(tf.matmul(hid_tm, weight_hid_input), 1) + \
                                tf.nn.conv1d(pad_memory, weight_mem_input, 1, 'SAME') + \
                                bias_input)
            score_input = tf.reduce_sum(att_input * weight_input, [2])
            prob_input = tf.nn.softmax(score_input * subtree_t)
            inp_t = tf.reduce_sum(tf.expand_dims(prob_input, dim=2) * pad_memory, reduction_indices=[1])

            # perform the rnn step
            hid_t, state_t = self.rnn_cell(inp_t, state_tm)

            # valid memory, left memory & right memory
            flat_vd_idx = base_idx + vd_idx_t
            flat_lt_idx = base_idx + lt_idx_t
            flat_rt_idx = base_idx + rt_idx_t

            vd_mem = tf.gather(tf.reshape(pad_memory, [-1, self.msize]), flat_vd_idx)  # valid memory
            lt_mem = tf.gather(tf.reshape(pad_memory, [-1, self.msize]), flat_lt_idx)  # left  memory
            rt_mem = tf.gather(tf.reshape(pad_memory, [-1, self.msize]), flat_rt_idx)  # right memory

            # parameters for left attention
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

            # left-arc score (head = valid memory, child = left memory)
            hd_att_left = tf.nn.conv1d(vd_mem, weight_hd_left, 1, 'SAME')
            cd_att_left = tf.nn.conv1d(lt_mem, weight_cd_left, 1, 'SAME')
            att_left = tf.tanh(tf.expand_dims(tf.matmul(hid_t, weight_hid_left), 1) + 
                               hd_att_left + cd_att_left + bias_left)
            score_left = tf.reduce_sum(att_left * weight_left, [2])

            # parameters for right attention
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

            # right-arc score (head = valid memory, child = right memory)
            hd_att_right = tf.nn.conv1d(vd_mem, weight_hd_right, 1, 'SAME')
            cd_att_right = tf.nn.conv1d(rt_mem, weight_cd_right, 1, 'SAME')
            att_right = tf.tanh(tf.expand_dims(tf.matmul(hid_t, weight_hid_right), 1) + 
                                hd_att_right + cd_att_right + bias_right)
            score_right = tf.reduce_sum(att_right * weight_right, [2])
            
            # concatenate and softmax
            score_t = tf.concat(1, [score_left, score_right])
            if mask_t is not None:
                logp_t = tf.nn.log_softmax(score_t * mask_t)
            else:
                logp_t = tf.nn.log_softmax(score_t)

            # use epsilon greedy as the exploring policy
            greedy_act_func = lambda: tf.argmax(logp_t, dimension=1)
            sample_act_func = lambda: tf.reshape(tf.multinomial(logp_t, 1), [-1])

            # rand_num = tf.random_uniform(shape=[1])[0]
            # act_t = tf.cond(rand_num>self.epsilon, greedy_act_func, sample_act_func)
            act_t = sample_act_func()
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
                subtree_t = subtree_masks[step_idx]
                vd_idx_t = valid_indices[step_idx]
                lt_idx_t = left_indices[step_idx]
                rt_idx_t = right_indices[step_idx]
                mask_t = valid_masks[step_idx] if valid_masks is not None else None

                
                # call step func
                hid_t, state_t, act_t, act_prob_t = step(state_tm, subtree_t, vd_idx_t, lt_idx_t, rt_idx_t, mask_t)

                # store step func returns
                hiddens.append(hid_t)
                states.append(state_t)
                actions.append(act_t)
                act_logps.append(act_prob_t)

        return hiddens, actions, act_logps

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

        self.num_layer = kwargs.get('num_layer', 1)
        self.rnn_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_class(num_units=self.hsize)] * self.num_layer)

        self.weight_intializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)

    def __call__(self, memory, input_indices, valid_indices, left_indices, right_indices, valid_masks, init_state=None):
        # initial states and variables across steps
        batch_size = tf.shape(memory)[0]
        if init_state is None:
            init_state = self.rnn_cell.zero_state(batch_size, dtype=tf.float32)
        
        # padding the memory with a dummy (all-zero) vector at the end of the 2nd dimension
        pad_memory = tf.pad(memory, [[0,0],[0,1],[0,0]])
        base_idx = tf.expand_dims(tf.range(batch_size) * tf.shape(pad_memory)[1], [1])

        def step(state_tm, in_idx_t, vd_idx_t, lt_idx_t, rt_idx_t, mask_t):
            # combine previsouly predicted head and child as the current input
            flat_in_idx = base_idx + in_idx_t
            inp_vecs = tf.gather(tf.reshape(pad_memory, [-1, self.msize]), flat_in_idx)

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

            vd_mem = tf.gather(tf.reshape(pad_memory, [-1, self.msize]), flat_vd_idx)  # valid memory
            lt_mem = tf.gather(tf.reshape(pad_memory, [-1, self.msize]), flat_lt_idx)  # left  memory
            rt_mem = tf.gather(tf.reshape(pad_memory, [-1, self.msize]), flat_rt_idx)  # right memory

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
            score_t = tf.concat(1, [score_left, score_right])
            if mask_t is not None:
                score_t = score_t * mask_t
            logp_t = tf.nn.log_softmax(score_t)

            # use epsilon greedy as the exploring policy
            greedy_act_func = lambda: tf.argmax(logp_t, dimension=1)
            sample_act_func = lambda: tf.reshape(tf.multinomial(logp_t, 1), [-1])

            # rand_num = tf.random_uniform(shape=[1])[0]
            # act_t = tf.cond(rand_num>self.epsilon, greedy_act_func, sample_act_func)
            act_t = sample_act_func()
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