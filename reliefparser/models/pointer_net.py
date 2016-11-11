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

        self.max_grad_norm = kwargs.get('max_grad_norm', 10)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)

        self.encoder = Encoder(self.enc_vsize, self.enc_esize, self.enc_hsize)
        self.decoder = Decoder(self.dec_isize, self.dec_hsize, self.dec_msize, self.dec_asize, self.buckets[-1])

    def __call__(self, enc_input, dec_input_indices, valid_indices, left_indices, right_indices, rewards):
        batch_size = tf.shape(enc_input)[0]
        # forward computation graph
        with tf.variable_scope(self.scope):
            # encoder output
            enc_memory, enc_final_state_fw, _ = self.encoder(enc_input)

            # padding the memory with a dummy (all-zero) vector at the end
            enc_memory = tf.pad(enc_memory, [[0,0],[0,1],[0,0]])

            # decoder 
            # TODO: discuss with Max about the initial hidden of the decoder
            dec_hiddens, dec_actions, dec_act_probs = self.decoder(
                                                            dec_input_indices, enc_memory, 
                                                            valid_indices, left_indices, right_indices, 
                                                            rewards, init_hidden=enc_final_state_fw)

            # cost
            costs = []
            for act_prob, reward in zip(dec_act_probs, rewards):
                costs.append(tf.reduce_mean(act_prob * (-reward)))

        # gradient computation graph
        self.params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        train_ops = []
        for limit in self.buckets:
            print '0 ~ %d' % (limit-1)
            grad_params = tf.gradients(tf.reduce_sum(tf.pack(costs[:limit])), self.params)
            clipped_gradients, norm = tf.clip_by_global_norm(grad_params, self.max_grad_norm)
            train_op = self.optimizer.apply_gradients(
                            zip(grad_params, self.params),
                            global_step=tf.contrib.framework.get_or_create_global_step())
            with tf.control_dependencies([train_op]):
                train_ops.append(tf.constant(1.))

        return dec_hiddens, dec_actions, train_ops

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

    input_indices, rewards = [], []
    valid_indices, left_indices, right_indices = [], [], []
    for i in range(max_len):
        rewards.append(tf.placeholder(dtype=tf.float32, name='reward_%d'%i))
        input_indices.append(tf.placeholder(dtype=tf.int32, shape=[None, 2], name='input_index_%d'%i))
        valid_indices.append(tf.placeholder(dtype=tf.int32, name='valid_index_%d'%i))
        left_indices.append (tf.placeholder(dtype=tf.int32, name='left_index_%d'%i))
        right_indices.append(tf.placeholder(dtype=tf.int32, name='right_index_%d'%i))

    # build computation graph
    dec_hiddens, dec_actions, train_ops = pointer_net(enc_input, input_indices, valid_indices, left_indices, right_indices, rewards)

    ####################
    # run-time section
    ####################

    lsize = 10
    bsize = 32

    all_feeds = []
    all_feeds.extend(rewards)
    all_feeds.extend(input_indices)
    all_feeds.extend(valid_indices)
    all_feeds.extend(right_indices)
    all_feeds.extend(left_indices)
    all_feeds.append(enc_input)

    all_fetches = []
    all_fetches.extend(dec_hiddens)
    all_fetches.extend(dec_actions)

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

        enc_input_np = np.random.randint(0, vsize, size=[bsize, lsize]).astype(np.int32)

        _, init_inidx_np, init_vdidx_np, init_ltidx_np, init_rtidx_np = take_action(None)

        bucket_id = bisect.bisect_left(buckets, lsize)
        train_op = train_ops[bucket_id]
        print train_op

        # bucket_id = bisect.bisect_left(buckets, lsize)
        # grad_w = grad_params_buckets[bucket_id]

        ##############################
        input_indices_np, valid_indices_np, left_indices_np, right_indices_np = [], [], [], []
        hiddens_np, actions_np, rewards_np = [], [], []

        input_indices_np.append(init_inidx_np)
        valid_indices_np.append(init_vdidx_np)
        left_indices_np.append(init_ltidx_np)
        right_indices_np.append(init_rtidx_np)

        # t = time()
        # feed_dict={enc_input:enc_input_np}
        # for i in range(lsize):
        #     # t_i = time()
        #     feed_dict.update({input_indices[i]:input_indices_np[i], 
        #                       valid_indices[i]:valid_indices_np[i], 
        #                       left_indices[i]:left_indices_np[i], 
        #                       right_indices[i]:right_indices_np[i]})

        #     h_i_np, a_i_np = sess.run([dec_hiddens[i], dec_actions[i]], feed_dict=feed_dict)

        #     hiddens_np.append(h_i_np)
        #     actions_np.append(a_i_np)

        #     reward_i, input_idx_np, valid_idx_np, left_idx_np, right_idx_np = take_action(actions_np[i])
            
        #     rewards_np.append(reward_i)
        #     input_indices_np.append(input_idx_np)
        #     valid_indices_np.append(valid_idx_np)
        #     left_indices_np.append(left_idx_np)
        #     right_indices_np.append(right_idx_np)
        #     # print i, time() - t_i
        # print time() - t

        # t = time()
        # # feed_dict.update({go:go_np for go, go_np in zip(rewards, rewards_np)})
        # # grad_w_np_2 = sess.run(grad_w, feed_dict=feed_dict)
        # sess.run(train_op, feed_dict=feed_dict)
        # print time() - t
        ##############################

        ##############################
        input_indices_np, valid_indices_np, left_indices_np, right_indices_np = [], [], [], []
        hiddens_np, actions_np, rewards_np = [], [], []

        input_indices_np.append(init_inidx_np)
        valid_indices_np.append(init_vdidx_np)
        left_indices_np.append(init_ltidx_np)
        right_indices_np.append(init_rtidx_np)

        t = time()
        # handle = sess.partial_run_setup(all_fetches+grad_w, all_feeds)
        handle = sess.partial_run_setup(all_fetches+[train_op], all_feeds)

        for i in range(lsize):
            # t_i = time()
            feed_dict = {input_indices[i]:input_indices_np[i], 
                         valid_indices[i]:valid_indices_np[i], 
                         left_indices[i]:left_indices_np[i], 
                         right_indices[i]:right_indices_np[i]}
            if i == 0:
                feed_dict.update({enc_input:enc_input_np})
            h_i_np, a_i_np = sess.partial_run(handle, [dec_hiddens[i], dec_actions[i]], feed_dict=feed_dict)
            
            hiddens_np.append(h_i_np)
            actions_np.append(a_i_np)

            reward_i, input_idx_np, valid_idx_np, left_idx_np, right_idx_np = take_action(actions_np[i])
            
            rewards_np.append(reward_i)
            input_indices_np.append(input_idx_np)
            valid_indices_np.append(valid_idx_np)
            left_indices_np.append(left_idx_np)
            right_indices_np.append(right_idx_np)
            # print i, time() - t_i
        print time() - t

        p_before = sess.run(pointer_net.params[0])

        t = time()
        # grad_w_np_1 = sess.partial_run(handle, grad_w, feed_dict={go:go_np for go, go_np in zip(rewards, rewards_np)})
        sess.partial_run(handle, train_op, feed_dict={go:go_np for go, go_np in zip(rewards, rewards_np)})
        print time() - t

        p_after =  sess.run(pointer_net.params[0])

        print np.allclose(p_before, p_after)

        # # # print type(grad_w_np_1), type(grad_w_np_2)
        # for g1, g2 in zip(grad_w_np_1, grad_w_np_2):
        #     if type(g1) != type(g2):
        #         print 'diff in type', type(g1), type(g2)
        #         continue
        #     elif not isinstance(g1, np.ndarray):
        #         print 'not numpy array', type(g1), type(g2)
        #         continue

        #     if not np.allclose(g1, g2):
        #         print 'g1', np.max(g1), np.min(g1)
        #         print 'g2', np.max(g2), np.min(g2)
        #     else:
        #         print 'Pass: g1 = g2', g1.shape, g2.shape
        #         if np.allclose(g1, np.zeros_like(g1)):
        #             print 'Fail: g1 != 0', np.max(g1), np.min(g1)

        #         if np.allclose(g2, np.zeros_like(g2)):
        #             print 'Fail: g2 != 0', np.max(g2), np.min(g2)