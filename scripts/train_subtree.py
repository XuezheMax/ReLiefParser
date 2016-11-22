import os, sys
sys.path.append('..')
import json
from time import time
import numpy as np

import tensorflow as tf

import reliefparser.environment as env
import reliefparser.io as dataio
import reliefparser.utils as utils

from reliefparser.models import *

# Basic model parameters as external flags.
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('device', '/cpu:0', 'Device to use.')
tf.app.flags.DEFINE_boolean('debug', True, 'Run program in debug mode.')

# np.random.seed(1)
logging = tf.logging
logging.set_verbosity(logging.INFO)

word_alphabet, pos_alphabet, type_alphabet = dataio.create_alphabets("data/alphabets/", 
            ['../data/ptb3.0-stanford.auto.cpos.train.conll', '../data/ptb3.0-stanford.auto.cpos.dev.conll'],#, '../data/Europarl/europarl-v7.da-en.en.auto.cpos.conll'], 
            100000)

if FLAGS.debug:
    data = dataio.read_data('../sample.conll', word_alphabet, pos_alphabet, type_alphabet)
else:
    data = dataio.read_data('../data/ptb3.0-stanford.auto.cpos.train.conll', word_alphabet, pos_alphabet, type_alphabet)

print 'number of samples: %s' % (' '.join('%d' % len(d) for d in data))

####################
# Hyper-parameters
####################
vsize = word_alphabet.size()
esize = 32
hsize = 32
asize = 32
isize = 32
num_layer = 1

buckets = [10]
# buckets = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
max_len = buckets[-1]

bsize = 1

####################
# Building model
####################
g = tf.Graph()
with g.as_default(), tf.device(FLAGS.device):
    # model initialization
    pointer_net = PointerNet(vsize, esize, hsize, asize, buckets, dec_isize=isize, num_layer=num_layer, tree_decoder=True)

    # placeholders
    enc_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='encoder_input')

    input_masks, values, valid_masks = [], [], []
    valid_indices, left_indices, right_indices = [], [], []
    for i in range(max_len):
        values.append(tf.placeholder(dtype=tf.float32, name='value_%d'%i))
        valid_masks.append(tf.placeholder(dtype=tf.float32, name='input_index_%d'%i))
        input_masks.append(tf.placeholder(dtype=tf.float32, shape=[None, None], name='input_mask_%d'%i))

        valid_indices.append(tf.placeholder(dtype=tf.int32, name='valid_index_%d'%i))
        left_indices.append (tf.placeholder(dtype=tf.int32, name='left_index_%d'%i))
        right_indices.append(tf.placeholder(dtype=tf.int32, name='right_index_%d'%i))

    # build computation graph
    dec_hiddens, dec_actions, train_ops = pointer_net(
                                                enc_input, input_masks, valid_indices, 
                                                left_indices, right_indices, values, 
                                                valid_masks=valid_masks)

########## End of section ##########

baselines = np.zeros(max_len)
max_iter = 100000

expected_value, observed_value = 0., 0.

logging.info('Begin training...')
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(graph=g, config=config) as sess:
# with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    uidx = 0
    # wid_inputs, pid_inputs, hid_inputs, tid_inputs, masks = dataio.get_batch(data, bsize)
    for eidx in range(max_iter):
        # if eidx > 0 and eidx % 200 == 0:
        #     new_epsilon = pointer_net.decoder.epsilon.value() / 2.
        #     pointer_net.decoder.epsilon.assign(new_epsilon)
        #     logging.info('New epsilon: %.4f' % sess.run(new_epsilon))

        wid_inputs, pid_inputs, hid_inputs, tid_inputs, masks = dataio.get_batch(data, bsize)
        expected_value += np.sum(masks[:,1:])
        curr_env = env.Environment(hid_inputs, masks)

        init_vdidx_np, init_ltidx_np, init_rtidx_np = curr_env.get_indexes()
        batch_size, seq_len = init_vdidx_np.shape

        # setup partial run handle
        bucket_id = bisect.bisect_left(buckets, seq_len)
        train_op = train_ops[bucket_id]

        all_feeds = []
        all_feeds.extend(values[:seq_len])
        all_feeds.extend(valid_masks[:seq_len])
        all_feeds.extend(input_masks[:seq_len])
        all_feeds.extend(valid_indices[:seq_len])
        all_feeds.extend(right_indices[:seq_len])
        all_feeds.extend(left_indices[:seq_len])
        all_feeds.append(enc_input)
        
        all_fetches = []
        # all_fetches.extend(dec_hiddens[:seq_len])
        all_fetches.extend(dec_actions[:seq_len])
        all_fetches.append(train_op)

        handle = sess.partial_run_setup(all_fetches, all_feeds)

        # initial data
        init_vdidx_np[init_vdidx_np==-1] = seq_len
        init_ltidx_np[init_ltidx_np==-1] = seq_len
        init_rtidx_np[init_rtidx_np==-1] = seq_len

        init_inmask_np = np.zeros((bsize, seq_len+1)).astype(np.float32)
        init_inmask_np[:,seq_len] = 1.

        # numerical data collections
        valid_masks_np, input_masks_np, valid_indices_np, left_indices_np, right_indices_np = [], [], [], [], []
        hiddens_np, actions_np, rewards_np = [], [], []

        init_valid_mask = np.tile(masks, (1,2))
        init_valid_mask[:,0] = 0
        init_valid_mask[:,seq_len] = 1
        valid_masks_np.append(init_valid_mask)
        input_masks_np.append(init_inmask_np)
        valid_indices_np.append(init_vdidx_np)
        left_indices_np.append(init_ltidx_np)
        right_indices_np.append(init_rtidx_np)

        for i in range(seq_len):
            feed_dict = {valid_masks[i]:valid_masks_np[i], 
                         input_masks[i]:input_masks_np[i], 
                         valid_indices[i]:valid_indices_np[i], 
                         left_indices[i]:left_indices_np[i], 
                         right_indices[i]:right_indices_np[i]}
            if i == 0:
                feed_dict.update({enc_input:wid_inputs})

            # h_i_np, a_i_np = sess.partial_run(handle, [dec_hiddens[i], dec_actions[i]], feed_dict=feed_dict)
            a_i_np = sess.partial_run(handle, dec_actions[i], feed_dict=feed_dict)
            if eidx == max_iter-1:
                print 'ac', a_i_np

            # hiddens_np.append(h_i_np)
            # actions_np.append(a_i_np)

            reward_i, _, _, input_mask, valid_idx, left_idx, right_idx = curr_env.take_action(a_i_np)

            # there are at most (seq_len - 1) actions to take within each minibatch
            if i < seq_len - 1:
                reward_i = reward_i * masks[:,i+1]
            else:
                reward_i *= 0.

            valid_mask = np.tile(np.asarray(valid_idx>=0, dtype=np.float32), (1, 2))
            valid_mask[:,0] = 0
            valid_mask[:,seq_len] = 1
            valid_idx[valid_idx==-1] = seq_len
            left_idx [left_idx ==-1] = seq_len
            right_idx[right_idx==-1] = seq_len

            rewards_np.append(reward_i)
            valid_masks_np.append(valid_mask)
            input_masks_np.append(input_mask)
            valid_indices_np.append(valid_idx)
            left_indices_np.append(left_idx)
            right_indices_np.append(right_idx)

            if eidx == max_iter-1:
                print 'rw', rewards_np[i]
        
        values_np = utils.reward_to_value(rewards_np)
        utils.apply_baseline(values_np, baselines, masks)

        sess.partial_run(handle, train_op, feed_dict={value:value_np for value, value_np in zip(values, values_np)})

        R = np.array(rewards_np)
        observed_value += R[R==1].shape[0]
        if eidx % 50 == 0:
            print '[%d] exp. %.4f, obs. %.4f, acc: %.4f, baselines: %s' % (
                        eidx, expected_value / 50., observed_value / 50., observed_value / expected_value, 
                        # ' '.join(['%.2f' % bl for bl in sess.run(pointer_net.baselines[:seq_len])])
                        ' '.join(['%.2f' % bl for bl in baselines[seq_len-2:0:-1]])
                    )

            expected_value, observed_value = 0., 0.
        uidx += 1

    for i in range(seq_len):
        print 'baseline %d: %.4f' %(i, sess.run(pointer_net.baselines[i]))
