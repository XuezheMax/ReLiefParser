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
tf.app.flags.DEFINE_integer('max_epoch', 1000, 'Max training epochs to run.')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Num of samples in each batch.')
tf.app.flags.DEFINE_integer('disp_freq', 100, 'Frequency (iteration) of logging.')
tf.app.flags.DEFINE_integer('valid_freq', 10, 'Frequency (epoch) of validation.')

# np.random.seed(1)
logging = tf.logging
logging.set_verbosity(logging.INFO)

word_alphabet, pos_alphabet, type_alphabet = dataio.create_alphabets("data/alphabets/", 
            ['../data/ptb3.0-stanford.auto.cpos.train.conll', '../data/ptb3.0-stanford.auto.cpos.dev.conll'],#, '../data/Europarl/europarl-v7.da-en.en.auto.cpos.conll'], 
            100000)

if FLAGS.debug:
    data = dataio.read_data('../sample.conll', word_alphabet, pos_alphabet, type_alphabet)
    dev_data = dataio.read_data('../sample.dev.conll', word_alphabet, pos_alphabet, type_alphabet)
else:
    data = dataio.read_data('../data/ptb3.0-stanford.auto.cpos.train.conll', word_alphabet, pos_alphabet, type_alphabet)
    dev_data = dataio.read_data('../data/ptb3.0-stanford.auto.cpos.dev.conll', word_alphabet, pos_alphabet, type_alphabet)

print 'number of samples: %s' % (' '.join('%d' % len(d) for d in data))

########################################
# Hyper-parameters
########################################
vsize = word_alphabet.size()
esize = 512
hsize = 512
asize = 512
isize = 512
num_layer = 1

buckets = [10, 20]
# buckets = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
max_len = buckets[-1]

########################################
# Building computational graph
########################################
g = tf.Graph()
with g.as_default(), tf.device(FLAGS.device):
    # model initialization
    pointer_net = PointerNet(vsize, esize, hsize, asize, buckets, dec_isize=isize, num_layer=num_layer)

    # placeholders
    enc_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='encoder_input')

    input_indices, values, valid_masks = [], [], []
    valid_indices, left_indices, right_indices = [], [], []
    for i in range(max_len):
        values.append(tf.placeholder(dtype=tf.float32, name='value_%d'%i))
        valid_masks.append(tf.placeholder(dtype=tf.float32, name='input_index_%d'%i))
        input_indices.append(tf.placeholder(dtype=tf.int32, name='input_index_%d'%i))

        valid_indices.append(tf.placeholder(dtype=tf.int32, name='valid_idx_%d'%i))
        left_indices.append (tf.placeholder(dtype=tf.int32, name='left_index_%d'%i))
        right_indices.append(tf.placeholder(dtype=tf.int32, name='right_index_%d'%i))

    # build computation graph
    dec_hiddens, dec_actions, train_ops = pointer_net(
                                                enc_input, input_indices, valid_indices, 
                                                left_indices, right_indices, values, 
                                                valid_masks=valid_masks)

########################################
# Main training loop
########################################
def forward(batch, training=True):
    # unpack each mini-batch and create the environment
    wid_inputs, pid_inputs, hid_inputs, tid_inputs, masks = batch
    curr_env = env.Environment(hid_inputs, masks)

    #### get initial enviroment state
    valid_idx, left_idx, right_idx = curr_env.get_indexes()
    batch_size, seq_len = valid_idx.shape
    
    # both head and child point to padding position
    input_idx  = np.array(np.ones((batch_size, 2)) * seq_len).astype(np.int32)
    
    # turn valid index into valid mask
    valid_mask = np.tile(np.asarray(valid_idx>=0, dtype=np.float32), (1, 2))
    valid_mask[:,0] = 0
    valid_mask[:,seq_len] = 1

    # point invalid index to padding position
    valid_idx[valid_idx==-1] = seq_len
    left_idx [left_idx ==-1] = seq_len
    right_idx[right_idx==-1] = seq_len

    #### setup partial run handle
    all_feeds = []
    all_feeds.extend(valid_masks[:seq_len])
    all_feeds.extend(input_indices[:seq_len])
    all_feeds.extend(valid_indices[:seq_len])
    all_feeds.extend(right_indices[:seq_len])
    all_feeds.extend(left_indices [:seq_len])
    all_feeds.append(enc_input)
    
    all_fetches = []
    all_fetches.extend(dec_actions[:seq_len])

    #### extra feeds and fetches for training
    if training:
        # get the corresponding train_op
        bucket_id = bisect.bisect_left(buckets, seq_len)
        train_op = train_ops[bucket_id]

        all_feeds.extend(values[:seq_len])
        all_fetches.append(train_op)

    handle = sess.partial_run_setup(all_fetches, all_feeds)

    #### numerical data collections
    rewards_np, actions_np = [], []

    #### partial run each forward decoding step
    for i in range(seq_len):
        feed_dict = {valid_masks[i]   : valid_mask,
                     input_indices[i] : input_idx,
                     valid_indices[i] : valid_idx,
                     left_indices[i]  : left_idx,
                     right_indices[i] : right_idx}
        if i == 0:
            feed_dict.update({enc_input:wid_inputs})

        action_i = sess.partial_run(handle, dec_actions[i], feed_dict=feed_dict)

        reward_i, head_idx, child_idx, _, valid_idx, left_idx, right_idx = curr_env.take_action(action_i)

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

        head_idx [head_idx ==-1] = seq_len
        child_idx[child_idx==-1] = seq_len
        input_idx = np.concatenate((head_idx.reshape(-1, 1), child_idx.reshape(-1, 1)), axis=1)

        rewards_np.append(reward_i)
        actions_np.append(action_i)

    if training:
        return handle, train_op, rewards_np, masks
    else:
        return rewards_np, masks

baselines = np.zeros(max_len)

logging.info('Begin training...')
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(graph=g, config=config) as sess:
    sess.run(tf.initialize_all_variables())

    #### training loop
    uidx = 0
    expected_value, observed_value = 0., 0.
    for eidx in range(FLAGS.max_epoch):
        for batch in dataio.iterate_batch(data, FLAGS.batch_size, shuffle=True):
            # # unpack each mini-batch and create the environment
            # wid_inputs, pid_inputs, hid_inputs, tid_inputs, masks = batch
            # curr_env = env.Environment(hid_inputs, masks)

            # #### get initial enviroment state
            # valid_idx, left_idx, right_idx = curr_env.get_indexes()
            # batch_size, seq_len = valid_idx.shape
            
            # # both head and child point to padding position
            # input_idx  = np.array(np.ones((batch_size, 2)) * seq_len).astype(np.int32)
            
            # # turn valid index into valid mask
            # valid_mask = np.tile(np.asarray(valid_idx>=0, dtype=np.float32), (1, 2))
            # valid_mask[:,0] = 0
            # valid_mask[:,seq_len] = 1

            # # point invalid index to padding position
            # valid_idx[valid_idx==-1] = seq_len
            # left_idx [left_idx ==-1] = seq_len
            # right_idx[right_idx==-1] = seq_len

            # #### setup partial run handle
            # bucket_id = bisect.bisect_left(buckets, seq_len)
            # train_op = train_ops[bucket_id]

            # all_feeds = []
            # all_feeds.extend(values[:seq_len])
            # all_feeds.extend(valid_masks[:seq_len])
            # all_feeds.extend(input_indices[:seq_len])
            # all_feeds.extend(valid_indices[:seq_len])
            # all_feeds.extend(right_indices[:seq_len])
            # all_feeds.extend(left_indices [:seq_len])
            # all_feeds.append(enc_input)
            
            # all_fetches = []
            # all_fetches.extend(dec_actions[:seq_len])
            # all_fetches.append(train_op)

            # handle = sess.partial_run_setup(all_fetches, all_feeds)

            # #### numerical data collections
            # rewards_np, actions_np = [], []

            # #### partial run each forward decoding step
            # for i in range(seq_len):
            #     feed_dict = {valid_masks[i]   : valid_mask,
            #                  input_indices[i] : input_idx,
            #                  valid_indices[i] : valid_idx,
            #                  left_indices[i]  : left_idx,
            #                  right_indices[i] : right_idx}
            #     if i == 0:
            #         feed_dict.update({enc_input:wid_inputs})

            #     action_i = sess.partial_run(handle, dec_actions[i], feed_dict=feed_dict)

            #     reward_i, head_idx, child_idx, _, valid_idx, left_idx, right_idx = curr_env.take_action(action_i)

            #     # there are at most (seq_len - 1) actions to take within each minibatch
            #     if i < seq_len - 1:
            #         reward_i = reward_i * masks[:,i+1]
            #     else:
            #         reward_i *= 0.

            #     valid_mask = np.tile(np.asarray(valid_idx>=0, dtype=np.float32), (1, 2))
            #     valid_mask[:,0] = 0
            #     valid_mask[:,seq_len] = 1

            #     valid_idx[valid_idx==-1] = seq_len
            #     left_idx [left_idx ==-1] = seq_len
            #     right_idx[right_idx==-1] = seq_len

            #     head_idx [head_idx ==-1] = seq_len
            #     child_idx[child_idx==-1] = seq_len
            #     input_idx = np.concatenate((head_idx.reshape(-1, 1), child_idx.reshape(-1, 1)), axis=1)

            #     rewards_np.append(reward_i)
            #     actions_np.append(action_i)

            handle, train_op, rewards_np, masks = forward(batch)

            # backward & parameter update
            values_np = utils.reward_to_value(rewards_np)
            utils.apply_baseline(values_np, baselines, masks)
            sess.partial_run(handle, train_op, feed_dict={value:value_np for value, value_np in zip(values, values_np)})

            # accumulate statistics
            expected_value += np.sum(masks[:,1:])
            observed_value += np.where(np.array(rewards_np)==1)[0].shape[0]

            # display information
            if uidx % FLAGS.disp_freq == 0:
                print '[%d-%d] exp. %.4f, obs. %.4f, acc: %.4f, baselines: %s' % (
                            eidx, uidx, expected_value / 50., observed_value / 50., observed_value / expected_value,
                            ' '.join(['%.2f' % bl for bl in baselines[max_len-1:0:-1]])
                        )

                expected_value, observed_value = 0., 0.
            uidx += 1
            
        # validation after each epoch
        if dev_data is not None and eidx > 0 and eidx % FLAGS.valid_freq == 0:
            dev_expected_value, dev_observed_value = 0., 0.
            for dev_batch in dataio.iterate_batch(dev_data, FLAGS.batch_size, shuffle=True):
                dev_rewards_np, dev_masks = forward(dev_batch, training=False)

                dev_expected_value += np.sum(dev_masks[:,1:])
                dev_observed_value += np.where(np.array(dev_rewards_np)==1)[0].shape[0]
            print '[Dev %d] exp. %.4f, obs. %.4f, acc: %.4f' % (eidx, dev_expected_value, dev_observed_value, dev_observed_value / dev_expected_value)
