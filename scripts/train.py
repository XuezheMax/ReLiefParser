import os, sys
sys.path.append('..')
import json
from time import time
import numpy as np

import tensorflow as tf

import reliefparser.environment as env
import reliefparser.io as dataio
from reliefparser.models import *

DICTDIR = "data/alphabets/"
DATADIR = ".."

np.random.seed(1)
logging = tf.logging

logging.set_verbosity(logging.INFO)

word_alphabet, pos_alphabet, type_alphabet = dataio.create_alphabets("data/alphabets/", ["../sample.conll"], 150000)

data = dataio.read_data("../sample.conll", word_alphabet, pos_alphabet, type_alphabet)

####################
# Hyper-parameters
####################
vsize = 100
esize = 8
hsize = 8
asize = 8
isize = 8

buckets = [10, 20]#, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
max_len = buckets[-1]

bsize = 1

####################
# Building model
####################

# model initialization
pointer_net = PointerNet(vsize, esize, hsize, asize, buckets, dec_isize=isize)

# placeholders
enc_input = tf.placeholder(dtype=tf.int32, shape=[None, None], name='encoder_input')

input_indices, values, valid_masks = [], [], []
valid_indices, left_indices, right_indices = [], [], []
for i in range(max_len):
    values.append(tf.placeholder(dtype=tf.float32, name='reward_%d'%i))
    valid_masks.append(tf.placeholder(dtype=tf.float32, name='input_index_%d'%i))
    input_indices.append(tf.placeholder(dtype=tf.int32, name='input_index_%d'%i))
    valid_indices.append(tf.placeholder(dtype=tf.int32, name='valid_index_%d'%i))
    left_indices.append (tf.placeholder(dtype=tf.int32, name='left_index_%d'%i))
    right_indices.append(tf.placeholder(dtype=tf.int32, name='right_index_%d'%i))

# build computation graph
dec_hiddens, dec_actions, train_ops = pointer_net(
                                            enc_input, input_indices, valid_indices, 
                                            left_indices, right_indices, values, 
                                            valid_masks=valid_masks)

all_feeds = []
all_feeds.extend(values)
all_feeds.extend(valid_masks)
all_feeds.extend(input_indices)
all_feeds.extend(valid_indices)
all_feeds.extend(right_indices)
all_feeds.extend(left_indices)
all_feeds.append(enc_input)

all_fetches = []
all_fetches.extend(dec_hiddens)
all_fetches.extend(dec_actions)
########## End of section ##########

max_iter = 1000

logging.info('Begin training...')
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    uidx = 0
    wid_inputs, pid_inputs, hid_inputs, tid_inputs, masks = dataio.get_batch(data, bsize)
    print wid_inputs
    print masks
    for eidx in range(max_iter):
        # if eidx > 0 and eidx % 500 == 0:
        #     pointer_net.decoder.epsilon.assign(pointer_net.decoder.epsilon.value() / 2.)
        curr_env = env.Environment(hid_inputs, masks)

        init_vdidx_np, init_ltidx_np, init_rtidx_np = curr_env.get_indexes()
        batch_size, seq_len = init_vdidx_np.shape
        # print seq_len

        # setup partial run handle
        bucket_id = bisect.bisect_left(buckets, seq_len)
        train_op = train_ops[bucket_id]
        handle = sess.partial_run_setup(all_fetches+[train_op], all_feeds)

        # initial data
        init_vdidx_np[init_vdidx_np==-1] = seq_len
        init_ltidx_np[init_ltidx_np==-1] = seq_len
        init_rtidx_np[init_rtidx_np==-1] = seq_len

        init_inidx_np = np.array(np.ones((bsize, 2)) * seq_len).astype(np.int32)

        # numerical data collections
        valid_masks_np, input_indices_np, valid_indices_np, left_indices_np, right_indices_np = [], [], [], [], []
        hiddens_np, actions_np, rewards_np = [], [], []

        init_valid_mask = np.tile(masks, (1,2))
        init_valid_mask[:,0] = 0
        init_valid_mask[:,seq_len] = 1
        valid_masks_np.append(init_valid_mask)
        input_indices_np.append(init_inidx_np)
        valid_indices_np.append(init_vdidx_np)
        left_indices_np.append(init_ltidx_np)
        right_indices_np.append(init_rtidx_np)

        for i in range(seq_len):
            if eidx == max_iter-1:
                print '===== Step %d =====' % i
                print 'va', curr_env.valid_actions()
                print 'ip', input_indices_np[i]
                print 'vm', valid_masks_np[i]
                print 'vd', valid_indices_np[i]
                # print 'lt', left_indices_np[i]
                # print 'rt', right_indices_np[i]
            
            feed_dict = {valid_masks[i]:valid_masks_np[i], 
                         input_indices[i]:input_indices_np[i], 
                         valid_indices[i]:valid_indices_np[i], 
                         left_indices[i]:left_indices_np[i], 
                         right_indices[i]:right_indices_np[i]}
            if i == 0:
                feed_dict.update({enc_input:wid_inputs})

            h_i_np, a_i_np = sess.partial_run(handle, [dec_hiddens[i], dec_actions[i]], feed_dict=feed_dict)
            if eidx == max_iter-1:
                print 'ac', a_i_np

            hiddens_np.append(h_i_np)
            actions_np.append(a_i_np)

            try:
                reward_i, head_idx, child_idx, valid_idx, left_idx, right_idx = curr_env.take_action(a_i_np)
            except:
                print 'ac', a_i_np
                print 'vm', valid_masks_np[i]
                sys.exit()

            valid_mask = np.tile(np.asarray(valid_idx>=0, dtype=np.float32), (1, 2))
            valid_mask[:,0] = 0
            valid_mask[:,seq_len] = 1
            head_idx [head_idx ==-1] = seq_len
            child_idx[child_idx==-1] = seq_len
            valid_idx[valid_idx==-1] = seq_len
            left_idx [left_idx ==-1] = seq_len
            right_idx[right_idx==-1] = seq_len

            input_idx = np.concatenate((head_idx.reshape(-1, 1), child_idx.reshape(-1, 1)), axis=1)

            rewards_np.append(reward_i * masks[:,i])
            valid_masks_np.append(valid_mask)
            input_indices_np.append(input_idx)
            valid_indices_np.append(valid_idx)
            left_indices_np.append(left_idx)
            right_indices_np.append(right_idx)

            R = np.array(rewards_np)
            if eidx == max_iter-1:
                print 'rw', rewards_np[i]
        
        values_np = np.zeros((len(rewards_np), rewards_np[0].shape[0]))
        for i in range(len(values_np)-1,-1,-1):
            values_np[i] = rewards_np[i] + values_np[i+1] * 0.95 if i < len(values_np)-1 else rewards_np[i]

        R = np.array(rewards_np)
        print R[R==1].shape[0], R[R==0].shape[0], R[R==-1].shape[0], ['%.2f' % bl for bl in sess.run(pointer_net.baselines[:seq_len])], [a[0] for a in actions_np]
        sess.partial_run(handle, train_op, feed_dict={reward:reward_np for reward, reward_np in zip(values, values_np)})
        uidx += 1

    for i in range(seq_len):
        print 'baseline %d: %.4f' %(i, sess.run(pointer_net.baselines[i]))