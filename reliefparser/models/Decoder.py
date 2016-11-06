import tensorflow as tf
import numpy as np
from time import time
import bisect
from scipy import stats

_buckets = [10, 20]#, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

max_len = _buckets[-1]

# placeholders
inputs = []
rewards = []
for i in range(max_len):
    inputs.append(tf.placeholder(dtype=tf.float32, name='input_%d'%i))
    rewards.append(tf.placeholder(dtype=tf.float32, name='reward_%d'%i))
init_hidden = tf.placeholder(dtype=tf.float32, name='init_hidden')

lsize = 20
bsize = 128
isize = 512
msize = 512
asize = 128

# Comes from the encoder
memory = tf.ones([bsize, lsize, msize])

############
# weights used to combine head and child
# - MLP vs Bilinear

# weights of the Embedding
# - 100k
# - intialization: Random, GloVe, Word2Vec

# weights of the Recurrent Nets
# - GRU vs LSTM
# - Vanilla RNN

# weights of the Pointer Nets
# - Memory side: left-arc: w_lh, w_lc; right-arc: w_rh, w_rc
# - Context side: w_ctx
# - Score: w_score

########################################
# Trainable variables
########################################

# Vanilla RNN (TODO: using variable scope)
def vanilla_rnn(i_t, h_tm):
    h_t = tf.tanh(tf.matmul(i_t, w_ih) + tf.matmul(h_tm, w_hh) + b_h)
    return h_t
w_ih = tf.Variable(np.random.randn(isize, isize), dtype=tf.float32, name='w_ih')
w_hh = tf.Variable(np.random.randn(isize, isize), dtype=tf.float32, name='w_hh')
b_h  = tf.Variable(np.random.randn(isize), dtype=tf.float32, name='b_h')

# Vanilla Pointer
w_ma = tf.Variable(np.random.randn(1, msize, asize), dtype=tf.float32, name='w_ma')
w_ha = tf.Variable(np.random.randn(isize, asize), dtype=tf.float32, name='w_ha')
w_as = tf.Variable(np.random.randn(asize), dtype=tf.float32, name='w_as')
b_a  = tf.Variable(np.random.randn(asize), dtype=tf.float32, name='b_a')

########################################
# Recurrent computation
########################################

def step(i_t, h_tm):
    h_t = vanilla_rnn(i_t, h_tm)
    a_t = tf.tanh(tf.expand_dims(tf.matmul(h_t, w_ha), 1) + tf.nn.conv1d(memory, w_ma, 1, 'SAME') + b_a)
    s_t = tf.reduce_sum(a_t * w_as, [2])
    p_t = tf.nn.softmax(s_t)
    o_t = tf.to_int32(tf.argmax(p_t, dimension=1))
    idx_t = tf.range(tf.shape(p_t)[0]) * tf.shape(p_t)[1] + o_t
    po_t = tf.gather(tf.reshape(p_t, [-1]), idx_t)
    return h_t, o_t, po_t

hiddens, outputs, prob_outs = [], [], []
for i in range(max_len):
    h_tm = hiddens[-1] if i > 0 else init_hidden
    i_t = inputs[i]
    h_t, o_t, po_t = step(i_t, h_tm)
    hiddens.append(h_t)
    outputs.append(o_t)
    prob_outs.append(po_t)

costs = []
for prob_out, reward in zip(prob_outs, rewards):
    costs.append(tf.reduce_sum(prob_out * reward))

w = [w_ih,w_hh,b_h,w_ma,w_ha,w_as,b_a]#tf.trainable_variables()
grad_w_buckets = []
for limit in _buckets:
    print '0 ~ %d'%(limit-1)
    grad_w = tf.gradients(costs[:limit], w)
    grad_w_buckets.append(grad_w)

hiddens_np, outputs_np, inputs_np, rewards_np = [], [], [], []

all_feeds = []
all_feeds.extend(inputs)
all_feeds.extend(rewards)
all_feeds.append(init_hidden)

all_fetches = []
all_fetches.extend(hiddens)
all_fetches.extend(outputs)

# get from evironment
def action(output):
    return np.random.randn(bsize, isize), np.ones(output.shape)

with tf.Session() as sess:
    tf.initialize_all_variables().run()

    init_input_np = np.random.randn(bsize, isize)
    init_hidden_np = np.zeros((bsize, isize))

    inputs_np.append(init_input_np)

    bucket_id = bisect.bisect_left(_buckets, lsize)
    grad_w = grad_w_buckets[bucket_id]

    t = time()
    handle = sess.partial_run_setup(all_fetches+grad_w, all_feeds)

    for i in range(lsize):
        feed_dict = {inputs[i]:inputs_np[i]}
        if i == 0:
            feed_dict.update({init_hidden:init_hidden_np})
        h_i_np, o_i_np = sess.partial_run(handle, [hiddens[i], outputs[i]], feed_dict=feed_dict)
        hiddens_np.append(h_i_np)
        outputs_np.append(o_i_np)
        input_i_np, reward_i_np = action(outputs_np[-1])
        inputs_np.append(input_i_np)
        rewards_np.append(reward_i_np)
    print time() - t

    t = time()
    grad_w_np_1 = sess.partial_run(handle, grad_w, feed_dict={go:go_np for go, go_np in zip(rewards, rewards_np)})
    print time() - t

    t = time()
    feed_dict={init_hidden:init_hidden_np}
    for i in range(lsize):
        feed_dict.update({inputs[i]:inputs_np[i]})
        h_i_np, o_i_np = sess.run([hiddens[i], outputs[i]], feed_dict=feed_dict)
        hiddens_np.append(h_i_np)
        outputs_np.append(o_i_np)
        input_i_np, reward_i_np = action(outputs_np[-1])
        inputs_np.append(input_i_np)
        rewards_np.append(reward_i_np)
    print time() - t
    
    t = time()
    feed_dict.update({go:go_np for go, go_np in zip(rewards, rewards_np)})
    grad_w_np_2 = sess.run(grad_w, feed_dict=feed_dict)
    print time() - t

for g1, g2 in zip(grad_w_np_1, grad_w_np_2):
    print np.allclose(g1, g2), np.allclose(g1, np.zeros_like(g1))
    # print stats.describe(g1)
    print np.max(g1), np.min(g1)
# print np.allclose(grad_w_np_1, grad_w_np_2)