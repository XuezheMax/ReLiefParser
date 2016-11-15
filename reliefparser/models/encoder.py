import numpy as np
import tensorflow as tf

class Encoder(object):
    def __init__(self, vsize, esize, hsize, rnn_class, **kwargs):
        super(Encoder, self).__init__()

        self.name  = kwargs.get('name', self.__class__.__name__)
        self.scope = kwargs.get('scope', self.name)
        
        self.vsize = vsize  # vocabulary size

        self.esize = esize  # embedding size
        self.hsize = hsize  # hidden size

        self.num_layer = kwargs.get('num_layer', 1)

        self.rnn_cell_fw = tf.nn.rnn_cell.MultiRNNCell([rnn_class(num_units=self.hsize)] * self.num_layer)
        self.rnn_cell_bw = tf.nn.rnn_cell.MultiRNNCell([rnn_class(num_units=self.hsize)] * self.num_layer)

        self.embed_initializer = tf.random_normal_initializer(mean=0.0, stddev=1.)

    def __call__(self, input):
        with tf.variable_scope(self.scope):
            # embedding
            embeddings = tf.get_variable('embeddings', shape=[self.vsize, self.esize],
                                         initializer=self.embed_initializer)
            embed  = tf.nn.embedding_lookup(embeddings, input)
            
            # forward rnn
            with tf.variable_scope('fw_rnn'):
                hiddens_fw, final_state_fw = tf.nn.dynamic_rnn(self.rnn_cell_fw, embed, dtype=tf.float32)

            # backward rnn
            with tf.variable_scope('bw_rnn'):
                hiddens_bw, final_state_bw = tf.nn.dynamic_rnn(self.rnn_cell_bw, embed[:,-1::-1], dtype=tf.float32)
                hiddens_bw = hiddens_bw[:,-1::-1]

            # concatenate
            hiddens = tf.concat(2, [hiddens_fw, hiddens_bw])

        return hiddens, final_state_fw, final_state_bw


#### test script
if __name__ == '__main__':
    vsize = 100
    esize = 256
    hsize = 256

    encoder = Encoder(vsize, esize, hsize)

    enc_input = tf.placeholder(dtype=tf.int32, shape=[None, None])

    outputs, final_state_fw, final_state_bw = encoder(enc_input)

    # with tf.variable_scope('Encoder'):
    #     embeddings = tf.Variable(tf.random_uniform([vsize, esize], -1.0, 1.0), 'embeddings')
    #     enc_embed  = tf.nn.embedding_lookup(embeddings, enc_input)
    #     outputs, final_state_fw, final_state_bw = tf.nn.dynamic_rnn(tf.nn.rnn_cell.GRUCell(num_units=hsize), enc_embed, dtype=tf.float32)

    lsize = 10
    bsize = 32

    params = tf.trainable_variables()

    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)

        enc_input_np = np.random.randint(0, vsize, size=[bsize, lsize])
        feed_dict = {enc_input: enc_input_np}
        outputs_np, states_np = sess.run([outputs, final_state_fw, final_state_bw], feed_dict=feed_dict)
        print outputs_np.shape
        print states_np.shape
