# coding:utf-8
import tensorflow as tf
from utils import *
from tensorflow.python.ops.rnn_cell import GRUCell
import logging
from tensorflow.nn import dynamic_rnn
import mimn


# import mann_simple_cell as mann_cell
class Model(object):
    def __init__(self, uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE,
                 MEMORY_SIZE, BATCH_SIZE, SEQ_LEN, use_negsample=False, Flag="DNN", args=None):
        self.model_flag = Flag
        self.use_negsample = use_negsample
        self.use_vec_loss = args.use_vec_loss
        self.att_scores = None
        with tf.name_scope('Inputs'):
            self.item_id_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='item_id_his_batch_ph')
            self.time_id_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='time_id_his_batch_ph')
            self.cate_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='cate_his_batch_ph')
            self.shop_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='shop_his_batch_ph')
            self.node_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='node_his_batch_ph')
            self.product_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='product_his_batch_ph')
            self.brand_his_batch_ph = tf.placeholder(tf.int32, [None, None], name='brand_his_batch_ph')
            self.uid_batch_ph = tf.placeholder(tf.int32, [None, ], name='uid_batch_ph')
            self.item_id_batch_ph = tf.placeholder(tf.int32, [None, ], name='item_id_batch_ph')
            self.time_id_batch_ph = tf.placeholder(tf.int32, [None, ], name='time_id_batch_ph')
            self.cate_id_batch_ph = tf.placeholder(tf.int32, [None, ], name='cate_id_batch_ph')
            self.shop_id_batch_ph = tf.placeholder(tf.int32, [None, ], name='shop_id_batch_ph')
            self.node_id_batch_ph = tf.placeholder(tf.int32, [None, ], name='node_id_batch_ph')
            self.product_id_batch_ph = tf.placeholder(tf.int32, [None, ], name='product_id_batch_ph')
            self.brand_id_batch_ph = tf.placeholder(tf.int32, [None, ], name='brand_id_batch_ph')
            self.mask = tf.placeholder(tf.float32, [None, None], name='mask_batch_ph')
            self.target_ph = tf.placeholder(tf.float32, [None, 2], name='target_ph')
            self.lr = tf.placeholder(tf.float64, [])

            with tf.name_scope('Embedding_layer'):
                self.item_id_embeddings_var = tf.get_variable("item_id_embedding_var", [item_n, EMBEDDING_DIM],
                                                              trainable=True)
                self.item_id_batch_embedded = tf.nn.embedding_lookup(self.item_id_embeddings_var, self.item_id_batch_ph)
                self.item_id_his_batch_embedded = tf.nn.embedding_lookup(self.item_id_embeddings_var,
                                                                         self.item_id_his_batch_ph)

                self.cate_id_embeddings_var = tf.get_variable("cate_id_embedding_var", [cate_n, EMBEDDING_DIM],
                                                              trainable=True)
                self.cate_id_batch_embedded = tf.nn.embedding_lookup(self.cate_id_embeddings_var, self.cate_id_batch_ph)
                self.cate_his_batch_embedded = tf.nn.embedding_lookup(self.cate_id_embeddings_var,
                                                                      self.cate_his_batch_ph)

                self.shop_id_embeddings_var = tf.get_variable("shop_id_embedding_var", [shop_n, EMBEDDING_DIM],
                                                              trainable=True)
                self.shop_id_batch_embedded = tf.nn.embedding_lookup(self.shop_id_embeddings_var, self.shop_id_batch_ph)
                self.shop_his_batch_embedded = tf.nn.embedding_lookup(self.shop_id_embeddings_var,
                                                                      self.shop_his_batch_ph)

                self.node_id_embeddings_var = tf.get_variable("node_id_embedding_var", [node_n, EMBEDDING_DIM],
                                                              trainable=True)
                self.node_id_batch_embedded = tf.nn.embedding_lookup(self.node_id_embeddings_var, self.node_id_batch_ph)
                self.node_his_batch_embedded = tf.nn.embedding_lookup(self.node_id_embeddings_var,
                                                                      self.node_his_batch_ph)

                self.product_id_embeddings_var = tf.get_variable("product_id_embedding_var", [product_n, EMBEDDING_DIM],
                                                                 trainable=True)
                self.product_id_batch_embedded = tf.nn.embedding_lookup(self.product_id_embeddings_var,
                                                                        self.product_id_batch_ph)
                self.product_his_batch_embedded = tf.nn.embedding_lookup(self.product_id_embeddings_var,
                                                                         self.product_his_batch_ph)
                self.brand_id_embeddings_var = tf.get_variable("brand_id_embedding_var", [brand_n, EMBEDDING_DIM],
                                                               trainable=True)
                self.brand_id_batch_embedded = tf.nn.embedding_lookup(self.brand_id_embeddings_var,
                                                                      self.brand_id_batch_ph)
                self.brand_his_batch_embedded = tf.nn.embedding_lookup(self.brand_id_embeddings_var,
                                                                       self.brand_his_batch_ph)

                self.time_id_embeddings_var = tf.get_variable("time_id_embedding_var", [2000, args.time_embedding_dim],
                                                              trainable=True)
                self.time_id_his_batch_embedded = tf.nn.embedding_lookup(self.time_id_embeddings_var,
                                                                         self.time_id_his_batch_ph)
                self.time_id_batch_embedded = tf.nn.embedding_lookup(self.time_id_embeddings_var, self.time_id_batch_ph)

                self.cate_id_batch_embeddeds = []
                self.cate_id_his_batch_embeddeds = []

        with tf.name_scope('init_operation'):
            self.item_id_embedding_placeholder = tf.placeholder(tf.float32, [item_n, EMBEDDING_DIM],
                                                                name="item_id_emb_ph")
            self.item_id_embedding_init = self.item_id_embeddings_var.assign(self.item_id_embedding_placeholder)
            self.cate_id_embedding_placeholder = tf.placeholder(tf.float32, [cate_n, EMBEDDING_DIM],
                                                                name="cate_id_emb_ph")
            self.cate_id_embedding_init = self.cate_id_embeddings_var.assign(self.cate_id_embedding_placeholder)
            self.shop_id_embedding_placeholder = tf.placeholder(tf.float32, [shop_n, EMBEDDING_DIM],
                                                                name="shop_id_emb_ph")
            self.shop_id_embedding_init = self.shop_id_embeddings_var.assign(self.shop_id_embedding_placeholder)
            self.node_id_embedding_placeholder = tf.placeholder(tf.float32, [node_n, EMBEDDING_DIM],
                                                                name="node_id_emb_ph")
            self.node_id_embedding_init = self.node_id_embeddings_var.assign(self.node_id_embedding_placeholder)
            self.product_id_embedding_placeholder = tf.placeholder(tf.float32, [product_n, EMBEDDING_DIM],
                                                                   name="product_id_emb_ph")
            self.product_id_embedding_init = self.product_id_embeddings_var.assign(
                self.product_id_embedding_placeholder)
            self.brand_id_embedding_placeholder = tf.placeholder(tf.float32, [brand_n, EMBEDDING_DIM],
                                                                 name="brand_id_emb_ph")
            self.brand_id_embedding_init = self.brand_id_embeddings_var.assign(self.brand_id_embedding_placeholder)

        if self.use_negsample:
            self.item_id_neg_batch_ph = tf.placeholder(tf.int32, [None, None], name='neg_his_batch_ph')
            self.cate_neg_batch_ph = tf.placeholder(tf.int32, [None, None], name='neg_cate_his_batch_ph')
            self.shop_neg_batch_ph = tf.placeholder(tf.int32, [None, None], name='neg_shop_his_batch_ph')
            self.node_neg_batch_ph = tf.placeholder(tf.int32, [None, None], name='neg_node_his_batch_ph')
            self.product_neg_batch_ph = tf.placeholder(tf.int32, [None, None], name='neg_product_his_batch_ph')
            self.brand_neg_batch_ph = tf.placeholder(tf.int32, [None, None], name='neg_brand_his_batch_ph')
            self.neg_item_his_eb = tf.nn.embedding_lookup(self.item_id_embeddings_var, self.item_id_neg_batch_ph)
            self.neg_cate_his_eb = tf.nn.embedding_lookup(self.cate_id_embeddings_var, self.cate_neg_batch_ph)
            self.neg_shop_his_eb = tf.nn.embedding_lookup(self.shop_id_embeddings_var, self.shop_neg_batch_ph)
            self.neg_node_his_eb = tf.nn.embedding_lookup(self.node_id_embeddings_var, self.node_neg_batch_ph)
            self.neg_product_his_eb = tf.nn.embedding_lookup(self.product_id_embeddings_var, self.product_neg_batch_ph)
            self.neg_brand_his_eb = tf.nn.embedding_lookup(self.brand_id_embeddings_var, self.brand_neg_batch_ph)
            self.neg_his_eb = tf.concat(
                [self.neg_item_his_eb, self.neg_cate_his_eb, self.neg_shop_his_eb, self.neg_node_his_eb,
                 self.neg_product_his_eb, self.neg_brand_his_eb], axis=2) * tf.reshape(self.mask,
                                                                                       (BATCH_SIZE, SEQ_LEN, 1))
            if args.data_type == 'book':
                self.neg_his_eb = tf.concat(
                    [self.neg_item_his_eb, self.neg_cate_his_eb], axis=2) * tf.reshape(self.mask,
                                                                                       (BATCH_SIZE, SEQ_LEN, 1))

        self.item_eb = tf.concat([self.item_id_batch_embedded, self.cate_id_batch_embedded, self.shop_id_batch_embedded,
                                  self.node_id_batch_embedded, self.product_id_batch_embedded,
                                  self.brand_id_batch_embedded], axis=1)
        self.item_his_eb = tf.concat(
            [self.item_id_his_batch_embedded, self.cate_his_batch_embedded, self.shop_his_batch_embedded,
             self.node_his_batch_embedded, self.product_his_batch_embedded, self.brand_his_batch_embedded],
            axis=2) * tf.reshape(self.mask, (BATCH_SIZE, SEQ_LEN, 1))
        if args.data_type == 'book':
            self.item_eb = tf.concat(
                [self.item_id_batch_embedded, self.cate_id_batch_embedded], axis=1)
            self.item_his_eb = tf.concat(
                [self.item_id_his_batch_embedded, self.cate_his_batch_embedded],
                axis=2) * tf.reshape(self.mask, (BATCH_SIZE, SEQ_LEN, 1))

        if args.use_time:
            logging.info("use time embedding!")
            if args.use_time_mode == 'concat':
                self.item_eb = tf.concat([self.item_eb, self.time_id_batch_embedded], axis=-1)
                self.item_his_eb = tf.concat([self.item_his_eb, self.time_id_his_batch_embedded], axis=-1)

        self.inputs = []

        if args.short_model_type == 'DIN' and args.short_seq_split:
            seq_split = [(int(x.split(":")[0]), int(x.split(":")[1])) for x in args.short_seq_split.split(",")]
            for idx, (left_idx, right_idx) in enumerate(seq_split):
                with tf.name_scope('short_din_layer_{0}'.format(idx)):
                    logging.info("short att layer {0}:{1}".format(left_idx, right_idx))
                    mask = self.mask[:, left_idx:right_idx]
                    attention_output = din_attention(self.item_eb, self.item_his_eb[:, left_idx:right_idx], HIDDEN_SIZE,
                                                     mask, stag='short_att_{0}'.format(idx), return_alphas=False)
                    att_fea = tf.reduce_sum(attention_output, 1)
                    self.inputs.append(att_fea)

        if args.short_model_type == 'DNN' and args.short_seq_split:
            seq_split = [(int(x.split(":")[0]), int(x.split(":")[1])) for x in args.short_seq_split.split(",")]
            for idx, (left_idx, right_idx) in enumerate(seq_split):
                with tf.name_scope('short_dnn_layer_{0}'.format(idx)):
                    logging.info("short layer {0}:{1}".format(left_idx, right_idx))
                    mask = self.mask[:, left_idx:right_idx]
                    item_his_sum_emb = tf.reduce_sum(self.item_his_eb[:, left_idx:right_idx] * mask[:, :, None], 1) / (
                                tf.reduce_sum(mask, 1, keepdims=True) + 1.0)
                    self.inputs.append(item_his_sum_emb)

        self.item_his_eb_sum = tf.reduce_sum(self.item_his_eb, 1)

    def build_fcn_net(self, inp, use_dice=False):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1')
        dnn1 = prelu(dnn1, scope='prelu_1')

        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name='f2')
        dnn2 = prelu(dnn2, scope='prelu_2')
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3')
        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        with tf.name_scope('Metrics'):
            # Cross-entropy loss and optimizer initialization
            ctr_loss = - tf.reduce_mean(tf.log(self.y_hat) * self.target_ph)
            self.loss = ctr_loss
            if self.use_negsample:
                self.loss += self.aux_loss
            if self.use_vec_loss:
                logging.info("use_vec_loss!")
                self.loss += self.vec_loss
            tf.summary.scalar('loss', self.loss)
            self.optimizer = tf.contrib.opt.LazyAdamOptimizer(learning_rate=self.lr).minimize(self.loss)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(self.y_hat), self.target_ph), tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)

        self.merged = tf.summary.merge_all()

    def auxiliary_loss(self, h_states, click_seq, noclick_seq, mask=None, stag=None):
        click_input_ = tf.concat([h_states, click_seq], -1)
        noclick_input_ = tf.concat([h_states, noclick_seq], -1)
        click_prop_ = self.auxiliary_net(click_input_, stag=stag)[:, :, 0]
        noclick_prop_ = self.auxiliary_net(noclick_input_, stag=stag)[:, :, 0]

        click_loss_ = - tf.reshape(tf.log(click_prop_), [-1, tf.shape(click_seq)[1]]) * mask
        noclick_loss_ = - tf.reshape(tf.log(1.0 - noclick_prop_), [-1, tf.shape(noclick_seq)[1]]) * mask

        loss_ = tf.reduce_mean(click_loss_ + noclick_loss_)
        return loss_

    def vec_auxiliary_loss(self, query, facts, mask, hidden_size):
        inputs = [self.item_eb]
        attention_output = din_attention(query, facts,
                                         mask=mask, att_func='dot', stag='att_vec_auxiliary')
        att_fea = tf.reduce_sum(attention_output, 1)
        inputs.append(att_fea)

        inp = tf.concat(inputs, -1)
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1_vec')
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name='f1_vec')
        dnn1 = prelu(dnn1, scope='dice_1_vec')
        dnn3 = tf.layers.dense(dnn1, 2, activation=None, name='f3_vec')
        y_hat = tf.nn.softmax(dnn3) + 0.00000001
        vec_auxiliary_loss = - tf.reduce_mean(tf.log(y_hat) * self.target_ph)
        return vec_auxiliary_loss

    def auxiliary_net(self, in_, stag='auxiliary_net'):
        bn1 = tf.layers.batch_normalization(inputs=in_, name='bn1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.layers.dense(bn1, 100, activation=None, name='f1' + stag, reuse=tf.AUTO_REUSE)
        dnn1 = tf.nn.sigmoid(dnn1)
        dnn2 = tf.layers.dense(dnn1, 50, activation=None, name='f2' + stag, reuse=tf.AUTO_REUSE)
        dnn2 = tf.nn.sigmoid(dnn2)
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name='f3' + stag, reuse=tf.AUTO_REUSE)
        y_hat = tf.nn.softmax(dnn3) + 0.000001
        return y_hat

    def init_uid_weight(self, sess, uid_weight):
        sess.run(self.uid_embedding_init, feed_dict={self.uid_embedding_placeholder: uid_weight})

    def init_item_id_weight(self, sess, item_id_weight):
        sess.run([self.item_id_embedding_init], feed_dict={self.item_id_embedding_placeholder: item_id_weight})

    def save_item_id_embedding_weight(self, sess):
        embedding = sess.run(self.item_id_embeddings_var)
        return embedding

    def save_uid_embedding_weight(self, sess):
        embedding = sess.run(self.uid_bp_memory)
        return embedding

    def get_feed_dict(self, inps):
        feed_dict = {
            self.uid_batch_ph: inps['uid_batch_ph'],
            self.item_id_batch_ph: inps['item_id_batch_ph'],
            self.cate_id_batch_ph: inps['cate_id_batch_ph'],
            self.shop_id_batch_ph: inps['shop_id_batch_ph'],
            self.node_id_batch_ph: inps['node_id_batch_ph'],
            self.product_id_batch_ph: inps['product_id_batch_ph'],
            self.brand_id_batch_ph: inps['brand_id_batch_ph'],
            self.item_id_his_batch_ph: inps['item_id_his_batch_ph'],
            self.cate_his_batch_ph: inps['cate_his_batch_ph'],
            self.shop_his_batch_ph: inps['shop_his_batch_ph'],
            self.node_his_batch_ph: inps['node_his_batch_ph'],
            self.product_his_batch_ph: inps['product_his_batch_ph'],
            self.brand_his_batch_ph: inps['brand_his_batch_ph'],
            self.mask: inps['mask'],
            self.target_ph: inps['target_ph'],
            self.time_id_batch_ph: inps['time_id_batch_ph'],
            self.time_id_his_batch_ph: inps['time_id_his_batch_ph'],
        }
        if self.use_negsample:
            feed_dict[self.item_id_neg_batch_ph] = inps['item_id_neg_batch_ph']
            feed_dict[self.cate_neg_batch_ph] = inps['cate_neg_batch_ph']
            feed_dict[self.shop_neg_batch_ph] = inps['shop_neg_batch_ph']
            feed_dict[self.node_neg_batch_ph] = inps['node_neg_batch_ph']
            feed_dict[self.product_neg_batch_ph] = inps['product_neg_batch_ph']
            feed_dict[self.brand_neg_batch_ph] = inps['brand_neg_batch_ph']

        return feed_dict

    def train(self, sess, inps):
        feed_dict = self.get_feed_dict(inps)
        feed_dict[self.lr] = inps['lr']

        if self.use_negsample:
            loss, aux_loss, accuracy, _ = sess.run([self.loss, self.aux_loss, self.accuracy, self.optimizer],
                                                   feed_dict=feed_dict)
        else:
            loss, aux_loss, accuracy, _ = sess.run([self.loss, self.loss, self.accuracy, self.optimizer],
                                                   feed_dict=feed_dict)
        return loss, accuracy, aux_loss, 0, 0

    def calculate(self, sess, inps):
        feed_dict = self.get_feed_dict(inps)

        if self.use_negsample:
            probs, loss, aux_loss, accuracy = sess.run([self.y_hat, self.loss, self.aux_loss, self.accuracy],
                                                       feed_dict=feed_dict)
        else:
            probs, loss, aux_loss, accuracy = sess.run([self.y_hat, self.loss, self.loss, self.accuracy],
                                                       feed_dict=feed_dict)
        return probs, loss, accuracy, aux_loss, None

    def save(self, sess, path):
        saver = tf.train.Saver()
        saver.save(sess, save_path=path)

    def restore(self, sess, path):
        saver = tf.train.Saver()
        saver.restore(sess, save_path=path)
        print('model restored from %s' % path)


class Model_DNN(Model):
    def __init__(self, uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE,
                 MEMORY_SIZE, BATCH_SIZE, SEQ_LEN=256, args=None):
        super(Model_DNN, self).__init__(uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM,
                                        HIDDEN_SIZE, MEMORY_SIZE,
                                        BATCH_SIZE, SEQ_LEN, Flag="DNN", args=args)

        inputs = self.inputs + [self.item_eb]
        if args and args.long_seq_split:
            seq_split = [(int(x.split(":")[0]), int(x.split(":")[1])) for x in args.long_seq_split.split(",")]
            for idx, (left_idx, right_idx) in enumerate(seq_split):
                with tf.name_scope('long_att_layer_{0}'.format(idx)):
                    mask = self.mask[:, left_idx:right_idx]
                    self.vec_loss = self.vec_auxiliary_loss(self.item_eb, self.item_his_eb[:, left_idx:right_idx], mask,
                                                            HIDDEN_SIZE)

                    attention_output, scores = din_attention(self.item_eb, self.item_his_eb[:, left_idx:right_idx],
                                                             mask=self.mask[:, left_idx:right_idx], att_func='dot',
                                                             return_alphas=True,
                                                             stag='att_vec_{0}'.format(idx))
                    top_k = args.first_att_top_k
                    scores -= top_kth_iterative(scores, top_k)
                    if args.level.lower() == 'debug':
                        scores = tf.Print(scores, ["score:", scores[0]], summarize=1000)
                    if args.use_first_att:
                        mask = tf.cast(tf.greater(scores, tf.zeros_like(scores)), tf.float32)
                        if args.level.lower() == 'debug':
                            mask = tf.Print(mask, ["mask:", mask[0]], summarize=1000)

                    item_his_sum_emb = tf.reduce_sum(self.item_his_eb[:, left_idx:right_idx] * mask[:, :, None], 1) / (
                                tf.reduce_sum(mask, 1, keepdims=True) + 1.0)
                    inputs.append(item_his_sum_emb)

        logging.info(inputs)
        inp = tf.concat(inputs, 1)

        self.build_fcn_net(inp, use_dice=False)


class Model_DIN(Model):
    def __init__(self, uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE,
                 MEMORY_SIZE, BATCH_SIZE, SEQ_LEN=256, args=None):
        super(Model_DIN, self).__init__(uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM,
                                        HIDDEN_SIZE, MEMORY_SIZE,
                                        BATCH_SIZE, SEQ_LEN, Flag="DIN", args=args)

        inputs = self.inputs + [self.item_eb]
        if args and args.long_seq_split:
            seq_split = [(int(x.split(":")[0]), int(x.split(":")[1])) for x in args.long_seq_split.split(",")]
            for idx, (left_idx, right_idx) in enumerate(seq_split):
                with tf.name_scope('long_att_layer_{0}'.format(idx)):
                    mask = self.mask[:, left_idx:right_idx]
                    self.vec_loss = self.vec_auxiliary_loss(self.item_eb, self.item_his_eb[:, left_idx:right_idx], mask,
                                                            HIDDEN_SIZE)

                    attention_output, scores = din_attention(self.item_eb, self.item_his_eb[:, left_idx:right_idx],
                                                             mask=self.mask[:, left_idx:right_idx], att_func='dot',
                                                             return_alphas=True,
                                                             stag='att_vec_{0}'.format(idx))
                    top_k = args.first_att_top_k
                    scores -= top_kth_iterative(scores, top_k)
                    if args.level.lower() == 'debug':
                        scores = tf.Print(scores, ["score:", scores[0]], summarize=1000)
                    if args.use_first_att:
                        mask = tf.cast(tf.greater(scores, tf.zeros_like(scores)), tf.float32)
                        if args.level.lower() == 'debug':
                            mask = tf.Print(mask, ["mask:", mask[0]], summarize=1000)
                    att_func = args.att_func
                    attention_output, scores = din_attention(self.item_eb, self.item_his_eb[:, left_idx:right_idx],
                                                             HIDDEN_SIZE, mask, att_func=att_func,
                                                             stag='att_{0}'.format(idx), return_alphas=True)
                    self.att_scores = scores
                    att_fea = tf.reduce_sum(attention_output, 1)
                    inputs.append(att_fea)

                    item_his_sum_emb = tf.reduce_sum(self.item_his_eb[:, left_idx:right_idx] * mask[:, :, None], 1) / (
                                tf.reduce_sum(mask, 1, keepdims=True) + 1.0)
                    inputs.append(item_his_sum_emb)

        logging.info(inputs)
        inp = tf.concat(inputs, 1)

        self.build_fcn_net(inp, use_dice=False)


class Model_MIMN(Model):
    def __init__(self, uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE,
                 MEMORY_SIZE, BATCH_SIZE, SEQ_LEN=400, Mem_Induction=0,
                 Util_Reg=0, use_negsample=False, mask_flag=False, args=None):
        super(Model_MIMN, self).__init__(uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM,
                                         HIDDEN_SIZE, MEMORY_SIZE,
                                         BATCH_SIZE, SEQ_LEN, use_negsample, Flag="MIMN", args=args)
        logging.info(locals())

        self.reg = args.util_reg
        seq_reduce = args.mimn_seq_reduce
        Mem_Induction = args.mem_induction
        MEMORY_SIZE = args.memory_size

        def clear_mask_state(state, begin_state, begin_channel_rnn_state, mask, cell, t):
            state["controller_state"] = (1 - tf.reshape(mask[:, t], (BATCH_SIZE, 1))) * begin_state[
                "controller_state"] + tf.reshape(mask[:, t], (BATCH_SIZE, 1)) * state["controller_state"]
            state["M"] = (1 - tf.reshape(mask[:, t], (BATCH_SIZE, 1, 1))) * begin_state["M"] + tf.reshape(mask[:, t], (
                BATCH_SIZE, 1, 1)) * state["M"]
            state["key_M"] = (1 - tf.reshape(mask[:, t], (BATCH_SIZE, 1, 1))) * begin_state["key_M"] + tf.reshape(
                mask[:, t], (BATCH_SIZE, 1, 1)) * state["key_M"]
            state["sum_aggre"] = (1 - tf.reshape(mask[:, t], (BATCH_SIZE, 1, 1))) * begin_state[
                "sum_aggre"] + tf.reshape(mask[:, t], (BATCH_SIZE, 1, 1)) * state["sum_aggre"]
            if Mem_Induction > 0:
                temp_channel_rnn_state = []
                for i in range(MEMORY_SIZE):
                    temp_channel_rnn_state.append(
                        cell.channel_rnn_state[i] * tf.expand_dims(mask[:, t], axis=1) + begin_channel_rnn_state[i] * (
                                1 - tf.expand_dims(mask[:, t], axis=1)))
                cell.channel_rnn_state = temp_channel_rnn_state
                temp_channel_rnn_output = []
                for i in range(MEMORY_SIZE):
                    temp_output = cell.channel_rnn_output[i] * tf.expand_dims(mask[:, t], axis=1) + \
                                  begin_channel_rnn_output[i] * (1 - tf.expand_dims(self.mask[:, t], axis=1))
                    temp_channel_rnn_output.append(temp_output)
                cell.channel_rnn_output = temp_channel_rnn_output

            return state

        inputs = self.inputs

        mimn_seq_split = args.mimn_seq_split if args.mimn_seq_split else args.long_seq_split
        if args and mimn_seq_split:
            seq_split = [(int(x.split(":")[0]), int(x.split(":")[1])) for x in mimn_seq_split.split(",")]
            for idx, (left_idx, right_idx) in enumerate(seq_split):
                SEQ_LEN = abs(right_idx - left_idx)
                with tf.name_scope('MIMN_Layer_{0}'.format(idx)):
                    logging.info("mimn_layer {0}:{1}".format(left_idx, right_idx))
                    mask = self.mask[:, left_idx:right_idx]

                item_his_eb = self.item_his_eb[:, left_idx:right_idx] * mask[:, :, None]
                if self.use_negsample:
                    neg_his_eb = self.neg_his_eb[:, left_idx:right_idx] * mask[:, :, None]
                item_eb = self.item_eb
                if args.mimn_update_emb == 0:
                    item_his_eb = tf.stop_gradient(item_his_eb)
                    item_eb = tf.stop_gradient(item_eb)
                    if self.use_negsample:
                        neg_his_eb = tf.stop_gradient(neg_his_eb)
                memory_vector_dim = self.item_his_eb.get_shape().as_list()[-1]
                head_num = 1
                if args.head_num:
                    head_num = args.head_num

                cell = mimn.MIMNCell(controller_units=HIDDEN_SIZE, memory_size=MEMORY_SIZE,
                                     memory_vector_dim=memory_vector_dim,
                                     read_head_num=head_num, write_head_num=head_num,
                                     reuse=False, output_dim=HIDDEN_SIZE, clip_value=100, batch_size=BATCH_SIZE,
                                     mem_induction=Mem_Induction, util_reg=Util_Reg)

                state = cell.zero_state(BATCH_SIZE, tf.float32)
                if Mem_Induction > 0:
                    begin_channel_rnn_output = cell.channel_rnn_output
                else:
                    begin_channel_rnn_output = 0.0

                begin_state = state
                self.state_list = [state]
                self.mimn_o = []

                if args.mimn_seq_reduce:
                    logging.info("mimn_seq_reduce:{0}".format(args.mimn_seq_reduce))
                    seq_reduce = args.mimn_seq_reduce
                    SEQ_LEN = int(SEQ_LEN / seq_reduce)

                    dim = item_his_eb.get_shape().as_list()[-1]
                    logging.info(dim)
                    item_his_eb = tf.reshape(item_his_eb, [-1, SEQ_LEN, seq_reduce, dim])
                    item_his_eb = tf.reduce_mean(item_his_eb, axis=-2)
                    logging.info(item_his_eb.get_shape())

                    mask = tf.reshape(mask, [-1, SEQ_LEN, seq_reduce])
                    mask = tf.reduce_sum(mask, axis=-1)
                    mask = tf.cast(tf.not_equal(mask, tf.zeros_like(mask)), tf.float32)

                for t in range(SEQ_LEN):
                    output, state, temp_output_list = cell(item_his_eb[:, t, :], state)
                    if mask_flag:
                        state = clear_mask_state(state, begin_state, begin_channel_rnn_output, mask, cell, t)
                    self.mimn_o.append(output)
                    self.state_list.append(state)

                self.mimn_o = tf.stack(self.mimn_o, axis=1)
                self.state_list.append(state)
                mean_memory = tf.reduce_mean(state['sum_aggre'], axis=-2)

                before_aggre = state['w_aggre']
                read_out, _, _ = cell(item_eb, state)

                if use_negsample:
                    aux_loss_1 = self.auxiliary_loss(self.mimn_o[:, :-1, :], item_his_eb[:, 1:, :],
                                                     neg_his_eb[:, 1:, :], mask[:, 1:], stag="bigru_0")
                    self.aux_loss = aux_loss_1

                if self.reg:
                    self.reg_loss = cell.capacity_loss(before_aggre)
                else:
                    self.reg_loss = tf.zeros(1)

                if Mem_Induction == 1:
                    channel_memory_tensor = tf.concat(temp_output_list, 1)
                    multi_channel_hist = din_attention(item_eb, channel_memory_tensor, HIDDEN_SIZE, None, stag='pal',
                                                       att_func=args.att_func)
                    inputs += [read_out, tf.squeeze(multi_channel_hist),
                               mean_memory * item_eb]
                else:
                    inputs += [read_out, mean_memory * item_eb]

        if args and args.long_seq_split:
            seq_split = [(int(x.split(":")[0]), int(x.split(":")[1])) for x in args.long_seq_split.split(",")]
            for idx, (left_idx, right_idx) in enumerate(seq_split):
                mask = self.mask[:, left_idx:right_idx]
                item_his_sum_emb = tf.reduce_sum(self.item_his_eb[:, left_idx:right_idx] * mask[:, :, None], 1) / (
                            tf.reduce_sum(mask, 1, keepdims=True) + 1.0)
            inputs.append(item_his_sum_emb)

        inp = tf.concat(inputs, 1)
        self.build_fcn_net(inp, use_dice=False)
