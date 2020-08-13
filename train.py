# coding:utf-8
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from data_iterator import DataIterator
import tensorflow as tf
from model_taobao_allfea import *
import time
import random
import sys
import json
from utils import *
import multiprocessing
from multiprocessing import Process, Value, Array
from wrap_time import time_it
from data_loader import DataLoader
import threading
from collections import deque
import logging


def file_num(x):
    if x < 10:
        return '0' + str(x)
    else:
        return str(x)


EMBEDDING_DIM = 4
HIDDEN_SIZE = EMBEDDING_DIM * 6
MEMORY_SIZE = 4


def generator_queue(generator, max_q_size=20,
                    wait_time=0.1, nb_worker=1):
    generator_threads = []
    q = multiprocessing.Queue(maxsize=max_q_size)
    _stop = multiprocessing.Event()
    try:
        def data_generator_task():
            while not _stop.is_set():
                try:
                    if q.qsize() < max_q_size:
                        generator_output = next(generator)
                        q.put(generator_output)
                    else:
                        time.sleep(wait_time)
                except Exception:
                    _stop.set()

        for i in range(nb_worker):
            thread = multiprocessing.Process(target=data_generator_task)
            generator_threads.append(thread)
            thread.daemon = True
            thread.start()
    except Exception:
        _stop.set()
        for p in generator_threads:
            if p.is_alive():
                p.terminate()
        q.close()

    return q, _stop, generator_threads


def prepare_data(src, target, args=None):
    nick_id, item_id, cate_id = src
    label, hist_item, hist_cate, neg_item, neg_cate, hist_mask = target

    # The time embedding is just for taobao dataset. It is not used in amazion dataset, as there are no
    # timestamp features in it.
    time_his_id = np.ones_like(hist_item)
    time_id = np.asarray(np.ones_like(item_id) * 1024, dtype=np.int32)
    if args.long_seq_split and args.search_mode == 'cate':
        seq_split = [(int(x.split(":")[0]), int(x.split(":")[1])) for x in args.long_seq_split.split(",")]
        for idx, (left_idx, right_idx) in enumerate(seq_split):
            hist_mask[:, left_idx:right_idx] = ((hist_cate == cate_id[:, None]) & (hist_mask > 0))[:,
                                               left_idx:right_idx] * 1.0

    result = {
        'uid_batch_ph': nick_id,
        'item_id_batch_ph': item_id,
        'time_id_batch_ph': time_id,
        'cate_id_batch_ph': cate_id,
        'shop_id_batch_ph': cate_id,
        'node_id_batch_ph': cate_id,
        'product_id_batch_ph': cate_id,
        'brand_id_batch_ph': cate_id,
        'item_id_his_batch_ph': hist_item,
        'cate_his_batch_ph': hist_cate,
        'shop_his_batch_ph': hist_cate,
        'node_his_batch_ph': hist_cate,
        'product_his_batch_ph': hist_cate,
        'brand_his_batch_ph': hist_cate,
        'item_id_neg_batch_ph': neg_item,
        'cate_neg_batch_ph': neg_cate,
        'shop_neg_batch_ph': neg_cate,
        'node_neg_batch_ph': neg_cate,
        'product_neg_batch_ph': neg_cate,
        'brand_neg_batch_ph': neg_cate,
        'mask': hist_mask,
        'time_id_his_batch_ph': time_his_id,
        'target_ph': label
    }
    return result


def eval(sess, test_file, model, model_path, batch_size, maxlen, best_auc=1.0, args=None):
    print("Testing starts------------")
    data_load = DataIterator(test_file, batch_size, maxlen=args.max_len, args=args)
    data_pool, _stop, _ = generator_queue(data_load)

    loss_sum = 0.
    accuracy_sum = 0.
    aux_loss_sum = 0.
    iterations = 0
    stored_arr = []

    while True:
        if _stop.is_set() and data_pool.empty():
            break
        if not data_pool.empty():
            src, tgt = data_pool.get()
        else:
            continue
        data = prepare_data(
            src, tgt, args)

        iterations += 1
        target = data['target_ph']
        prob, loss, acc, aux_loss, att_scores = model.calculate(sess, data)

        loss_sum += loss
        aux_loss_sum = aux_loss
        accuracy_sum += acc
        prob_1 = prob[:, 0].tolist()
        target_1 = target[:, 0].tolist()
        # user_l = user_id.tolist()
        for p, t in zip(prob_1, target_1):
            stored_arr.append([p, t])

    logging.info("test end!")
    test_auc = calc_auc(stored_arr)
    accuracy_sum = accuracy_sum / iterations
    loss_sum = loss_sum / iterations
    aux_loss_sum / iterations
    if best_auc[0] < test_auc:
        best_auc[0] = test_auc
        model.save(sess, model_path)

    result = {
        "auc": test_auc,
        "loss": loss_sum,
        "accuracy": accuracy_sum,
        "aux_loss": aux_loss_sum,
        "best_auc": best_auc[0]
    }
    return result


def train(
        train_file,
        test_file,
        batch_size=256,
        maxlen=1000,
        test_iter=500,
        save_iter=5000,
        model_type='DNN',
        Memory_Size=4,
        Parral_Stag=1,  # 0 no, 1 soft, 2 hard
        Ntm_Flag="learned,0",
        args=None
):
    EMBEDDING_DIM = args.embedding_dim
    TEM_MEMORY_SIZE = Memory_Size
    model_path = "dnn_save_path/taobao_ckpt_noshuff" + model_type
    best_model_path = "dnn_best_model/taobao_ckpt_noshuff" + model_type
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n = [435027, 435027, 435027, 435027, 435027, 435027,
                                                                     435027]
        BATCH_SIZE = batch_size
        SEQ_LEN = args.max_len

        if model_type == 'DNN':
            model = Model_DNN(uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE,
                              MEMORY_SIZE, BATCH_SIZE, SEQ_LEN, args)
        elif model_type == 'DIN':
            model = Model_DIN(uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE,
                              MEMORY_SIZE, BATCH_SIZE, SEQ_LEN, args)
        elif model_type == 'MIMN':
            model = Model_MIMN(uid_n, item_n, cate_n, shop_n, node_n, product_n, brand_n, EMBEDDING_DIM, HIDDEN_SIZE,
                               MEMORY_SIZE=MEMORY_SIZE, BATCH_SIZE=BATCH_SIZE, SEQ_LEN=SEQ_LEN,
                               Mem_Induction=args.mem_induction,
                               mask_flag=True,
                               use_negsample=args.use_negsample, args=args)
        else:
            print ("Invalid model_type : %s", model_type)
            return

        # 参数初始化
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        sys.stdout.flush()
        logging.info('training begin')
        sys.stdout.flush()

        start_time = time.time()
        iter = 0
        lr = 0.001
        best_auc = [0.0]
        loss_sum = 0.0
        accuracy_sum = 0.
        left_loss_sum = 0.
        aux_loss_sum = 0.
        mem_loss_sum = 0.
        epoch = args.epoch
        data_thread_num = args.data_thread_num
        logging.debug(data_thread_num)
        for itr in range(epoch):
            logging.info("epoch: " + str(itr))
            data_load = DataIterator(train_file, batch_size, maxlen=args.max_len, args=args)
            data_pool, _stop, _ = generator_queue(data_load)

            _start_total_time = time.time()
            sum_sess_time = 0
            sum_total_time = 0
            while True:
                if _stop.is_set() and data_pool.empty():
                    break
                if not data_pool.empty():
                    src, tgt = data_pool.get()
                else:
                    continue
                data = prepare_data(
                    src, tgt, args)

                _start_sess_time = time.time()

                data['lr'] = lr

                loss, acc, aux_loss, mem_loss, left_loss = model.train(sess, data)

                loss_sum += loss
                accuracy_sum += acc
                left_loss_sum += left_loss
                aux_loss_sum += aux_loss
                mem_loss_sum += mem_loss
                iter += 1
                sys.stdout.flush()
                sess_time = time.time() - _start_sess_time
                sum_sess_time += sess_time
                sum_total_time += time.time() - _start_total_time
                _start_total_time = time.time()

                if (iter % test_iter) == 0:
                    test_time = time.time()
                    logging.info(
                        '%d:iter=%d, train_loss=%.4f, train_accuracy=%.4f, train_aux_loss=%.4f, train_left_loss=%.4f, total_time=%.4f ms, sess_time=%.4f ms, train_time=%.4f s' % (
                        itr, iter, loss_sum / test_iter, accuracy_sum / test_iter, \
                        aux_loss_sum / test_iter, left_loss_sum / test_iter,
                        (1000 * sum_total_time) / (batch_size * test_iter),
                        sum_sess_time * 1000 / (batch_size * test_iter),
                        test_time - start_time))
                    loss_sum = 0.0
                    accuracy_sum = 0.0
                    left_loss_sum = 0.0
                    aux_loss_sum = 0.
                    mem_loss_sum = 0.
                    if (iter % save_iter) == 0:
                        logging.info('save model iter: %d' % (iter))
                        model.save(sess, model_path + "--" + str(iter))
                        eval_result = eval(sess, test_file, model, best_model_path, batch_size, maxlen, best_auc, args)
                        eval_result['itr'] = itr
                        eval_result['iter'] = iter
                        logging.info(
                            'Testing finishes-------{itr}:iter={iter},test_auc={auc:.4f}, test_loss={loss:.4f}, test_accuracy={accuracy:.4f}, test_aux_loss={aux_loss:.4f}, best_auc={best_auc:.4f}'.format(
                                **eval_result))
                    sum_sess_time = 0

            logging.debug("epoch {0} train end".format(itr))

            logging.debug("train epoch {0} end".format(itr))


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode")
    parser.add_argument("-seed", type=int, default=2)
    parser.add_argument("-use_time", type=bool, default=False)
    parser.add_argument("-use_first_att", type=bool, default=False)
    parser.add_argument("-first_att_top_k", type=int, default=999)
    parser.add_argument("-use_vec_loss", type=bool, default=False)
    parser.add_argument("-use_time_mode", type=str, default='concat')
    parser.add_argument("-long_seq_split", type=str, default="")
    parser.add_argument("-short_seq_split", type=str, default="900:1000")
    parser.add_argument("-short_model_type", type=str)
    parser.add_argument("-long_model_type", type=str)
    parser.add_argument("-save_iter", type=int, default=5000)
    parser.add_argument("-test_file_num", type=int, default=10)
    parser.add_argument("-test_iter", type=int, default=500)
    parser.add_argument("-max_len", type=int, default=100)
    parser.add_argument("-seq_len", type=int, default=1000)
    parser.add_argument("-min_train_file_id", type=int, default=0)
    parser.add_argument("-max_train_file_id", type=int, default=160)
    parser.add_argument("-data_thread_num", type=int, default=5)
    parser.add_argument("-epoch", type=int, default=1)
    parser.add_argument("-memory_size", type=int, default=1)
    parser.add_argument("-embedding_dim", type=int, default=4)
    parser.add_argument("-batch_size", type=int, default=256)
    parser.add_argument("-parral_stag", type=int, default=0)
    parser.add_argument("-mimn_seq_reduce", type=int, default=1)
    parser.add_argument("-head_num", type=int, default=1)
    parser.add_argument("-mem_induction", type=int, default=0)
    parser.add_argument("-ntm_flag", type=str, default='learned,0')
    parser.add_argument("-search_mode", type=str, default='')
    parser.add_argument("-level", type=str, default='INFO')
    parser.add_argument("-data_type", type=str, default='taobao')
    parser.add_argument("-use_negsample", type=bool, default=False)
    parser.add_argument("-util_reg", type=int, default=0)
    parser.add_argument("-time_embedding_dim", type=int, default=4)
    parser.add_argument("-att_func", type=str, default='all')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logging.basicConfig(format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                        level=args.level, stream=sys.stderr)
    logging.info(args)
    SEED = args.seed
    Memory_Size = args.memory_size
    Parral_Stag = args.parral_stag
    Ntm_Flag = args.ntm_flag

    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    train_file = './data/book_data/book_train.txt'
    test_file = './data/book_data/book_test.txt'
    logging.info(train_file)
    if args.mode == 'train':
        train(train_file=train_file, test_file=test_file, model_type=args.long_model_type, Memory_Size=Memory_Size,
              Parral_Stag=Parral_Stag, Ntm_Flag=Ntm_Flag,
              save_iter=args.save_iter, test_iter=args.test_iter, args=args, batch_size=args.batch_size)


if __name__ == '__main__':
    main()
