import sys
# import os
from os import path
# import io
# from io import StringIO
# import collections as coll
import random
import numpy as np
import json
import time
import threading
from collections import deque
import logging

class DataLoader:

    def __init__(
        self,
        data_path,
        data_file,
        batch_size,
        sleep_time=1,
        max_queue_size = 2,
        start_file_id=0,
        end_file_id=160
    ):
        # load data 
        self.queue = deque() #multiprocessing.Queue(maxsize=max_queue_size) # it may change in future if we decide to split data into many small chunks instead of 4
        self.batch_size = batch_size
        self.data_path = data_path
        self.data_file = data_file
        self.end_file_id = end_file_id
        self.sleep_time = sleep_time
        self.max_queue_size = max_queue_size 
        self.help_count = 0
        self.start_file_id = start_file_id
 
    def __iter__(self):
        return self

    def data_read(self, start_id, total_thread):
        sample_id = start_id
        logging.debug('help_count={0}'.format(self.help_count))
        while (sample_id + self.start_file_id) <= self.end_file_id:
            logging.debug('{0} sample id:{1}'.format(self.data_file,sample_id+self.start_file_id))
            logging.debug("{0} len queue:{1}".format(self.data_file,len(self.queue)))
            if len(self.queue) >= self.max_queue_size:
                time.sleep(1)
                continue
            processed_data_path = self.data_path + self.data_file + "_" + str(sample_id+self.start_file_id) + '_processed.npz'
            logging.info('Start loading processed data...' + processed_data_path)
            st = time.time()
            # try:
            data = np.load(processed_data_path)
            # except IOError:
            #     logging.info("read data end!")
            #     continue
            source = data['source_array']
            uid_array = np.array(source)[:,0]
            item_array = np.array(source)[:,1]
            cate_array = np.array(source)[:,2]
            shop_array = np.array(source)[:,3]
            node_array = np.array(source)[:,4]
            product_array = np.array(source)[:,5]
            brand_array = np.array(source)[:,6]

            target = data['target_array']
            history_item = data['history_item_array']
            history_cate = data['history_cate_array']
            history_shop = data['history_shop_array']
            history_node = data['history_node_array']
            history_product = data['history_product_array']
            history_brand = data['history_brand_array']

            neg_history_item = data['neg_history_item_array']
            neg_history_cate = data['neg_history_cate_array']
            neg_history_shop = data['neg_history_shop_array']
            neg_history_node = data['neg_history_node_array']
            neg_history_product = data['neg_history_product_array']
            neg_history_brand = data['neg_history_brand_array']
            logging.debug('Finish loading processed data id '+ str(sample_id+self.start_file_id) + ',Time cost = %.4f' % (time.time()-st))
            data_file  = (uid_array,item_array,cate_array,shop_array,node_array,product_array,brand_array,\
                target, history_item,history_cate,history_shop, history_node,history_product,history_brand,\
                neg_history_item,neg_history_cate,neg_history_shop, neg_history_node,neg_history_product,neg_history_brand)
            while self.help_count % total_thread != start_id:
                logging.debug('waitting help count:{0}'.format(self.help_count))
                time.sleep(1)
            logging.debug('help_count={0}'.format(self.help_count))
            self.queue.append(data_file)
            self.help_count += 1
            sample_id = sample_id + total_thread
        logging.debug('finish data read!')


    def _batch_data(self, data, data_slice,args):
        uid_array,item_array,cate_array,shop_array,node_array,product_array,brand_array,\
            target, history_item,history_cate,history_shop, history_node,history_product,history_brand,\
                neg_history_item,neg_history_cate,neg_history_shop, neg_history_node,neg_history_product,neg_history_brand = data
        #print("in _batch_data func")
        user_id = uid_array[data_slice]
        item_id = item_array[data_slice]
        cate_id = cate_array[data_slice]
        shop_id = shop_array[data_slice]
        node_id = node_array[data_slice]
        product_id = product_array[data_slice]
        brand_id = brand_array[data_slice]
        label = target[data_slice, :]
        time_id = np.asarray(np.ones_like(item_id)*1024,dtype=np.int32)
        # logging.info(hist_item.shape)
        # logging.info(cate_id.shape)
        # logging.info(hist_item[0])

        if args.seq_len > 0:
            hist_item = history_item[data_slice, :args.seq_len]
            hist_cate = history_cate[data_slice, :args.seq_len]
            hist_shop = history_shop[data_slice, :args.seq_len]
            hist_node = history_node[data_slice, :args.seq_len]
            hist_product = history_product[data_slice, :args.seq_len]
            hist_brand = history_brand[data_slice, :args.seq_len]

            neg_hist_item = neg_history_item[data_slice, :args.seq_len]
            neg_hist_cate = neg_history_cate[data_slice, :args.seq_len]
            neg_hist_shop = neg_history_shop[data_slice, :args.seq_len]
            neg_hist_node = neg_history_node[data_slice, :args.seq_len]
            neg_hist_product = neg_history_product[data_slice, :args.seq_len]
            neg_hist_brand = neg_history_brand[data_slice, :args.seq_len]
        else:
            hist_item = history_item[data_slice, args.seq_len:]
            hist_cate = history_cate[data_slice, args.seq_len:]
            hist_shop = history_shop[data_slice, args.seq_len:]
            hist_node = history_node[data_slice, args.seq_len:]
            hist_product = history_product[data_slice, args.seq_len:]
            hist_brand = history_brand[data_slice, args.seq_len:]

            neg_hist_item = neg_history_item[data_slice, args.seq_len:]
            neg_hist_cate = neg_history_cate[data_slice, args.seq_len:]
            neg_hist_shop = neg_history_shop[data_slice, args.seq_len:]
            neg_hist_node = neg_history_node[data_slice, args.seq_len:]
            neg_hist_product = neg_history_product[data_slice, args.seq_len:]
            neg_hist_brand = neg_history_brand[data_slice, args.seq_len:]

        # logging.info(item_id.shape)
        time_his_id = np.asarray([range(hist_item.shape[1]) for i in range(hist_item.shape[0])],dtype=np.int32)

        hist_mask = np.greater( hist_item, 0) * 1.0
        if args.long_seq_split and args.search_mode == 'cate':
            seq_split = [(int(x.split(":")[0]),int(x.split(":")[1])) for x in args.long_seq_split.split(",")]
            for idx,(left_idx,right_idx) in enumerate(seq_split):
                hist_mask[:,left_idx:right_idx] = ((hist_cate == cate_id[:, None]) & (hist_item > 0))[:,left_idx:right_idx] * 1.0
            # if cate_id[0] in hist_cate[0]:
            #     logging.info(hist_mask[0])


        elif args.long_seq_split and args.search_mode == 'all':
            seq_split = [(int(x.split(":")[0]),int(x.split(":")[1])) for x in args.long_seq_split.split(",")]
            hist_mask = (hist_cate == cate_id[:,None]) & (hist_item > 0)
            # hist_mask = hist_mask | (hist_item > 0)
            hist_mask = hist_mask | (hist_shop == shop_id[:,None])
            hist_mask = hist_mask | (hist_node == node_id[:,None])
            hist_mask = hist_mask | (hist_product == product_id[:,None])
            hist_mask = hist_mask | (hist_brand == brand_id[:,None])
            hist_mask = hist_mask | (hist_item == item_id[:,None])
            hist_mask = hist_mask * 1.0

            for idx,(left_idx,right_idx) in enumerate(seq_split):
                hist_mask[:,:left_idx] =  (hist_item > 0)[:,:left_idx]*1.0
                hist_mask[:,right_idx:] =  (hist_item > 0)[:,right_idx:]*1.0





        # cross_item_and_hist_item = hist_item * item_id[:,None] % args.max_item_item_cross_num
        # cross_cate_and_hist_cate = hist_cate * cate_id[:,None] % args.max_cate_cate_cross_num
        # cross_item_and_hist_cate = hist_cate * item_id[:,None] % args.max_item_cate_cross_num

        # neg_hist_item = neg_history_item[data_slice, :]
        # neg_hist_cate = neg_history_cate[data_slice, :]
        # neg_hist_shop = neg_history_shop[data_slice, :]
        # neg_hist_node = neg_history_node[data_slice, :]
        # neg_hist_product = neg_history_product[data_slice, :]
        # neg_hist_brand = neg_history_brand[data_slice, :]


        result = {
            # 'cross_item_and_item_id_his_batch_ph':cross_item_and_hist_item,
            # 'cross_cate_and_cate_id_his_batch_ph': cross_cate_and_hist_cate,
            # 'cross_item_and_cate_id_his_batch_ph': cross_item_and_hist_cate,
            'uid_batch_ph':user_id,
           'item_id_batch_ph':item_id,
            'time_id_batch_ph':time_id,
           'cate_id_batch_ph':cate_id,
           'shop_id_batch_ph':shop_id,
           'node_id_batch_ph':node_id,
           'product_id_batch_ph':product_id,
           'brand_id_batch_ph':brand_id,
           'item_id_his_batch_ph':hist_item,
           'cate_his_batch_ph':hist_cate,
            'shop_his_batch_ph':hist_shop,
            'node_his_batch_ph': hist_node,
            'product_his_batch_ph':hist_product,
            'brand_his_batch_ph':hist_brand,
            'item_id_neg_batch_ph':neg_hist_item,
            'cate_neg_batch_ph':neg_hist_cate,
            'shop_neg_batch_ph':neg_hist_shop,
            'node_neg_batch_ph':neg_hist_node,
            'product_neg_batch_ph':neg_hist_product,
            'brand_neg_batch_ph':neg_hist_brand,
            'mask': hist_mask,
            'time_id_his_batch_ph':time_his_id,
            'target_ph':label
        }

        if args.cross_feature:
            for item in args.cross_feature.strip().split(","):
                cross_name, max_id_num = item.split(":")
                target_name, hist_name = cross_name.split("_")
                if target_name == 'item':
                    target = item_id
                if target_name == 'cate':
                    target = cate_id
                if hist_name == 'item':
                    hist = hist_item
                if hist_name == 'cate':
                    hist = hist_cate
                result[cross_name] = hist * target[:,None] % int(max_id_num)


        # for key in result:
        #     result[key][result[key] < 0] = 0
        return result

        # return [user_id, item_id, cate_id,shop_id, node_id, product_id, brand_id,
        #     label, hist_item, hist_cate, hist_shop, hist_node, hist_product, hist_brand,
        #     hist_mask, neg_hist_item, neg_hist_cate, neg_hist_shop, neg_hist_node,
        #     neg_hist_product, neg_hist_brand ]

    def next(self,args):
        previous_data_out = []
        data_file_read = 0
        batch_id = 0
        #print('in next func')
        #import pdb; pdb.set_trace()
        previous_line = 0
        while len(self.queue) < 2:
            logging.debug('waitting queue')
            time.sleep(1)
        logging.debug('Now the queue has {0} data file loaded in!'.format(len(self.queue)))
        total_file_num = self.end_file_id - self.start_file_id + 1
        logging.debug("total_file_num:{0}".format(total_file_num))
        while data_file_read < total_file_num :
            logging.debug("{0} data file read:{1} queue len {2}".format(self.data_file,data_file_read,len(self.queue)))
            if len(self.queue) == 0:
                logging.debug('len queue:{0}'.format(len(self.queue)))
                time.sleep(1)
                continue
            data = self.queue.popleft()
            file_line_num = data[0].shape[0]
            start_ind = 0
            data_file_read = data_file_read + 1
            stime = time.time()
            #print('start one file,time=', stime)
            while start_ind <= file_line_num - self.batch_size:

                if previous_line != 0:
                    batch_left = self.batch_size - previous_line
                else:
                    batch_left = self.batch_size
                data_slice = slice(start_ind, start_ind + batch_left)
                # slice the data from the list
                data_out = self._batch_data(data, data_slice,args) #data_out is tuple
                
                if previous_line != 0:
                    #attach the data
                    # for i in range(len(data_out)):
                    #     data_out[i] = np.concatenate(
                    #         [previous_data_out[i], data_out[i]],
                    #         axis=0
                    #     )

                    for key in data_out:
                        data_out[key] = np.concatenate(
                            [previous_data_out[key], data_out[key]],
                            axis=0
                        )
                if self.batch_size != len(data_out['uid_batch_ph']):
                    raise ValueError('batch fetched wrong!')
                
                start_ind = start_ind + batch_left
             
                previous_line = 0
                #print("start_ind ", start_ind)
                yield data_out
            if start_ind != file_line_num:
                data_slice = slice(start_ind, file_line_num)
                previous_data_out = self._batch_data(data, data_slice,args)
                previous_line = file_line_num - start_ind
                logging.debug("Left batch of size %d" %( previous_line))
            etime = time.time()
            logging.debug('Consume one file takes time= %.4f' %(etime-stime))
        logging.debug('drop last batch since it is not full batch size')

   
def test():
    data_load = DataLoader('/disk3/w.wei/dien-new/process_data_maxlen100_0225/', 'train_sample', 256, 15)
    producer1 = threading.Thread(target=data_load.data_read, args=(0, 3))  
    producer2 = threading.Thread(target=data_load.data_read, args=(1, 3))  
    producer3 = threading.Thread(target=data_load.data_read, args=(2, 3))  
    producer1.start()
    producer2.start()
    producer3.start()
    #data_i = iter(data_load)
    #data_o = next(data_i)
    #print('print=====',len(data_o))
    num = 0
    for data in data_load.next():
        num = num+1
        cnt = 1
        for i in range(10000):
            cnt = cnt * 1.0
        if num%1000 == 0:
            print('i=',num,',cnt=',cnt)

    producer1.join()
    producer2.join()
    producer3.join()

if __name__ == '__main__':
    test()
