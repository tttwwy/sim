# Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction
Implementation of Search-based User Interest Modeling with Lifelong Sequential Behavior Data for Click-Through Rate Prediction using tensorflow

## Prerequisites
- Python 2.x
- Tensorflow 1.15.0

## Data
- [Amazon Book Data](http://jmcauley.ucsd.edu/data/amazon/)<br/>

## Getting Started
First we need to prepare data.<br/>

### Amazon Prepare
- Because getting and processing the data is time consuming，we had processed Amazon data and upload it for you.<br/>
```
tar -xzf data.tar.gz
```


## Running

```
usage: train_taobao_and_book.py [-h] [-mode MODE] [-seed SEED]
                                [-use_first_att USE_FIRST_ATT]
                                [-first_att_top_k FIRST_ATT_TOP_K]
                                [-use_vec_loss USE_VEC_LOSS]
                                [-long_seq_split LONG_SEQ_SPLIT]
                                [-short_seq_split SHORT_SEQ_SPLIT]
                                [-short_model_type SHORT_MODEL_TYPE]
                                [-long_model_type LONG_MODEL_TYPE]
                                [-save_iter SAVE_ITER]
                                [-test_iter TEST_ITER] [-max_len MAX_LEN]
                                [-seq_len SEQ_LEN]
                                [-epoch EPOCH] [-memory_size MEMORY_SIZE]
                                [-batch_size BATCH_SIZE]
                                [-search_mode SEARCH_MODE] [-level LEVEL]
                                [-data_type DATA_TYPE]
                                [-att_func ATT_FUNC]
```


### Base Model
The example for DNN
```
python train.py -mode train \
-data_type book \
 -max_len 100 \
 -short_model_type DIN \
 -short_seq_split '90:100' \
 -long_model_type DNN \
 -long_seq_split '0:90' \
 -seed 2  \
  -epoch 2 \
 -save_iter 10 \
 -test_iter 20 \
 -search_mode 'None' 
```
The model type below had been supported: 
- DNN 
- DIN
- MIMN

### SIM
You can train SIM with two kinds of search unit:


- hard-search

```
python train.py -mode train \
-data_type book \
 -max_len 100 \
 -short_model_type DIN \
 -short_seq_split '90:100' \
 -long_model_type DIN \
 -long_seq_split '0:90' \
 -seed 2  \
  -epoch 2 \
 -save_iter 10 \
 -test_iter 20 \
 -search_mode 'cate' \
 -att_func 'dot' 
```



- soft-search

```
python train.py -mode train \
-data_type book \
 -max_len 100 \
 -short_model_type DIN \
 -short_seq_split '90:100' \
 -long_model_type DIN \
 -long_seq_split '0:90' \
 -seed 2  \
  -epoch 2 \
 -data_thread_num 5 \
 -save_iter 10 \
 -test_iter 20 \
 -search_mode 'None' \
  -use_first_att True \
 -first_att_top_k 50 \
 -use_vec_loss True \
  -att_func 'dot' 
```

