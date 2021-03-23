# SIM hard-search
python train.py -mode train \
 -data_type book \
 -max_len 100 \
 -short_model_type DIN \
 -short_seq_split '90:100' \
 -long_model_type DIN \
 -long_seq_split '0:90' \
 -seed 2 \
 -epoch 2 \
 -save_iter 10 \
 -test_iter 20 \
 -search_mode 'cate' \
 -att_func 'dot' 

# SIM soft-search
python train.py -mode train \
 -data_type book \
 -max_len 100 \
 -short_model_type DIN \
 -short_seq_split '90:100' \
 -long_model_type DIN \
 -long_seq_split '0:90' \
 -seed 2 \
 -epoch 2 \
 -data_thread_num 5 \
 -save_iter 10 \
 -test_iter 20 \
 -search_mode 'None' \
 -use_first_att True \
 -first_att_top_k 50 \
 -use_vec_loss True \
 -att_func 'dot' 




# The example for DNN
python train.py -mode train \
 -data_type book \
 -max_len 100 \
 -short_model_type DIN \
 -short_seq_split '90:100' \
 -long_model_type DNN \
 -long_seq_split '0:90' \
 -seed 2 \
 -epoch 2 \
 -save_iter 10 \
 -test_iter 20 \
 -search_mode 'None' 
