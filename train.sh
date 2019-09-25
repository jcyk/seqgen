export https_proxy="http://10.223.133.20:52107"
export http_proxy="http://10.223.133.20:52107"
pip install torch torchvision --upgrade --user #0.4.1
dataset=$1
random_mask=$2
python train.py  --vocab_src ${dataset}/vocab_src \
    --vocab_tgt ${dataset}/vocab_tgt \
    --embed_dim 512 \
    --hidden_size 512 \
    --num_layers 2 \
    --dropout 0.33 \
    --epochs 50 \
    --lr 0.001 \
    --train_batch_size 256 \
    --print_every 1000 \
    --eval_every 10000 \
    --train_data ${dataset}/train \
    --dev_data  ${dataset}/test \
    --stop_words_file ${dataset}/stop_words \
    --random_mask ${random_mask} \
    --input_feed
