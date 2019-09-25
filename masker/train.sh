export https_proxy="http://10.223.133.20:52107"
export http_proxy="http://10.223.133.20:52107"
pip install torch torchvision --upgrade --user #0.4.1
dataset=$1
python train.py --vocab_src ${dataset}/vocab_src \
                --vocab_tgt ${dataset}/vocab_src \
                --stop_words_file ${dataset}/stop_words \
                --embed_dim 512 \
                --ff_embed_dim 1024 \
                --num_heads 8 \
                --num_layers 2 \
                --dropout 0.3 \
                --epochs 5 \
                --lr 0.0001 \
                --train_batch_size 128 \
                --print_every 100 \
                --eval_every 1000 \
                --train_data ${dataset}/mask_train \
                --dev_data ${dataset}/mask_dev
