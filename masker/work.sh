export https_proxy="http://10.223.133.20:52107"
export http_proxy="http://10.223.133.20:52107"
#pip install torch torchvision --upgrade --user #0.4.1
python work.py --test_batch_size 128 \
                --test_data ../all_processed \
                --load_path ckpt/epoch1_batch9999_best_ff0.772 ckpt/epoch0_batch4999_best_ff0.767  ckpt/epoch0_batch4999_best_ff0.768  ckpt/epoch2_batch12999_best_ff0.762 ckpt/epoch1_batch7999_best_ff0.763\
                --min_vote 1  \
                --output_file ./test.multiple.1out5
