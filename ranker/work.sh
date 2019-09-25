export https_proxy="http://10.223.133.20:52107"
export http_proxy="http://10.223.133.20:52107"
python work.py --test_batch_size 512 \
                --test_data ../test_out \
                --load_path ckpt/epoch11_batch587999_acc_0.915 \
                --output_file ./test_out
