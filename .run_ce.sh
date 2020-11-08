#在LD_LIBRARY_PATH中添加cuda库的路径
export LD_LIBRARY_PATH=/home/work/cuda-9.0/lib64
#在LD_LIBRARY_PATH中添加cudnn库的路径
export LD_LIBRARY_PATH=/home/work/cudnn/cudnn_v7/cuda/lib64:$LD_LIBRARY_PATH
#需要先下载NCCL，然后在LD_LIBRARY_PATH中添加NCCL库的路径
export LD_LIBRARY_PATH=/home/work/nccl/nccl2.3.7_cuda9.0/lib:$LD_LIBRARY_PATH
#如果FLAGS_sync_nccl_allreduce为1，则会在allreduce_op_handle中调用cudaStreamSynchronize（nccl_stream），这种模式在某些情况下可以获得更好的性能
#export FLAGS_fraction_of_gpu_memory_to_use=0.95

export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fraction_of_gpu_memory_to_use=1.0
export FLAGS_fast_eager_deletion_mode=true 
FLAGS_fast_eager_deletion_mode=True
BERT_BASE_PATH="chinese_L-12_H-768_A-12"
TASK_NAME='fewshot'
DATA_PATH=data/
CKPT_PATH=pretrain_model

train(){
/mnt/du/yanghao16/paddle_few_shot/6-paddle_1.7.2_py3.7/miniconda3/bin/python3 -u run_classifier.py --task_name ${TASK_NAME} \
                   --use_cuda true \
                   --do_train true \
                   --do_val true \
                   --do_test true \
                   --batch_size 1 \
                   --init_checkpoint chinese_L-12_H-768_A-12/params \
                   --data_dir ${DATA_PATH} \
                   --vocab_path chinese_L-12_H-768_A-12/vocab.txt \
                   --checkpoints ${CKPT_PATH} \
                   --save_steps 1000 \
                   --weight_decay  0.01 \
                   --warmup_proportion 0.06 \
                   --validation_steps 100 \
                   --epoch 1 \
                   --max_seq_len 512 \
                   --bert_config_path chinese_L-12_H-768_A-12/bert_config.json \
                   --learning_rate 1e-4 \
                   --skip_steps 10 \
                   --random_seed 100 \
                   --enable_ce \
                   --shuffle false \
                   --train_iter 30000 \
                   --val_iter 1000 \
                   --test_iter 1000 \
                   --k 10 \
                   --n 5 \
                   --q 1 

}

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
train 
