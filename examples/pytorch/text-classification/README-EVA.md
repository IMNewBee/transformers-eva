# 具体运行命令如下

注意把命令中的地址换为你自己的地址

机器一（假设为gpu4）

export NCCL_SOCKET_IFNAME=ib0; export NCCL_IB_DISABLE=0; torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=gpu4 --master_port=25641 /home/ltzhang/transformers-eva/examples/pytorch/text-classification/run_glue.py --model_name_or_path /home/ltzhang/models/roberta-base/ --task_name mrpc --optim eva --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 64 --learning_rate 1e-5 --num_train_epochs 10 --output_dir /home/ltzhang/transformers-eva/examples/pytorch/text-classification/output/roberta-base-mrpc/ --logging_steps=1 --evaluation_strategy epoch --overwrite_output_dir

机器二（假设为gpu5）

export NCCL_SOCKET_IFNAME=ib0; export NCCL_IB_DISABLE=0; torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=gpu5 --master_port=25641 /home/ltzhang/transformers-eva/examples/pytorch/text-classification/run_glue.py --model_name_or_path /home/ltzhang/models/roberta-base/ --task_name mrpc --optim eva --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 64 --learning_rate 1e-5 --num_train_epochs 10 --output_dir /home/ltzhang/transformers-eva/examples/pytorch/text-classification/output/roberta-base-mrpc/ --logging_steps=1 --evaluation_strategy epoch --overwrite_output_dir