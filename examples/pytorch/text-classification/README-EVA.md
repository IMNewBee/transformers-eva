# 支持EVA和ACP_ADAMW

## 具体运行命令如下

注意把命令中的地址换为你自己的地址

### adamw_acp

机器一（假设为gpu4）

export NCCL_SOCKET_IFNAME=ib0; export NCCL_IB_DISABLE=0; torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=gpu4 --master_port=25640 /home/ltzhang/transformers-eva/examples/pytorch/text-classification/run_glue.py --model_name_or_path /home/ltzhang/models/roberta-base/ --task_name mrpc --optim adamw_acp --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 64 --learning_rate 5e-5 --num_train_epochs 20 --output_dir /home/ltzhang/transformers-eva/examples/pytorch/text-classification/output/roberta-base-mrpc/ --logging_steps=1 --evaluation_strategy epoch --overwrite_output_dir

机器二（假设为gpu5）

export NCCL_SOCKET_IFNAME=ib0; export NCCL_IB_DISABLE=0; torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=gpu4 --master_port=25640 /home/ltzhang/transformers-eva/examples/pytorch/text-classification/run_glue.py --model_name_or_path /home/ltzhang/models/roberta-base/ --task_name mrpc --optim adamw_acp --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 64 --learning_rate 5e-5 --num_train_epochs 20 --output_dir /home/ltzhang/transformers-eva/examples/pytorch/text-classification/output/roberta-base-mrpc/ --logging_steps=1 --evaluation_strategy epoch --overwrite_output_dir

### eva

机器一（假设为gpu4）

export NCCL_SOCKET_IFNAME=ib0; export NCCL_IB_DISABLE=0; torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=gpu4 --master_port=25640 /home/ltzhang/transformers-eva/examples/pytorch/text-classification/run_glue.py --model_name_or_path /home/ltzhang/models/roberta-base/ --task_name mrpc --optim eva --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 64 --learning_rate 0.1 --num_train_epochs 20 --output_dir /home/ltzhang/transformers-eva/examples/pytorch/text-classification/output/roberta-base-mrpc/ --logging_steps=1 --evaluation_strategy epoch --overwrite_output_dir

机器二（假设为gpu5）

export NCCL_SOCKET_IFNAME=ib0; export NCCL_IB_DISABLE=0; torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=gpu4 --master_port=25640 /home/ltzhang/transformers-eva/examples/pytorch/text-classification/run_glue.py --model_name_or_path /home/ltzhang/models/roberta-base/ --task_name mrpc --optim eva --do_train --do_eval --max_seq_length 128 --per_device_train_batch_size 64 --learning_rate 0.1 --num_train_epochs 20 --output_dir /home/ltzhang/transformers-eva/examples/pytorch/text-classification/output/roberta-base-mrpc/ --logging_steps=1 --evaluation_strategy epoch --overwrite_output_dir

## 上述运行命令的解释

1. export NCCL_SOCKET_IFNAME=ib0; export NCCL_IB_DISABLE=0; -> 使用ib网络
2. torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=gpu13 --master_port=25640 -> torchrun是用来运行torch ddp的命令，每个节点4张卡，总共2个节点，本命令节点编号为0（所有节点编号各不相同），主机IP为gpu4，主机端口为25640（随意指定即可）
3. /home/ltzhang/transformers-eva/examples/pytorch/text-classification/run_glue.py -> torchrun要运行python文件
4. --model_name_or_path -> 模型所在地
5. 后略

## 部分结果

### eva

| lr       | acc   |
|----------|-------|
| 1.00E-02 | 0.875 |
| 2.00E-02 | 0.882 |
| 3.00E-02 | 0.873 |
| 4.00E-02 | 0.878 |
| 5.00E-02 | 0.88  |
| 6.00E-02 | 0.833 |
| 7.00E-02 | 0.68  |
| 8.00E-02 | 0.87  |
| 9.00E-02 | 0.865 |
| 1.00E-01 | 0.858 |
| 2.00E-01 | 0.846 |
| 3.00E-01 | 0.873 |
| 4.00E-01 | 0.865 |
| 5.00E-01 | 0.8   |
