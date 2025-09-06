#python app.py
#CUDA_VISIBLE_DEVICES=0 \
#python3 autoregressive/sample/sample_t2i.py \
#  --vq-ckpt /home/ldap-users/s2220411/Code/new_explore_multimodel/LlamaGen/vq_ds16_t2i.pt \
#  --gpt-ckpt /home/ldap-users/s2220411/Code/new_explore_multimodel/LlamaGen/t2i_XL_stage1_256.pt \
#  --gpt-model GPT-XL --image-size 256


#CUDA_VISIBLE_DEVICES=2 \
#python3 autoregressive/sample/sample_t2i.py \
#  --vq-ckpt /home/ldap-users/s2220411/Code/new_explore_multimodel/LlamaGen/pretrained_models/vq_ds16_t2i.pt \
#  --gpt-ckpt /home/ldap-users/s2220411/Code/new_explore_multimodel/LlamaGen/t2i_XL_stage2_512.pt \
#  --gpt-model GPT-XL --image-size 512


#config_name="configs_gpt"
#CUDA_VISIBLE_DEVICES=2 python autoregressive/train/train_t2s.py \
#  --configs_training configs_model/$config_name/configs_training.yaml \
#  --config_model configs_model/$config_name \

#set -x
#CUDA_VISIBLE_DEVICES=2 \
#torchrun \
#--nnodes=1 --nproc_per_node=1 --node_rank=0 \
#--master_addr=127.0.0.1 --master_port=9999 \
#autoregressive/train/train_t2i.py \
#--vq-ckpt ./pretrained_models/vq_ds16_t2i.pt \
#--data-path /path/to/laion_coco50M \
#--t5-feat-path /path/to/laion_coco50M_flan_t5_xl \
#--dataset t2i \
#--image-size 256 \
#--cloud-save-path "test" \
#--gpt-model GPT-XXL_speech

#config_name=configs_gpt
#CUDA_VISIBLE_DEVICES=0,2 torchrun --master_port=29500 --nproc_per_node=2 --nnodes=1 --node_rank=0 autoregressive/train/train_t2s.py \
#  --configs_training configs_model/$config_name/configs_training.yaml \
#  --config_model configs_model/$config_name \

#config_name=configs_gpt
#CUDA_VISIBLE_DEVICES=2 torchrun --master_port=29502 --nproc_per_node=1 --nnodes=1 --node_rank=0 autoregressive/train/train_t2s.py \
#  --configs_training configs_model/$config_name/configs_training.yaml \
#  --config_model configs_model/$config_name \

config_name=configs_gpt
CUDA_VISIBLE_DEVICES=1 torchrun --master_port=29001 --nproc_per_node=1 --nnodes=1 --node_rank=0 autoregressive/train/train_t2is.py \
  --configs_training configs_model/$config_name/configs_training_cosyvoice.yaml \
  --config_model configs_model/$config_name \

#config_name=configs_gpt
#CUDA_VISIBLE_DEVICES=0 torchrun --master_port=29000 --nproc_per_node=1 --nnodes=1 --node_rank=0 autoregressive/train/train_t2is_t5.py \
#  --configs_training configs_model/$config_name/configs_training_cosyvoice.yaml \
#  --config_model configs_model/$config_name \


#config_name=configs_gpt
#CUDA_VISIBLE_DEVICES=0 torchrun --master_port=29000 --nproc_per_node=1 --nnodes=1 --node_rank=0 autoregressive/train/train_t2is_simple.py \
#  --configs_training configs_model/$config_name/configs_training_cosyvoice.yaml \
#  --config_model configs_model/$config_name \
