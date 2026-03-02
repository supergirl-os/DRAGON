master_port=18765
model=llama3.1
# model=qwen-14b
# model=qwen2.5-14b
# model=llama2-13b
dataset=TOFU01
lr=2e-5
epoch=3
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=$master_port train.py --config-name=finetune.yaml num_epochs=${epoch} batch_size=2 gradient_accumulation_steps=4 model_family=${model} lr=${lr} dataset=${dataset}
