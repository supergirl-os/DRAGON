port=12345
root_path=/home/azureuser/cloudfiles/code/Users/yaxuan.wang/baseline
model_family=llama2-7b
evaluation_method=None
split=forget01_perturbed
model_path=${root_path}/tofu/data/ft_epoch5_lr1e-05_llama2-7b_full_wd0.01/checkpoint-625
method_name=DRAGON
save_file_name=evaluation_results/tofu/${split}_${model_family}_${method_name}_${evaluation_method}.csv
path_to_aggregated_retain_result=data/eval/tofu_sides/ft_epoch5_lr1e-05_llama2-7b_retain99_wd0.01/checkpoint-618/eval_results/ds_size300/eval_log_aggregated.json
path_to_aggregated_ckp_result=results/${method_name}_tofu_${split}/eval_log_aggregated.json

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=$port tofu_runner.py\
 model_family=$model_family split=$split method=${method_name} \
 model_path=$model_path retain_result=${path_to_aggregated_retain_result} ckpt_result=${path_to_aggregated_ckp_result} \
 method_name=${method_name} save_file=${save_file_name} evaluation_method=${evaluation_method} 

# Calculate the KFR and KRR
# cd tofu/unlearn/evals
# forget_path=${root_path}/rag_unlearn/results/${method_name}_tofu_${split}/eval_log_forget_generated_text.json
# retain_path=${root_path}/rag_unlearn/results/${method_name}_tofu_${split}/eval_log_generated_text.json
# output_path=results/${split}_${model_family}_${method_name}_${evaluation_method}.json
# python evaluate.py --language_model_path ${model_path} --forget_path ${forget_path} --retain_path ${retain_path} --output_path ${output_path}
# cd ../../..