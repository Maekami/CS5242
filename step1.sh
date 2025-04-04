#export DEBUGPY='1'
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
data_name="aime"
teacher_cot_dir="generated_cot"
teacher_cot_path="${teacher_cot_dir}/${data_name}"

CUDA_VISIBLE_DEVICES='4,5,6,7' \
python eval_step1.py \
--model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
--teacher_name "open-thoughts/OpenThoughts-114k" \
--teacher_cot_path $teacher_cot_path \
--data_name $data_name \
--temperature 0.7 \
--num_of_responses 3 \
#--random_teacher # test random teacher cot