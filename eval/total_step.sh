data_name="math"
teacher_cot_dir="generated_cot"
teacher_cot_path="${teacher_cot_dir}/${data_name}"

CUDA_VISIBLE_DEVICES='0,1,2,3' \
python eval_step1.py \
--model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
--teacher_name "open-thoughts/OpenThoughts-114k" \
--teacher_cot_path $teacher_cot_path \
--data_name $data_name \
--temperature 0.7 \
--random_teacher

CUDA_VISIBLE_DEVICES='0,1,2,3' \
python eval_step2.py \
--model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
--teacher_cot_path $teacher_cot_path \
--data_name $data_name \
--prompt_type "qwen-instruct" \
--temperature 0.7 \
--start_idx 0 \
--end_idx -1 \
--n_sampling 1 \
--k 1 \
--split "test" \
--max_tokens 8192 \
--seed 0 \
--top_p 1 \
--use_teacher