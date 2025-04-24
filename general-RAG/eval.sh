export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=4,5,6,7

MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
teacher_cot_path="generated_cot"
teacher_cot_filename="generated_reasonings.jsonl"


python eval_step1.py \
--model_name_or_path $MODEL \
--teacher_name "GeneralReasoning/GeneralThought-430K" \
--teacher_cot_path $teacher_cot_path \
--teacher_cot_filename $teacher_cot_filename \
--max_tokens 32768 \
--temperature 0.3 \
--intermediate_dir "./intermediate_outputs" \
--analyze_type 'insight'

