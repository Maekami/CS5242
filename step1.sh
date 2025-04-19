export DEBUGPY=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0,1


MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
# TASK=math_500
TASK=aime24
# python eval_step1.py \
#     --model_name_or_path $MODEL \
#     --teacher_name "open-thoughts/OpenThoughts-114k" \
#     --teacher_cot_path "generated_cot" \
#     --data_name $TASK \
#     --max_tokens 32768 \
#     --temperature 0.3 \
#     --intermediate_dir "./intermediate_outputs"

python eval_step1.py \
    --model_name_or_path $MODEL \
    --teacher_name "open-thoughts/OpenThoughts-114k" \
    --teacher_cot_path "generated_cot_test" \
    --data_name $TASK \
    --max_tokens 32768 \
    --temperature 0.7 \
    --intermediate_dir "./intermediate_outputs"