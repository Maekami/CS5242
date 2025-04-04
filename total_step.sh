export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# TASK=math_500
TASK=aime24
python eval_step1.py \
--model_name_or_path $MODEL \
--teacher_name "open-thoughts/OpenThoughts-114k" \
--teacher_cot_path "generated_cot" \
--data_name $TASK \
--max_tokens 32768 \
--temperature 0.5 \

NUM_GPUS=1
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:16384,temperature:0.3,top_p:0.95}"
# MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:8192,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL



# AIME 2024++
TASK=aime24
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks eval_step2.py \
    --output-dir $OUTPUT_DIR \
    --save-details \
    # --use-chat-template \

# MATH-500
# TASK=math_500
# lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
#     --custom-tasks src/open_r1/evaluate.py \
#     --output-dir $OUTPUT_DIR \
#     # --use-chat-template \

# # # GPQA Diamond
# # TASK=gpqa:diamond
# # lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
# #     --custom-tasks src/open_r1/evaluate.py \
# #     --use-chat-template \
# #     --output-dir $OUTPUT_DIR

# # # LiveCodeBench
# # lighteval vllm $MODEL_ARGS "extended|lcb:codegeneration|0|0" \
# #     --use-chat-template \
# #     --output-dir $OUTPUT_DIR 

# # GRGC
# TASK=grgc
# lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
#     --custom-tasks src/open_r1/evaluate.py \
#     --output-dir $OUTPUT_DIR \
#     --save-details \