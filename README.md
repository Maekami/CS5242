# RAG Teacher CoT and Prompt Framework

## Environment Setup

When setting up the environment, pay attention to package version numbers, especially for those with specific version requirements noted in the documentation.

```bash
pip install -r requirements.txt
```

### Quick Use

Execute the script using:

```bash
bash total_step.sh
```

Parameters in `total_step.sh`:

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' \
python eval_step1.py \
--model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
--data_type "math" \
--temperature 0.7 \
--max_tokens 8192 \

python eval_step2.py \
--model_name_or_path "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
--data_name "test" \
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
```


## Acknowledgments

Our evaluation code is modified from [LIMO]([https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation](https://github.com/GAIR-NLP/LIMO)). We thank their team for their valuable contributions to the community.
