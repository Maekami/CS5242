import pandas as pd

# 读取 parquet 文件
path = "data/evals/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/details/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B/2025-04-19T05-48-46.201202/details_custom|aime24|0_2025-04-19T05-48-46.201202.parquet"
df = pd.read_parquet(path)



# 或保存为纯文本，每行一条记录
df.to_csv("output.txt", sep="\t", index=False)