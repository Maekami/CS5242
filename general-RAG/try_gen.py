from datasets import load_dataset, Dataset
import random

# 设置随机种子
random.seed(42)

# 加载数据集
dataset = load_dataset("GeneralReasoning/GeneralThought-430K", split="train")

# 原始筛选条件
filtered = dataset.filter(
    lambda x: (
        x["question_source"] == "Meta/natural_reasoning" and
        x["task"] == "Engineering" and
        x["model_name"] == "DeepSeek/DeepSeek-R1" and
        x["reference_answer"] not in [None, ""]
    )
)

# 额外过滤掉含有省略号的 question 或 reference_answer
clean_filtered = filtered.filter(
    lambda x: "..." not in x["question"] and "..." not in x["reference_answer"]
)

# 随机选取500条为测试集
test_indices = random.sample(range(len(clean_filtered)), 500)
test_set = clean_filtered.select(test_indices)

# 剩余部分
remaining_indices = list(set(range(len(clean_filtered))) - set(test_indices))
remaining_set = clean_filtered.select(remaining_indices)

# 构造输出
test_output = Dataset.from_dict({
    "question": [x["question"] for x in test_set],
    "reference_answer": [x["reference_answer"] for x in test_set]
})

remaining_output = Dataset.from_dict({
    "question": [x["question"] for x in remaining_set],
    "model_reasoning": [x["model_reasoning"] for x in remaining_set]
})

# 可选择保存为文件
test_output.to_json("test_set.json", indent=2)
remaining_output.to_json("remaining_set.json", indent=2)
