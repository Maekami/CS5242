from datasets import load_dataset
import re
import os
import json
import pickle
import torch
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
import random


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def query_llm_api(prompts,llm,sampling_params):
    completions = llm.generate(prompts, sampling_params)
    return [completion.outputs[0].text.strip() for completion in completions]
def rewriting_prompt(query):
    return f"""Rewrite the given query to optimize it for both keyword-based and semantic-similarity search methods. Follow these guidelines:

- Identify the core concepts and intent of the original query.
- Expand the query by including relevant synonyms, related terms, and alternate phrasings.
- Maintain the original meaning and intent of the query.
- Include specific keywords that are likely to appear in relevant documents.
- Incorporate natural language phrasing to capture semantic meaning.
- Include domain-specific terminology if applicable to the query's context.
- Ensure that the rewritten query covers both broad and specific aspects of the topic.
- Remove ambiguous or unnecessary words that might confuse the search.
- Combine all elements into a single, coherent paragraph that flows naturally.
- Aim for a balance between keyword richness and semantic clarity.

Provide the rewritten query as a single paragraph that incorporates various search aspects, such as keyword-focused, semantically focused, or domain-specific aspects. 
Only output the rewritten query and do not output your thoughts.

query: {query}
        """
def reranking_batch(prompts, llm, sampling_params):
    return query_llm_api(prompts, llm, sampling_params)
def parse_question(example, data_name=None):
    question = ""
    for key in ["question", "problem", "Question", "input"]:
        if key in example:
            question = example[key]
            break
    
    return question.strip()
# 文本提取函数
def extract_qa_pairs(example):
    """
    提取问答对并处理特殊标记
    返回格式：[(query1, answer1), (query2, answer2), ...]
    """
    pairs = []
    current_query = None

    for conv in example["conversations"]:
        if conv["from"] == "user":
            current_query = conv["value"]
        elif conv["from"] == "assistant" and current_query:
            # 清理特殊标记
            clean_answer = re.sub(r'<\|.*?\|>', '', conv["value"])
            pairs.append((current_query, clean_answer))
            current_query = None
    return pairs


def save_index(index, queries,thoughts, save_dir="./saved_index"):
    """保存索引和关联文本数据"""
    os.makedirs(save_dir, exist_ok=True)

    # 保存FAISS索引
    index_path = os.path.join(save_dir, "faiss_index.index")
    faiss.write_index(index, index_path)

    # 保存文本数据
    queries_path = os.path.join(save_dir, "queries.pkl")
    with open(queries_path, "wb") as f:
        pickle.dump(queries, f)

    thoughts_path = os.path.join(save_dir, "thoughts.pkl")
    with open(thoughts_path, "wb") as f:
        pickle.dump(thoughts, f)


    print(f"索引已保存至：{save_dir}")


def load_index(save_dir="./saved_index"):
    """加载保存的索引和文本"""
    # 加载索引
    index_path = os.path.join(save_dir, "faiss_index.index")
    index = faiss.read_index(index_path)

    # 加载文本数据
    queries_path = os.path.join(save_dir, "queries.pkl")
    with open(queries_path, "rb") as f:
        queries = pickle.load(f)

    thoughts_path = os.path.join(save_dir, "thoughts.pkl")
    with open(thoughts_path, "rb") as f:
        thoughts = pickle.load(f)

    print(f"已从 {save_dir} 加载索引（包含 {len(queries)} 条数据）")
    return index, queries,thoughts


class Retriever:
    def __init__(self,llm,sampling_params, save_dir="./database"):
        self.save_dir = save_dir
        self.index = None
        self.queries = []
        self.thoughts=[]
        self.llm=llm
        self.sampling_params = sampling_params

    

    def build_and_save(self, queries,thoughts, encoder_model='sentence-transformers/all-mpnet-base-v2'):
        """构建并保存索引"""
        # GPU加速编码
        encoder = SentenceTransformer(encoder_model).to(device)
        embeddings = encoder.encode(queries, show_progress_bar=True)

        # 构建FAISS索引
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings).astype('float32'))

        # 保存资源
        self.index = index
        self.queries = queries
        self.thoughts = thoughts
        save_index(index, queries,thoughts, self.save_dir)

    def load(self):
        """加载已有索引"""
        self.index, self.queries,self.thoughts = load_index(self.save_dir)
        return self

    def reranking(self, query, document):
        top_k = len(document) // 3
        doc_prompt = ""
        for i, doc in enumerate(document):
            doc_prompt += f"Document {i}: \n{doc}\n"
        prompt = f"""
Each document in the following list has a number next to it along with its content. A query is also provided.
Respond with the numbers of the documents that are similar to the query, in order of relevance. You should always give and only give me {top_k} numbers.
Your response should EXACTLY follow the following format:
[number_1, number_2, ..., number_k]

Query: {query}
{doc_prompt}
"""
        response = query_llm_api(prompt)[0]
        numbers_str_list = re.findall(r'\d+', response)
        if len(numbers_str_list) == 0:
            return list(range(top_k))
        numbers_list = [int(num) for num in numbers_str_list]
        if max(numbers_list) >= len(document) or min(numbers_list)<0:
            return list(range(top_k))

        # 根据提取结果决定返回的列表
        if len(numbers_list) >= top_k:
            return numbers_list[-top_k:]
        else:
            return list(range(top_k))

    

    def search(self, query, k=1, random_t= False, enable_reranking=True,enable_rewriting=True):
        """执行搜索"""
        if random_t:
            indices = np.random.randint(0, len(self.queries),size=k)
            indices = indices.tolist()
            return indices,[self.queries[i] for i in indices],[self.thoughts[i] for i in indices]

        if enable_rewriting:
            query += rewriting_prompt(query)
        query_embed = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').encode([query])
        if enable_reranking:
            distances, indices = self.index.search(query_embed, k*3)
            indices=indices[0]
            retrieved_queries=[self.queries[i] for i in indices]
            retrieved_thoughts=[self.thoughts[i] for i in indices]
            number_list=self.reranking(query, retrieved_queries)
            return [indices[i] for i in number_list],[retrieved_queries[i] for i in number_list], [retrieved_thoughts[i] for i in number_list]

        else:
            distances, indices = self.index.search(query_embed, k)
            indices=indices[0]
            return indices,[self.queries[i] for i in indices],[self.thoughts[i] for i in indices]



class ThoughtfulGenerator:
    def __init__(self, model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",temperature=0.7,max_tokens=8192,top_p=1.0):


        self.llm=LLM(
        model=model_name_or_path,
        tensor_parallel_size=4,
        trust_remote_code=True,
        gpu_memory_utilization=0.8
    )
        self.sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )

    def query_llm_api(self,prompts):
        completions = self.llm.generate(prompts, self.sampling_params)
        return [completion.outputs[0].text.strip() for completion in completions]
    def generate(self, query, retrieved_queries,retrieved_thoughts, max_length=1024, num_candidates=3):
        """生成带思考过程的回答"""
        prompt = self._format_prompt(query, retrieved_queries,retrieved_thoughts)
        response = self.query_llm_api(prompt)[0]

        return response

    def _format_prompt(self, query, retrieved_queries,retrieved_thoughts):
        """构造符合数据格式的提示"""
        context_str = "\n".join([f"Example {i + 1}: {ctx}" for i, ctx in enumerate(retrieved_queries)])
        context_str = "\n".join([f"Example {i + 1}:\nQuery:\n{ctx}\nThoughts:\n{thought}" for i, (ctx, thought) in
                                 enumerate(zip(retrieved_queries, retrieved_thoughts))])

        return f"""Generate thoughts based on the given query. There are {len(retrieved_queries)} examples.
Examples:
{context_str}

Now it's your turn to generate thoughts.

Query：{query}
"""

def rag_pipeline(query,retriever,generator, top_k=3):
    # 检索相关上下文
    indices,retrieved_queries,retrieved_thoughts = retriever.search(query, k=top_k)
    # 生成回答
    return generator.generate(query, retrieved_queries, retrieved_thoughts)

def load_queries_thoughts(teacher_name="GeneralReasoning/GeneralThought-430K", data_type="engineering",split_num=500, question_source="Meta/natural_reasoning", task="Engineering", testdata_save_path=""):
    if teacher_name == "GeneralReasoning/GeneralThought-430K":
        # 加载数据
        ds = load_dataset(teacher_name, split="train")

        # 筛选条件 + 去掉包含省略号的数据
        filtered = ds.filter(
            lambda x: (
                x.get("question_source") == question_source and
                x.get("task") == task and
                x.get("model_name") == "DeepSeek/DeepSeek-R1" and
                x.get("reference_answer") not in [None, ""] and
                "..." not in x.get("question", "") and
                "..." not in x.get("reference_answer", "")
            )
        )

        # 随机划分 500 条测试，其余训练
        random.seed(42)
        test_indices = random.sample(range(len(filtered)), split_num)
        train_indices = list(set(range(len(filtered))) - set(test_indices))

        # test set: question + reference_answer
        test_examples = filtered.select(test_indices)
        os.makedirs(testdata_save_path, exist_ok=True)
        with open(testdata_save_path+"/questions_with_answers.jsonl", "w", encoding="utf-8") as f:
            for ex in test_examples:
                json.dump({
                    "question": ex["question"],
                    "reference_answer": ex["reference_answer"]
                }, f, ensure_ascii=False)
                f.write("\n")

        # train: 用于构建索引
        train_examples = filtered.select(train_indices)
        queries = [x["question"] for x in train_examples]
        thoughts = [x["model_reasoning"] for x in train_examples]
        return queries, thoughts
    else:
        # 原始逻辑
        ds = load_dataset(teacher_name, "metadata", split="train")
        reference_data = [item for item in ds if item.get("domain") == data_type]
        queries = [item.get("problem", "") for item in reference_data]
        thoughts = [item.get("deepseek_reasoning", "") for item in reference_data]
        return queries, thoughts

def data_prepare(task="General Chemistry",question_source="Meta/natural_reasoning", teacher_name="GeneralReasoning/GeneralThought-430K", split_num=500, teacher_cot_path="./generated_cot", file_name="test_questions_with_matched_cots.jsonl",random_t=False, database_save_path="./database", save_test_dir="./test_data",enable_reranking=True, enable_rewriting=True):
    if task=="General Chemistry":
        data_type="chemistry"
    database_save_path = os.path.join(database_save_path, data_type)
    testdata_save_path = os.path.join(save_test_dir,data_type)
    shared_llm = LLM(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        tensor_parallel_size=4,
        trust_remote_code=True,
        gpu_memory_utilization=0.8
    )
    shared_sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=32768,
        top_p=1.0
    )
    retriever = Retriever(llm=shared_llm, sampling_params=shared_sampling_params, save_dir=database_save_path)

    if not os.path.exists(retriever.save_dir):
        queries, thoughts = load_queries_thoughts(teacher_name=teacher_name, data_type=data_type, split_num=split_num, question_source=question_source, task=task, testdata_save_path=testdata_save_path)
        retriever.build_and_save(queries, thoughts)
    else:
        retriever = retriever.load()

    test_json_path = os.path.join(testdata_save_path, "questions_with_answers.jsonl")
    test_questions = []
    if os.path.exists(test_json_path):
        with open(test_json_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                test_questions.append(data["question"])
    else:
        raise FileNotFoundError(f"Test set not found at {test_json_path}")

    # Step 1: batch rewrite
    rewrite_prompts = [rewriting_prompt(q) for q in test_questions]
    rewrite_outputs = query_llm_api(rewrite_prompts, shared_llm, shared_sampling_params)
    rewritten_queries = [f"{orig} {rewritten}" for orig, rewritten in zip(test_questions, rewrite_outputs)]

    # Step 2: embed all rewritten queries
    encoder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
    embeddings = encoder.encode(rewritten_queries, show_progress_bar=True)

    # Step 3: FAISS search for all
    indices_list = []
    retrieved_queries_list = []
    retrieved_thoughts_list = []
    rerank_prompts = []
    rerank_metadata = []

    for i, (q_embed, rewritten_q) in enumerate(zip(embeddings, rewritten_queries)):
        q_embed = np.expand_dims(q_embed.astype('float32'), axis=0)
        distances, indices = retriever.index.search(q_embed, 3)
        retrieved_queries = [retriever.queries[j] for j in indices[0]]
        doc_prompt = "".join([f"Document {j}:\n{doc}\n" for j, doc in enumerate(retrieved_queries)])
        rerank_prompt = f"""
Each document in the following list has a number next to it along with its content. A query is also provided.
Respond with the numbers of the documents that are similar to the query, in order of relevance. You should always give and only give me 1 number.
Your response should EXACTLY follow the following format:
[number_1]

Query: {rewritten_q}
{doc_prompt}
"""
        rerank_prompts.append(rerank_prompt)
        rerank_metadata.append((indices[0], retrieved_queries, [retriever.thoughts[j] for j in indices[0]]))

    # Step 4: batch rerank
    rerank_outputs = reranking_batch(rerank_prompts, shared_llm, shared_sampling_params)

    for rerank_result, (faiss_indices, faiss_queries, faiss_thoughts) in zip(rerank_outputs, rerank_metadata):
        numbers = re.findall(r'\d+', rerank_result)
        if not numbers:
            best = 0
        else:
            best = int(numbers[0])
            best = min(best, len(faiss_indices) - 1)
        best_idx = faiss_indices[best]
        indices_list.append(best_idx)
        retrieved_queries_list.append(faiss_queries[best])
        retrieved_thoughts_list.append(faiss_thoughts[best])

    os.makedirs(teacher_cot_path, exist_ok=True)
    with open(os.path.join(teacher_cot_path, file_name), "w", encoding="utf-8") as f:
        for q, idx, ref_q, cot in zip(test_questions, indices_list, retrieved_queries_list, retrieved_thoughts_list):
            json.dump({
                "test_question": q,
                "matched_reference_index": int(idx),
                "matched_reference_question": ref_q,
                "matched_cot": cot
            }, f, ensure_ascii=False)
            f.write("\n")

    print(f"✅ All outputs saved in JSONL format under {teacher_cot_path}/{file_name}")




if __name__ == "__main__":
    data_prepare()

