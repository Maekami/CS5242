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
from together import Together
from utils.data_loader import load_data
from utils.parser import *



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    def __init__(self, base_url, api_key, save_dir="./database"):
        self.save_dir = save_dir
        self.index = None
        self.queries = []
        self.thoughts=[]
        self.base_url = base_url
        self.api_key = api_key

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
        response = query_llm_api(prompt, self.base_url, self.api_key)
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

    def rewriting(self, query):
        prompt=f"""Rewrite the given query to optimize it for both keyword-based and semantic-similarity search methods. Follow these guidelines:

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
        response = query_llm_api(prompt, self.base_url, self.api_key)
        return response

    def search(self, query, k=1, random_t= False, enable_reranking=True,enable_rewriting=True):
        """执行搜索"""
        if random_t:
            indices = np.random.randint(0, len(self.queries),size=k)
            indices = indices.tolist()
            return indices,[self.queries[i] for i in indices],[self.thoughts[i] for i in indices]

        if enable_rewriting:
            query += self.rewriting(query)
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


def query_llm_api(prompt, base_url, api_key,max_len=1024):
    """client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="qwen2.5-7b-instruct",
        messages=messages,
        temperature=0.7,
        max_completion_tokens=max_len,
    )
    return response.choices[0].message.content
    """
    client = Together(
        base_url='https://api.together.xyz/v1',
        api_key='421a34ebbd5811e9a12cd1184f6d9ae4ad427a4340cf09e97d74eb82d95000da'
    )
    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        messages=messages,
        temperature=0.7,
        max_completion_tokens=max_len,
    )
    response=response.choices[0].message.content
    response=response.split('</think>')[-1]
    return response

class LLM_Responser:
    def __init__(self,model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",temperature=0.7,max_tokens=8192,top_p=1.0):
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


class ThoughtfulGenerator:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key

    def generate(self, query, retrieved_queries,retrieved_thoughts, max_length=1024, num_candidates=3):
        """生成带思考过程的回答"""
        prompt = self._format_prompt(query, retrieved_queries,retrieved_thoughts)
        response = query_llm_api(prompt, self.base_url, self.api_key, max_length)

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

def load_queries_thoughts(teacher_name="open-thoughts/OpenThoughts-114k",data_type="math"):
    ds = load_dataset(teacher_name, "metadata", split="train")
    reference_data = [item for item in ds if item.get("domain") == data_type]
    queries = [item.get("problem", "") for item in reference_data]
    thoughts = [item.get("deepseek_reasoning", "") for item in reference_data]
    return queries, thoughts

def get_repo(data_name):
    if data_name=='math_500':
        return "HuggingFaceH4/MATH-500"
    elif data_name=='aime24':
        return "HuggingFaceH4/aime_2024"

def data_prepare(data_type="math", teacher_name="open-thoughts/OpenThoughts-114k", external_dataset_name="math_500", teacher_load_dir="./data",
                 teacher_cot_path="./generated_cot", random_t= False, base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                 api_key = "sk-64405534cdbc4692aa8d32c8cecc5a46", database_save_path="./database", enable_reranking=True,enable_rewriting=True):
    # 初始化检索器
    database_save_path=os.path.join(database_save_path, data_type)
    #llm_responser=LLM_Responser()
    retriever = Retriever(base_url, api_key, database_save_path)

    # 构建并保存（只需运行一次）
    if not os.path.exists(retriever.save_dir):
        queries, thoughts = load_queries_thoughts(teacher_name=teacher_name, data_type=data_type)
        retriever.build_and_save(queries, thoughts)
    else:
        retriever = retriever.load()
    if teacher_load_dir is not None:
        if external_dataset_name == 'math_500':
            external_dataset_name='math'
        external_ds = load_data(external_dataset_name, 'test', teacher_load_dir)
    else:
        if external_dataset_name == 'aime24':
            train_type = 'train'
        elif external_dataset_name == 'math_500':
            train_type = 'test'
        external_ds  = load_data(get_repo(external_dataset_name), train_type, teacher_load_dir)
    test_questions = [{"question": parse_question(example, external_dataset_name)} for example in external_ds]
    test_question_texts = [q["question"] for q in test_questions]
    test_question_texts=test_question_texts[:10]
    indices_list=[]
    retrieved_queries_list=[]
    retrieved_thoughts_list=[]
    for q in test_question_texts:
        indices,retrieved_queries,retrieved_thoughts=retriever.search(q, k=1,random_t=random_t,enable_reranking=enable_reranking,enable_rewriting=enable_rewriting)
        if len(indices)==1:
            indices_list.append(indices[0])
            retrieved_queries_list.append(retrieved_queries[0])
            retrieved_thoughts_list.append(retrieved_thoughts[0])
        else:
            indices_list.append(indices)
            retrieved_queries_list.append(retrieved_queries)
            retrieved_thoughts_list.append(retrieved_thoughts)

    os.makedirs(f'{teacher_cot_path}', exist_ok=True)
    with open(f"{teacher_cot_path}/test_questions_with_matched_cots.jsonl", "w", encoding="utf-8") as f:
        for q, idx, ref_q, cot in zip(test_question_texts, indices_list, retrieved_queries_list, retrieved_thoughts_list):
            json.dump({
                "test_question": q,
                "matched_reference_index": int(idx),
                "matched_reference_question": ref_q,
                "matched_cot": cot
            }, f, ensure_ascii=False)
            f.write("\n")
    print("✅ All outputs saved in JSONL format under"+teacher_cot_path+"/test_questions_with_matched_cots.jsonl")




if __name__ == "__main__":
    data_prepare()

