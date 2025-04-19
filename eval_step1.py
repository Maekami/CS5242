import os
import re
import sys
import json
import time
import random
import pickle
import argparse
from datetime import datetime
from math import comb
import gc
import torch
from datasets import load_dataset
import numpy as np
import faiss
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import vllm.envs as envs

from tqdm import tqdm
from setproctitle import setproctitle

from data_prepare import data_prepare
from cot_prepare import step1_extract_prompt,step2_comparison,step3_analyze,generate_prompt_best_of_n

import debugpy
def init_debug():
    debugpy.listen(5678)
    print("🧠 Waiting for debugger to attach on port 5678...")
    debugpy.wait_for_client()
    print("✅ Debugger attached!")

def decide_data_type(data_name):
    if 'math' in data_name or 'aime' in data_name:
        return 'math'

def parse_list(arg):
    return arg.split(',')
def save_completions(completions, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(completions, file)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True, help="model dir")
    parser.add_argument('--teacher_name', type=str, required=True)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument('--data_name', type=str, default="math")
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_tokens', type=int, default=8192)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--random_teacher", action="store_true",  help="use_random_teacher_prompt")
    parser.add_argument("--empty_teacher", action="store_true",  help="use_empty_teacher_prompt")
    parser.add_argument('--teacher_cot_path', type=str, default="./generated_cot", help="output dir")    
    parser.add_argument('--file_name',type=str, default="test_questions_with_matched_cots.jsonl",help="rag questions&cots")
    parser.add_argument('--intermediate_dir',type=str,default="./intermediate_outputs")
    parser.add_argument('--analyze_type',type=str,default="insight")
    args = parser.parse_args()
    return args

def load_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

def get_model_and_tokenizer(model_name_or_path):
    """load vLLM model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    llm = LLM(
        model=model_name_or_path,
        tensor_parallel_size=len(available_gpus), 
        trust_remote_code=True,
        gpu_memory_utilization=0.9
    )
    return llm, tokenizer

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def generate_responses(llm, prompts, sampling_params):
    completions = llm.generate(prompts, sampling_params)
    return [completion.outputs[0].text.strip() for completion in completions]

def get_conversation_prompt_by_messages(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def flatten_cot(cot_text):
    return cot_text.replace('\n', ' ').strip()
def teacher_OT(args):

    '''
    # === STEP 1: Dataset Preprocessing & Matching ===
    os.makedirs(args.intermediate_dir, exist_ok=True)
    os.makedirs(args.teacher_cot_path, exist_ok=True)
    data_type = decide_data_type(args.data_name)
    
    # Load Teacher dataset
    data_prepare(data_type=data_type, teacher_name=args.teacher_name, external_dataset_name=args.data_name, teacher_cot_path=args.teacher_cot_path, file_name=args.file_name)
    llm, tokenizer = get_model_and_tokenizer(args.model_name_or_path) 
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p
    )
    step1_output_path = os.path.join(args.intermediate_dir, "cot_rewritten.jsonl")
    step2_output_path = os.path.join(args.intermediate_dir, "cot_comparison.jsonl")
    step3_output_path = os.path.join(args.intermediate_dir, "cot_analyzation.jsonl")
    input_file = f"{args.teacher_cot_path}/{args.file_name}"
    temp1_data = step1_extract_prompt(llm=llm,tokenizer=tokenizer,sampling_params=sampling_params,input_file=input_file,output_path=step1_output_path)
    temp2_data = step2_comparison(llm=llm,tokenizer=tokenizer,sampling_params=sampling_params,temp_data=temp1_data,output_path=step2_output_path)
    temp3_data = step3_analyze(llm=llm,tokenizer=tokenizer,sampling_params=sampling_params,temp_data=temp2_data,output_path=step3_output_path,data_type=data_type,analyze_type=args.analyze_type)
    output_file = os.path.join(args.teacher_cot_path, "generated_reasonings.jsonl")
    generate_prompt_best_of_n(
    llm=llm,
    tokenizer=tokenizer,
    sampling_params=sampling_params,
    temp_data=temp3_data,
    output_path=output_file,
    analyze_type=args.analyze_type,
    n=3
    )
    '''
    from eval_refine import iterative_refinement
    # from cot_prepare import select_best_of_n
    llm, tokenizer = get_model_and_tokenizer(args.model_name_or_path)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p
    )
    input_file = "/disk4/Haonan/zihang/rag-refine/RAG-Teacher-Prompt/saved_generated_reasonings/generated_reasonings_7B_example.jsonl"
    input_data = load_jsonl(input_file)
    # multi_version_data = [{
    #     "test_question": item["test_question"],
    #     "generated_reasoning_versions": item["generated_reasoning_versions"],
    #     "analyze": item["analyze"],
    # } for item in input_data]

    final_data = iterative_refinement(llm,tokenizer,input_data,args,sampling_params,iterations=3)
    # final_data = select_best_of_n(llm, tokenizer, sampling_params, multi_version_data)
    #output_path = f"/disk4/Haonan/zihang/rag-refine/RAG-Teacher-Prompt/generated_cot_test/generated_reasonings_final.jsonl"
    output_path = os.path.join(args.teacher_cot_path, "generated_reasonings_final.jsonl")
    save_jsonl(final_data, output_path)

    # delete llm
    del llm
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

if __name__ == "__main__":
    if os.getenv("DEBUGPY") == "1":
        init_debug()
    args = parse_args()
    teacher_OT(args)