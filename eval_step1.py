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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import vllm.envs as envs

from tqdm import tqdm
from setproctitle import setproctitle

from OT_data_prepare import data_prepare
from prompts.openthoughts import construct_teacher_prompt

from utils.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from utils.parser import *
from utils.data_loader import load_data
from utils.math_normalization import *
from utils.grader import *
from utils.data_type import decide_data_type

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
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_tokens', type=int, default=8192)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument("--random_teacher", action="store_true",  help="use_random_teacher_prompt")
    parser.add_argument('--teacher_cot_path', type=str, default="./generated_cot", help="output dir")    
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
def save_completions(completions, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(completions, file)

def get_model_and_tokenizer(model_name_or_path):
    """load vLLM model and tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    llm = LLM(
        model=model_name_or_path,
        tensor_parallel_size=4, 
        trust_remote_code=True,
        gpu_memory_utilization=0.8
    )
    return llm, tokenizer


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


def teacher_OT(args):
    # === STEP 1: Dataset Preprocessing & Matching ===

    data_type = decide_data_type(args.data_name)
    # Load Teacher dataset
    data_prepare(data_type,args.teacher_name, args.data_name, args.data_dir, args.teacher_cot_path, random_t = args.random_teacher)
    input_file = f"{args.teacher_cot_path}/test_questions_with_matched_cots.jsonl"

    # === STEP 2: Teacher Prompt Generation ===
    input_data = load_jsonl(input_file)
    
    llm, tokenizer = get_model_and_tokenizer(args.model_name_or_path) 
    prompts = [
        construct_teacher_prompt(item["matched_reference_question"], item["matched_cot"], item["test_question"], data_type)
        for item in input_data
    ]  
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p
    )
    print("🔄 Generating responses...")
    generated_responses = generate_responses(llm, prompts, sampling_params)   # TEACHER COT GENERATED HERE!
    output_data = [
        {
            "test_question": item["test_question"],
            "generated_reasoning": gen_response,
        }
        for item, gen_response in zip(input_data, generated_responses)
    ]
    
    output_file = os.path.join(args.teacher_cot_path, "generated_reasonings.jsonl")
    
    save_jsonl(output_data, output_file)
    
    print(f"✅ Finished! Results saved to {output_file}")
    # delete llm
    del llm
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

if __name__ == "__main__":
    args = parse_args()
    teacher_OT(args)

