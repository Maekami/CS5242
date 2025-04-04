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
from prompts.teacher_prompt import construct_teacher_prompt

from eval_refine import iterative_refinement
import debugpy
def init_debug():
    import debugpy
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
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max_tokens', type=int, default=8192)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_of_responses', type=int, default=1)
    parser.add_argument("--random_teacher", action="store_true",  help="use_random_teacher_prompt")
    parser.add_argument("--empty_teacher", action="store_true",  help="use_empty_teacher_prompt")
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
    available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    llm = LLM(
        model=model_name_or_path,
        tensor_parallel_size=len(available_gpus), 
        trust_remote_code=True,
        gpu_memory_utilization=0.8
    )
    return llm, tokenizer

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def generate_responses(llm, prompts, sampling_params):
    completions = llm.generate(prompts, sampling_params)
    #return [completion.outputs[0].text.strip() for completion in completions]
    return [[completion.outputs[i].text.strip() for i in range(len(completion.outputs))] for completion in completions]

def generate_responses_with_check_one_by_one(llm, prompts, sampling_params, max_attempts=10):
    validated_responses = []
    for i, prompt in enumerate(tqdm(prompts, desc="Generating with tag check")):
        for attempt in range(max_attempts):
            try:
                completion = llm.generate([prompt], sampling_params)[0].outputs[0].text.strip()
                if "</think>" in completion or "Final Answer" in completion:
                    validated_responses.append(completion)
                    break
                else:
                    print(f"[{i}] ⚠️ Missing </think> in attempt {attempt+1}, retrying...")
            except Exception as e:
                print(f"[{i}] ❌ Error on attempt {attempt+1}: {e}")
                time.sleep(1)
        else:
            print(f"[{i}] ❌ Failed after {max_attempts} attempts. Using fallback.")
            validated_responses.append(completion if 'completion' in locals() else "")
    return validated_responses

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
    llm, tokenizer = get_model_and_tokenizer(args.model_name_or_path) 
    # Load Teacher dataset
    # data_prepare(data_type,tokenizer, args.teacher_name, args.data_name, args.data_dir, args.teacher_cot_path, random_t = args.random_teacher,empty_t = args.empty_teacher)
    input_file = f"{args.teacher_cot_path}/test_questions_with_matched_cots.jsonl"

    # === STEP 2: Teacher Prompt Generation ===
    input_data = load_jsonl(input_file)
    
    '''
    prompts = [
        construct_teacher_prompt(item["matched_reference_question"], item["matched_cot"], item["test_question"], data_type)
        for item in input_data
    ]  
    
    prompt_batch = []
    for prompt in prompts:
        messages = [
                {"role": "user", "content": prompt},
            ]
        cot_prompts = get_conversation_prompt_by_messages(tokenizer=tokenizer, messages=messages)
        prompt_batch.append(cot_prompts)
    sampling_params = SamplingParams(
        n=args.num_of_responses,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p
    )
    print("🔄 Generating responses...")
    #generated_responses = generate_responses_with_check_one_by_one(llm, prompt_batch, sampling_params)   # TEACHER COT GENERATED HERE!
    generated_responses = generate_responses(llm, prompt_batch, sampling_params) # TODO: check_one_by_one要改成可以生成多次response的
    output_data = [
        {
            "test_question": item["test_question"],
            "generated_reasoning": gen_response,
        }
        for item, gen_response in zip(input_data, generated_responses)
    ]
    
    output_file = os.path.join(args.teacher_cot_path, "generated_reasonings_v2.jsonl")
    
    save_jsonl(output_data, output_file)
    
    print(f"✅ Finished! Results saved to {output_file}")


    # refinement process
    
    print("##########################Start refinement##########################")
    input_data = [
        {
            "example_question": item["matched_reference_question"],
            "example_cot": item["matched_cot"],
            "current_question": item["test_question"],
            "cot_candidates": gen_response,
        }
        for item, gen_response in zip(input_data, generated_responses)
    ]
    '''
    input_file2 = f"{args.teacher_cot_path}/generated_reasonings_v2.jsonl"
    generated_responses = load_jsonl(input_file2)
    input_data = [
        {
            "example_question": item["matched_reference_question"],
            "example_cot": item["matched_cot"],
            "current_question": item["test_question"],
            "cot_candidates": gen_response["generated_reasoning"],
        }
        for item, gen_response in zip(input_data, generated_responses)
    ]
    #input_data = input_data[:2]

    data = iterative_refinement(model=llm, tokenizer=tokenizer, input_data=input_data, args=args, iterations=2)


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

