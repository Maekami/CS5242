import json
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import re
from datasets import load_dataset
import importlib.util
import os
import argparse
import vllm.envs as envs
import random
import time
from datetime import datetime
from tqdm import tqdm
from utils.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from utils.parser import *
from utils.data_loader import load_data
from utils.math_normalization import *
from utils.grader import *
from utils.data_type import decide_data_type
import pickle
from math import comb
from setproctitle import setproctitle
#from process_results.coding import LCB_generation_process_results
setproctitle("refine")  # 这里的 "my_script" 就是你希望在 nvidia-smi 显示的名字

# envs.VLLM_HOST_IP="0.0.0.0" or "127.0.0.1"

def parse_list(arg):
    return arg.split(',')

def save_completions(completions, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(completions, file)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="./", help="model dir")
    parser.add_argument('--n_sampling', type=int, default=1, help="n for sampling")
    parser.add_argument("--k", type=int, default=1, help="Value of k for pass@k calculation")
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument('--data_name', type=str, default="math", help='identify how to extract answer')
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument('--start_idx', type=int, default=0, help="data[start:end]")
    parser.add_argument('--end_idx', type=int, default=-1, help="data[start:end], if -1, data[start:]")
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--max_tokens", default=2048, type=int)
    parser.add_argument("--prompt_type", default="qwen-base", type=str)
    parser.add_argument("--prompt_file_path", default="./prompts", type=str)
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument('--stop', type=parse_list)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dtype", default='auto', type=str)
    parser.add_argument("--completions_save_dir", default='./completions', type=str)
    parser.add_argument('--cot_dir', type=str, default="./generated_cot", help="output dir") 
    parser.add_argument("--use_teacher", action="store_true",  help="use_teacher_prompt")
    parser.add_argument("--teacher_cot_path", default="./generated_cot", type=str)
    parser.add_argument("--surround_with_messages", action="store_true",  help="use_teacher_prompt")
    args = parser.parse_args()
    
    args.top_p = 1 if args.temperature == 0 else args.top_p # top_p must be 1 when using greedy 
    print(f"current stop list: {args.stop}")
    return args

def get_conversation_prompt_by_messages(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def get_three_prompt(prompt_type, data_name):
    file_path = os.path.join(".", "prompts", prompt_type, f"{data_name}.py")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    spec = importlib.util.spec_from_file_location("dynamic_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if hasattr(module, 'system_prompt'):
        system_prompt = module.system_prompt
    else:
        raise AttributeError(f"'system_prompt' not found in {file_path}")
    
    if hasattr(module, 'few_shot_prompt'):
        few_shot_prompt = module.few_shot_prompt
    else:
        raise AttributeError(f"'few_shot_prompt' not found in {file_path}")
    
    if hasattr(module, 'question_format'):
        question_format = module.question_format
    else:
        raise AttributeError(f"'question_format' not found in {file_path}")

    return system_prompt, few_shot_prompt, question_format


def infer(args):
    model_name_or_path = args.model_name_or_path
    print(f"current eval model: {model_name_or_path}")
    examples = load_data(args.data_name, args.split, args.data_dir)
    data_type = decide_data_type(args.data_name)
    n_sampling = args.n_sampling
    factor = 1
    for i in range(2, 65):
        if n_sampling % i == 0:
            factor = i
    generation_epoch = n_sampling // factor
    print(f"use n = {factor}, generation epoch is: {generation_epoch}")
    sampling_params = SamplingParams(temperature=args.temperature, 
                                     max_tokens=args.max_tokens, 
                                     n=factor,
                                     top_p=args.top_p,
                                     )
    
    model_name = "/".join(args.model_name_or_path.split("/")[-3:])
    out_file_prefix = f'{args.split}_{args.prompt_type}_t{args.temperature}'
    out_file = f'{args.output_dir}/{model_name}/{args.data_name}/{out_file_prefix}_k{args.n_sampling}_s{args.start_idx}_e{args.end_idx}.jsonl'
    
    
    if os.path.exists(out_file):
        print(f"Completely same name file({out_file}) exist, skip generation, save file and check correct")
        return
    os.makedirs(f'{args.output_dir}/{model_name}/{args.data_name}', exist_ok=True)
    os.makedirs(f'{args.completions_save_dir}/{model_name}/{args.data_name}', exist_ok=True)
    
    available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(available_gpus) == 1:
        envs.VLLM_HOST_IP="0.0.0.0" or "127.0.0.1"
    print(f"available_gpus: {available_gpus}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    prompt_batch = []
    prompt_save_path = f"{args.output_dir}/{model_name}/{args.data_name}/prompts.txt"
    with open(prompt_save_path, "w", encoding="utf-8") as f:
        f.write("")
    data_list = []
    with open(f"{args.teacher_cot_path}/generated_reasonings.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip()) 
            data_list.append(data)
    for item in tqdm(data_list, total=len(data_list)):
        # parse question and answer
        question = item["test_question"]
        cot = item["generated_reasoning"]
        # preprocess Teacher COT
        cot = cot.split("</think>")[0].strip()
        cot = cot.split("**Final Answer**")[0].strip()
        if data_type == 'math':
            cur_prompt = f"""Please reason step by step and put your final answer within \\boxed{{}}.\n Question: {question}"""
        elif data_type == 'code':
            cur_prompt = f"""Please reason step by step for the following coding problem. Provide the final implementation in Python code.\n\nQuestion:\n{question}"""
        if args.use_teacher:
            messages = [
                {"role": "user", "content": cur_prompt},
                {"role": "assistant", "content": cot},
            ]
            cur_prompt = get_conversation_prompt_by_messages(tokenizer=tokenizer, messages=messages)
        else:
            messages = [
                {"role": "user", "content": cur_prompt},
            ]
            cur_prompt = get_conversation_prompt_by_messages(tokenizer=tokenizer, messages=messages)
        with open(prompt_save_path, "a", encoding="utf-8") as f:

            f.write(f"Prompt for question {len(prompt_batch)}:\n{cur_prompt}\n\n")
        prompt_batch.append(cur_prompt)
    print(prompt_batch[0])
    
    llm = LLM(model=model_name_or_path, 
              tensor_parallel_size=len(available_gpus), 
              trust_remote_code=True, 
              gpu_memory_utilization=0.8,
              )
    
    file_outputs = []
    correct_cnt = 0
    for cur_generation_epoch in range(generation_epoch):
        completions_save_file = f'{args.completions_save_dir}/{model_name}/{args.data_name}/{out_file_prefix}_k{args.n_sampling}_s{args.start_idx}_e{args.end_idx}_gen_round{cur_generation_epoch}.pkl'
        
        completions = llm.generate(prompt_batch, sampling_params)

        save_completions(completions, completions_save_file)
        for i, item in enumerate(data_list):
            question = item["test_question"]
            generated_responses = [completions[i].outputs[j].text for j in range(len(completions[i].outputs))]
            if cur_generation_epoch == 0:
                file_outputs.append({
                    "question": question,
                    "generated_responses": generated_responses,
                })
            else:
                file_outputs[i]['generated_responses'] += generated_responses
    print("llm generate done")
    print(len(file_outputs))
    
    pass_at_k_list = []
    k = args.k
    
    if data_type == 'math':
        for i in tqdm(range(len(examples)), "check correct..."):
            d = examples[i]
            gt_cot, gt_ans = parse_ground_truth(d, args.data_name)
            generated_responses = file_outputs[i]['generated_responses']
            generated_answers = [extract_answer(generated_response, args.data_name) for generated_response in generated_responses]
            is_correct_list = [check_is_correct(generated_answer, gt_ans) for generated_answer in generated_answers]
            is_correct = any(is_correct_list)
            if is_correct:
                correct_cnt += 1
            file_outputs[i]['generated_answers'] = generated_answers
            file_outputs[i]['gold_answer'] = gt_ans
            file_outputs[i]['is_correct'] = is_correct
            file_outputs[i]['answers_correctness'] = is_correct_list
            
            if len(is_correct_list) > 1:
                correct_answers = sum(is_correct_list)
                n = len(generated_answers)
                if correct_answers > 0:
                    if n - correct_answers < k:
                        pass_at_k = 1
                    else:
                        pass_at_k = 1 - (comb(n - correct_answers, k) / comb(n, k))
                    pass_at_k_list.append(pass_at_k)
                else:
                    pass_at_k_list.append(0)
    # elif args.data_type == 'code':
    #     pbar = tqdm(range(len(data_list)), desc="check correct...")
    #     for i in pbar:
    #         ...
    #         generated_responses = file_outputs[i]['generated_responses'][0]
    #         question= ds[i]
    #         # print(LCB_generation_process_results(question, generated_responses))
    #         correct_cnt+=LCB_generation_process_results(question, generated_responses)
    #         pbar.set_postfix({
    #             "correct": f"{correct_cnt}/{i+1}",
    #             "acc": f"{correct_cnt / (i+1):.4f}"
    #         })

            
    
    temp_out_file = out_file + ".tmp"
    with open(temp_out_file, 'w', encoding='utf-8') as f:
        count = 0
        for d in tqdm(file_outputs, "writing generation to jsonl file..."):
            f.write(json.dumps(d, ensure_ascii=False))
            f.write("\n")
            count += 1
            if count % 100 == 0:
                f.flush()
        f.flush()
    os.rename(temp_out_file, out_file)
    
    print(f"correct cnt / total cnt: {correct_cnt}/{len(data_list)}")
    print(f"Acc: {correct_cnt / len(data_list):.4f}")

    if pass_at_k_list:
        average_pass_at_k = sum(pass_at_k_list) / len(pass_at_k_list)
        print(f"Pass@{k}: {sum(pass_at_k_list)}/{len(pass_at_k_list)} = {average_pass_at_k:.4f}")
    else:
        print(f"Pass@1: {correct_cnt}/{len(data_list)} = {correct_cnt / len(data_list):.4f}")

if __name__ == "__main__":
    # import debugpy
    # debugpy.listen(5678)
    # print("等待调试器附加…")
    # debugpy.wait_for_client()
    args = parse_args()
    set_seed(args.seed)
    infer(args)