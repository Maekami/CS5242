from vllm import LLM, SamplingParams
from typing import Any, Dict, List, Optional
from refine_prompt import construct_selection_prompt,construct_suggestion_prompt,construct_refinement_prompt
from tqdm import tqdm
import re
import os
import json
import time

def save_jsonl(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

def match_number(text):
    match = re.search(r'\b\d+\b', text)
    if match:
        number = int(match.group())
        #print(number)
    else:
        #print("Number not found")
        number = -99
    return number

def get_conversation_prompt_by_messages(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def get_final_prompt(tokenizer, prompts):
    prompt_batch = []
    for prompt in prompts:
        messages = [
                {"role": "user", "content": prompt},
            ]
        cot_prompts = get_conversation_prompt_by_messages(tokenizer=tokenizer, messages=messages)
        prompt_batch.append(cot_prompts)
    return prompt_batch

def extract_reasoning_simple(cot: str) -> str:
    """
    提取 reasoning：优先截断在 </think>，否则在 'Final Answer' 或 'final answer' 前的句号。
    """
    # 1. 优先处理 </think>
    end_think_match = re.search(r"</think>", cot, flags=re.IGNORECASE)
    if end_think_match:
        cot = cot[:end_think_match.start()].strip()

    # 2. 匹配 Final Answer 字样
    final_answer_match = re.search(r"\bFinal Answer\b", cot, flags=re.IGNORECASE)
    if final_answer_match:
        # 找 final answer 前最近的句号
        before_text = cot[:final_answer_match.start()]
        last_period_index = before_text.rfind(".")
        if last_period_index != -1:
            return cot[:last_period_index + 1].strip()  # 保留句号
        else:
            return before_text.strip()  # 没句号就保留整个前缀

    # 3. 都没有，返回全文
    return cot.strip()

'''
# one generation
def generate_once(model: LLM, tokenizer, prompt, sampling_params):
    message = [{"role": "user", "content": prompt}]
    conv_prompt = get_conversation_prompt_by_messages(tokenizer=tokenizer, messages=message)
    completion = model.generate(conv_prompt, sampling_params)
    return completion[0].outputs[0].text.strip()

def generate_feedback(model: LLM, tokenizer, input_data, add_reference=False, max_attempts=10):
    output_data = []
    sampling_params = SamplingParams(
        n=1,
        temperature=0.7,
        max_tokens=8192,
        top_p=1.0
    )
    
    for item in tqdm(input_data, desc="Generating feedback..."):
        current_attempts = 0
        cots = item["cot_candidates"]
        if add_reference:
            prompt = construct_evaluation_prompt(item["example_question"], item["example_cot"], item["current_question"], cots)
        else:
            prompt = construct_evaluation_prompt(None, None, item["current_question"], cots)

        while current_attempts < max_attempts:
            try:
                current_attempts += 1
                generated_response = generate_once(model, tokenizer, prompt, sampling_params)

                # 提取并验证
                if "## Best Path:" not in generated_response or "## Optimization Advice:" not in generated_response:
                    raise ValueError("Missing expected sections in the response.")

                best_cot_section = generated_response.split("## Best Path:")[1].split("## Optimization Advice:")[0]
                advice_section = generated_response.split("## Optimization Advice:")[1].split("\n\n")[0].strip()

                best_cot_idx = match_number(best_cot_section)
                if not isinstance(best_cot_idx, int) or best_cot_idx < 1 or best_cot_idx > len(cots):
                    raise ValueError("Invalid best_cot_idx extracted.")
                if not advice_section or len(tokenizer.encode(advice_section, add_special_tokens=False)) < 20:
                    raise ValueError("Advice section is empty or too short.")

                best_cot = cots[best_cot_idx - 1]
                advice = advice_section

                data = {
                    "example_question": item["example_question"],
                    "example_cot": item["example_cot"],
                    "current_question": item["current_question"],
                    "best_cot": best_cot,
                    "advice": advice,
                }

                output_data.append(data)
                break  # 成功则跳出 while True
            except Exception as e:
                #print(f"[Retrying] Failed to extract from response. Retrying...\nReason: {e}\nResponse: {generated_response[:300]}...")
                tqdm.write(f"[Retrying] Failed to extract from response. Retrying...\nReason: {e}\n")

        #output_data.append(data)

    return output_data


def refine_cot(model: LLM, tokenizer, input_data, num_of_responses, add_reference=False, max_attempts=10):
    output_data = []
    sampling_params = SamplingParams(
        n=1,
        temperature=0.7,
        max_tokens=8192,
        top_p=1.0
    )

    for item in tqdm(input_data, desc="Generating refinement..."):
        current_attempts = 0
        cot_candidates = []
        if add_reference:
            prompt = construct_refinement_prompt(item["example_question"], item["example_cot"], item["current_question"], item["best_cot"], item["advice"])
        else:
            prompt = construct_refinement_prompt(None, None, item["current_question"], item["best_cot"], item["advice"])

        while len(cot_candidates) < num_of_responses and current_attempts < max_attempts:
            try:
                current_attempts += 1
                generated_response = generate_once(model, tokenizer, prompt, sampling_params)

                cot_section = re.findall(r'<response>(.*?)</response>', generated_response, re.DOTALL)
                if not cot_section:
                    raise ValueError("cot section is empty.")
                if len(tokenizer.encode(cot_section[0].strip(), add_special_tokens=False)) < 50:
                    raise ValueError("cot section is too short.")

                cot_candidates.append(cot_section[0].strip())
            except Exception as e:
                #print(f"[Retrying] Failed to extract from response. Retrying...\nReason: {e}\nResponse: {generated_response[:300]}...")
                tqdm.write(f"[Retrying] Failed to extract from response. Retrying...\nReason: {e}\n")

        data = {
            "example_question": item["example_question"],
            "example_cot": item["example_cot"],
            "current_question": item["current_question"],
            "cot_candidates": cot_candidates,
        }

        output_data.append(data)

    return output_data
#
'''

def generate_selection_with_check_batch(llm, prompts, sampling_params, origin_input_data, max_attempts=10):
    results_dict = {}
    cot_candidates=[item["generated_reasoning_versions"] for item in origin_input_data] # 每个问题的候选cot
    cot_counts = len(cot_candidates[0]) # 每个问题的候选cot数量
    failed = [(i, p) for i, p in enumerate(prompts)]

    attempt = 0
    while failed and attempt < max_attempts:
        attempt += 1
        tqdm.write(f"Attempt {attempt} of {max_attempts}...")

        indices, batch_prompts = zip(*failed)
        completions = llm.generate(batch_prompts, sampling_params)

        new_failed = []
        for idx, prompt_idx in enumerate(indices):
            response = completions[idx].outputs[0].text.strip()
            if "</think>" in response:
                response = response.split("</think>")[1].strip() # 将<think>部分去掉
            
            match = re.search(r"\{\{(\d+)\}\}", response, re.DOTALL) # 允许跨行匹配
            if match:
                best_path_idx = int(match.group(1))
                if 1 <= best_path_idx <= cot_counts:
                    results_dict[prompt_idx] = cot_candidates[prompt_idx][best_path_idx-1]
            else:
                tqdm.write(f"[{prompt_idx}] ⚠️ No valid number in {{}} found in attempt {attempt}, retrying...")
                new_failed.append((prompt_idx, batch_prompts[idx]))

        failed = new_failed
        if failed:
            tqdm.write(f"[{len(failed)}] prompts still need to be retried.")
            time.sleep(1)

    for i in range(len(prompts)):
        if i not in results_dict:
            results_dict[i] = cot_candidates[i][0] # 如果选择不出来，默认使用第一个

    return [results_dict[i] for i in range(len(prompts))]

def generate_suggestion_with_check_batch(llm, prompts, sampling_params, max_attempts=10):
    results_dict = {}
    failed = [(i, p) for i, p in enumerate(prompts)]

    attempt = 0
    while failed and attempt < max_attempts:
        attempt += 1
        tqdm.write(f"Attempt {attempt} of {max_attempts}...")

        indices, batch_prompts = zip(*failed)
        completions = llm.generate(batch_prompts, sampling_params)

        new_failed = []
        for idx, prompt_idx in enumerate(indices):
            response = completions[idx].outputs[0].text.strip()
            if "</think>" in response:
                response = response.split("</think>")[1].strip() # 将<think>部分去掉
            
            # 匹配多行文本
            match = re.search(r"\{\{(.*?)\}\}", response, re.DOTALL)
            if match:
                inner_text = match.group(1).strip()
                results_dict[prompt_idx] = f"{inner_text}"  # 例如 {{The calculation...}}
            else:
                tqdm.write(f"[{prompt_idx}] ⚠️ No {{...}} block found in attempt {attempt}, retrying...")
                new_failed.append((prompt_idx, batch_prompts[idx]))

        failed = new_failed
        if failed:
            tqdm.write(f"[{len(failed)}] prompts still need to be retried.")
            time.sleep(1)

    for i in range(len(prompts)):
        if i not in results_dict:
            results_dict[i] = "No issue is given, please review the path for yourself"

    return [results_dict[i] for i in range(len(prompts))]

# 
def generate_refinement_with_check_batch(llm, prompts, sampling_params, max_attempts=10):
    results_dict = {}
    failed = [(i, p) for i, p in enumerate(prompts)]

    attempt = 0
    while failed and attempt < max_attempts:
        attempt += 1
        tqdm.write(f"Attempt {attempt} of {max_attempts}...")

        indices, batch_prompts = zip(*failed)
        completions = llm.generate(batch_prompts, sampling_params)

        new_failed = []
        for idx, prompt_idx in enumerate(indices):
            response = completions[idx].outputs[0].text.strip()
            
            # 检查回答是否包含</think>或者Final Answer
            if "</think>" in response or "Final Answer" in response:
                results_dict[prompt_idx] = f"{response}"  # 例如 {{Okay, let's see. So, ...}}
            else:
                tqdm.write(f"[{prompt_idx}] ⚠️ Missing </think> in attempt {attempt}, retrying...")
                new_failed.append((prompt_idx, batch_prompts[idx]))

        failed = new_failed
        if failed:
            tqdm.write(f"[{len(failed)}] prompts still need to be retried.")
            time.sleep(1)

    for i in range(len(prompts)):
        if i not in results_dict:
            results_dict[i] = ""

    return [results_dict[i] for i in range(len(prompts))]

def selection(model: LLM, tokenizer, input_data, sampling_params, max_attempts=10):
    prompts = [
        construct_selection_prompt(current_question=item["test_question"], cot_candidates=item["generated_reasoning_versions"]).strip()
        for item in input_data
    ]

    prompt_batch = get_final_prompt(tokenizer, prompts)

    generated_responses = generate_selection_with_check_batch(model, prompt_batch, sampling_params, input_data, max_attempts) # TODO:

    output_data = [{
        "test_question": item["test_question"],
        "generated_reasoning": best_path,
    } for item, best_path in zip(input_data, generated_responses)]

    return output_data

def suggestion(model: LLM, tokenizer, input_data, sampling_params, max_attempts=10):
    prompts = [
        construct_suggestion_prompt(current_question=item["test_question"], best_path=extract_reasoning_simple(item["generated_reasoning"])).strip()
        for item in input_data
    ]

    prompt_batch = get_final_prompt(tokenizer, prompts)

    generated_responses = generate_suggestion_with_check_batch(model, prompt_batch, sampling_params, max_attempts) # TODO:

    output_data = [{
        "test_question": item["test_question"],
        "generated_reasoning": item["generated_reasoning"],
        "issue": issue,
    } for item, issue in zip(input_data, generated_responses)]  

    return output_data

def refinement(model: LLM, tokenizer, input_data, sampling_params, num_of_responses=3, max_attempts=10):
    prompts = [
        construct_refinement_prompt(current_question=item["test_question"], best_path=item["generated_reasoning"], issue=item["issue"]).strip()
        for item in input_data
    ]

    prompts = [item for item in prompts for _ in range(num_of_responses)] # 将每个问题复制n次，以生成n次回答

    prompt_batch = get_final_prompt(tokenizer, prompts)

    generated_responses = generate_refinement_with_check_batch(model, prompt_batch, sampling_params, max_attempts) # TODO:

    grouped_outputs = [generated_responses[i:i+num_of_responses] for i in range(0, len(generated_responses), num_of_responses)] # 将同一个问题的回答合并

    output_data = [{
        "test_question": item["test_question"],
        "generated_reasoning_versions": generated_reasoning_versions,
    } for item, generated_reasoning_versions in zip(input_data, grouped_outputs)]  

    return output_data

def iterative_refinement(model: LLM, tokenizer, input_data, args, sampling_params, iterations=1):   
    current_refined_version = input_data
    for i in tqdm(range(iterations), desc="Iterative refinement..."):
        tqdm.write(f"#############start refinement iteration {i+1}#############")
        # feedback = generate_feedback(model, tokenizer, input_data=current_refined_version)
        # current_refined_version = refine_cot(model=model, tokenizer=tokenizer, input_data=feedback, num_of_responses=3)
        # temp_data1 = selection(model, tokenizer, current_refined_version, max_attempts=10)
        temp_data1 = suggestion(model, tokenizer, current_refined_version, sampling_params, max_attempts=10)
        temp_data2 = refinement(model, tokenizer, temp_data1, sampling_params, num_of_responses=3, max_attempts=10)
        current_refined_version = selection(model, tokenizer, temp_data2, sampling_params, max_attempts=10)

        # 每次iteration保存一次文件，跑完整个流程再进行eval
        file_name = f"generated_reasonings_iteration_{i+1}.jsonl"
        output_file_path = os.path.join(args.teacher_cot_path, file_name)
        save_jsonl(current_refined_version, output_file_path)
        tqdm.write(f"iteration {i+1} results saved to {output_file_path}")

    final_version = current_refined_version

    return final_version
    