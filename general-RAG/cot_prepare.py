from teacher_prompt import construct_teacher_prompt, extract_useful_prompt, generate_comparison_prompt, construct_answer
import json
import os
import time
import re  # For case-insensitive search
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
def flatten_cot(cot_text):
    return cot_text.replace('\n', ' ').strip()
def get_conversation_prompt_by_messages(tokenizer, messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def generate_comparisons_with_check_batch(llm, prompts, sampling_params, max_attempts=10):
    results_dict = {}
    failed = [(i, p) for i, p in enumerate(prompts)]

    attempt = 0
    while failed and attempt < max_attempts:
        attempt += 1
        print(f"Attempt {attempt} of {max_attempts}...")

        indices, batch_prompts = zip(*failed)
        completions = llm.generate(batch_prompts, sampling_params)

        new_failed = []
        for idx, prompt_idx in enumerate(indices):
            response = completions[idx].outputs[0].text.strip()

            if "**Comparison**" in response:
                # 截取从 "**Comparison:**" 开始的部分
                comparison_start = response.index("**Comparison**")
                truncated_response = response[comparison_start:]
                results_dict[prompt_idx] = truncated_response
            else:
                print(f"[{prompt_idx}] ⚠️ '**Comparison**' not found in attempt {attempt}, retrying...")
                new_failed.append((prompt_idx, batch_prompts[idx]))

        failed = new_failed
        if failed:
            print(f"[{len(failed)}] prompts still need to be retried.")
            time.sleep(1)
    for i in range(len(prompts)):
        if i not in results_dict:
            results_dict[i] = ""

    return [results_dict[i] for i in range(len(prompts))]

def generate_transferable_insights_with_check_batch(llm, prompts, sampling_params, max_attempts=10):
    results_dict = {}
    failed = [(i, p) for i, p in enumerate(prompts)]

    disallowed_keywords = ["final answer", "Final Answer", "Final answer", "Answer"]

    attempt = 0
    while failed and attempt < max_attempts:
        attempt += 1
        print(f"Attempt {attempt} of {max_attempts}...")

        indices, batch_prompts = zip(*failed)
        completions = llm.generate(batch_prompts, sampling_params)

        new_failed = []
        for idx, prompt_idx in enumerate(indices):
            response = completions[idx].outputs[0].text.strip()

            contains_insights = "**Transferable Insights:**" in response
            contains_disallowed = any(re.search(rf"\b{kw}\b", response, re.IGNORECASE) for kw in disallowed_keywords)

            if contains_insights and not contains_disallowed:
                start = response.index("**Transferable Insights:**")
                truncated = response[start:]
                results_dict[prompt_idx] = truncated
            else:
                reason = "missing '**Transferable Insights:**'" if not contains_insights else "contains disallowed keyword"
                print(f"[{prompt_idx}] ⚠️ {reason} in attempt {attempt}, retrying...")
                new_failed.append((prompt_idx, batch_prompts[idx]))

        failed = new_failed
        if failed:
            print(f"[{len(failed)}] prompts still need to be retried.")
            time.sleep(1)

    for i in range(len(prompts)):
        if i not in results_dict:
            results_dict[i] = ""

    return [results_dict[i] for i in range(len(prompts))]
def generate_transferable_structure_with_check_batch(llm, prompts, sampling_params, max_attempts=10):
    results_dict = {}
    failed = [(i, p) for i, p in enumerate(prompts)]

    attempt = 0
    while failed and attempt < max_attempts:
        attempt += 1
        print(f"Attempt {attempt} of {max_attempts}...")

        indices, batch_prompts = zip(*failed)
        completions = llm.generate(batch_prompts, sampling_params)

        new_failed = []
        for idx, prompt_idx in enumerate(indices):
            response = completions[idx].outputs[0].text.strip()
            
            if "**Structure:**" in response:
                start = response.index("**Structure:**")
                truncated = response[start:]
                results_dict[prompt_idx] = truncated
            else:
                print(f"[{prompt_idx}] ⚠️ Missing '**Structure:**' in attempt {attempt}, retrying...")
                new_failed.append((prompt_idx, batch_prompts[idx]))

        failed = new_failed
        if failed:
            print(f"[{len(failed)}] prompts still need to be retried.")
            time.sleep(1)

    for i in range(len(prompts)):
        if i not in results_dict:
            results_dict[i] = ""

    return [results_dict[i] for i in range(len(prompts))]
def generate_responses_with_check_batch(llm, prompts, sampling_params, max_attempts=10):
    results_dict = {}
    failed = [(i, p) for i, p in enumerate(prompts)]

    attempt = 0
    while failed and attempt < max_attempts:
        attempt += 1
        print(f"Attempt {attempt} of {max_attempts}...")
        
        indices, batch_prompts = zip(*failed)
        completions = llm.generate(batch_prompts, sampling_params)

        new_failed = []
        for idx, prompt_idx in enumerate(indices):
            response = completions[idx].outputs[0].text.strip()
            if "</think>" in response or "Final Answer" in response:
                results_dict[prompt_idx] = response
            else:
                print(f"[{prompt_idx}] ⚠️ Missing </think> in attempt {attempt}, retrying...")
                new_failed.append((prompt_idx, batch_prompts[idx]))

        failed = new_failed
        if failed:
            print(f"[{len(failed)}] prompts still need to be retried.")
            time.sleep(1)

    for i in range(len(prompts)):
        if i not in results_dict:
            results_dict[i] = ""

    return [results_dict[i] for i in range(len(prompts))]
def generate_raw_responses_with_number_check(llm, prompts, sampling_params, n_choices=3, max_attempts=5):
    import time
    import re

    results_dict = {}
    failed = [(i, p) for i, p in enumerate(prompts)]
    pattern = rf"\b([1-{n_choices}])\b"

    attempt = 0
    while failed and attempt < max_attempts:
        attempt += 1
        print(f"Attempt {attempt} of {max_attempts} for numeric check...")

        indices, batch_prompts = zip(*failed)
        completions = llm.generate(batch_prompts, sampling_params)

        new_failed = []
        for idx, prompt_idx in enumerate(indices):
            response = completions[idx].outputs[0].text.strip()
            if re.search(pattern, response):
                results_dict[prompt_idx] = response
            else:
                print(f"[{prompt_idx}] ⚠️ No valid number 1-{n_choices} found in response: \"{response}\"")
                new_failed.append((prompt_idx, batch_prompts[idx]))

        failed = new_failed
        if failed:
            time.sleep(1)

    # fallback to empty string if still failed
    for i in range(len(prompts)):
        if i not in results_dict:
            results_dict[i] = ""  # keep the output format consistent

    return [results_dict[i] for i in range(len(prompts))]
def step1_extract_prompt(llm,tokenizer,sampling_params,input_file,output_path):
    input_data = load_jsonl(input_file)
    prompts_step_1 = [
        extract_useful_prompt(item["matched_reference_question"], flatten_cot(item["matched_cot"]))
        for item in input_data
    ]
    prompt_batch_1 = []
    for prompt in prompts_step_1:
        messages = [
                {"role": "user", "content": prompt},
            ]
        cot_prompts = get_conversation_prompt_by_messages(tokenizer=tokenizer, messages=messages)
        prompt_batch_1.append(cot_prompts)
    print("🔄 Generating responses...")
    generated_responses = llm.generate(prompt_batch_1, sampling_params)
    temporary_data = []
    with open(output_path, "w", encoding="utf-8") as f:
        for item, gen_response in zip(input_data, generated_responses):
            data_piece = {
                "test_question": item["test_question"],
                "matched_cot": gen_response.outputs[0].text.strip(),
                "matched_question": item["matched_reference_question"]
            }
            temporary_data.append(data_piece)
            json.dump(data_piece, f)
            f.write("\n")
    print(f"✅ new cot 已保存到 {output_path}")
    return temporary_data

def step2_comparison(llm,tokenizer,sampling_params,temp_data,output_path):
    prompts_step_2 = [
        generate_comparison_prompt(item["matched_question"], item["test_question"])
        for item in temp_data
    ]
    prompt_batch_2 = []
    for prompt in prompts_step_2:
        messages = [
                {"role": "user", "content": prompt},
            ]
        cot_prompts = get_conversation_prompt_by_messages(tokenizer=tokenizer, messages=messages)
        prompt_batch_2.append(cot_prompts)
    generated_responses = generate_comparisons_with_check_batch(llm, prompt_batch_2, sampling_params)
    temp2_data = []
    with open(output_path, "w", encoding="utf-8") as f:
        for item, gen_response in zip(temp_data, generated_responses):
            data_piece = {
                "test_question": item["test_question"],
                "matched_cot": item["matched_cot"],
                "matched_question": item["matched_question"],
                "comparison_idea": gen_response,
            }
            temp2_data.append(data_piece)
            json.dump(data_piece, f)
            f.write("\n")

    print(f"✅ new cot with comparison 已保存到 {output_path}")
    return temp2_data
def step3_analyze(llm,tokenizer,sampling_params,temp_data,output_path,analyze_type):
    prompts = [
        construct_teacher_prompt(item["matched_question"], flatten_cot(item["matched_cot"]), item["test_question"], item["comparison_idea"],analyze_type).strip()
        for item in temp_data
    ]  
    
    prompt_batch_3 = []
    for prompt in prompts:
        messages = [
                {"role": "user", "content": prompt},
            ]
        cot_prompts = get_conversation_prompt_by_messages(tokenizer=tokenizer, messages=messages)
        prompt_batch_3.append(cot_prompts)
    print(prompt_batch_3[0])
    if analyze_type=='insight':
        generated_responses = generate_transferable_insights_with_check_batch(llm, prompt_batch_3, sampling_params)
    elif analyze_type=='structure':
        generated_responses = generate_transferable_structure_with_check_batch(llm, prompt_batch_3, sampling_params)
    temp3_data=[]
    with open(output_path, "w", encoding="utf-8") as f:
        for item, gen_response in zip(temp_data, generated_responses):
            data_piece={
                "test_question": item["test_question"],
                "analyze": gen_response,
                "matched_question": item["matched_question"],
                "matched_cot": item["matched_cot"],
                "comparison_idea": item["comparison_idea"],
            }
            temp3_data.append(data_piece)
            json.dump(data_piece, f)
            f.write("\n")

    print(f"✅ new cot with comparison 已保存到 {output_path}")
    return temp3_data
def generate_multiple_prompt_versions(llm, tokenizer, sampling_params, temp_data,analyze_type, n=5):
    prompts = [
        construct_answer(item["matched_question"],item["test_question"],item["comparison_idea"], item["analyze"],analyze_type).strip()
        for item in temp_data
    ]

    prompt_batch = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        cot_prompt = get_conversation_prompt_by_messages(tokenizer=tokenizer, messages=messages)
        prompt_batch.extend([cot_prompt] * n)

    print(f"🔄 Generating {n} responses per question...")
    all_responses = generate_responses_with_check_batch(llm, prompt_batch, sampling_params)

    grouped_outputs = [all_responses[i:i+n] for i in range(0, len(all_responses), n)]

    output_data = []
    for item, gen_versions in zip(temp_data, grouped_outputs):
        data_piece = {
            "test_question": item["test_question"],
            "generated_reasoning_versions": gen_versions,
            "analyze": item["analyze"],
        }
        output_data.append(data_piece)

    return output_data


def construct_selection_prompt(question, versions):
    numbered = "\n\n".join([
        f"### Reasoning {i+1}\n{v.strip()}" for i, v in enumerate(versions)
    ])
    return f"""
You are given a math problem and {len(versions)} candidate step-by-step solutions.

---

**Problem:**  
{question}

---

{numbered}

---

Your task is to carefully read all {len(versions)} candidate reasonings and choose the **best** one in terms of correctness and completeness.

Please output ONLY the number (1 to {len(versions)}) corresponding to the best solution.
""".strip()


def select_best_of_n(llm, tokenizer, sampling_params, output_data):
    selection_prompts = []
    for item in output_data:
        prompt_text = construct_selection_prompt(item["test_question"], item["generated_reasoning_versions"])
        messages = [{"role": "user", "content": prompt_text}]
        prompt = get_conversation_prompt_by_messages(tokenizer, messages)
        selection_prompts.append(prompt)

    print("🔎 Selecting best reasoning version...")
    # best_choices = generate_raw_responses_with_number_check(llm, selection_prompts, sampling_params, n_choices=3)
    best_choices = generate_responses_with_check_batch(llm, selection_prompts, sampling_params)

    # 选出最佳序号并更新每条数据
    for item, best in zip(output_data, best_choices):
        try:
            selected_index = int(re.search(r"\b([1-3])\b", best).group(1)) - 1
        except:
            selected_index = 0  # fallback
        item["selected_index"] = selected_index
        item["generated_reasoning"] = item["generated_reasoning_versions"][selected_index]

    return output_data


def generate_prompt_best_of_n(llm, tokenizer, sampling_params, temp_data, output_path,analyze_type, n=3):
    # 1. 生成多个版本
    multi_version_data = generate_multiple_prompt_versions(llm, tokenizer, sampling_params, temp_data,analyze_type, n=n)

    # 2. 评估选择最优版本
    final_data = select_best_of_n(llm, tokenizer, sampling_params, multi_version_data)

    # 3. 保存到文件
    save_jsonl(final_data, output_path)
    print(f"✅ Best-of-{n} reasoning saved to {output_path}")
    return final_data
