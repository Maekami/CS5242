from vllm import LLM, SamplingParams
from typing import Any, Dict, List, Optional
from prompts.refine import construct_evaluation_prompt, construct_refinement_prompt
from tqdm import tqdm
import re
import os
import json

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

def iterative_refinement(model: LLM, tokenizer, input_data, args ,iterations=1):   
    current_refined_version = input_data
    for i in tqdm(range(iterations), desc="Iterative refinement..."):
        tqdm.write(f"#############start refinement iteration {i+1}#############")
        feedback = generate_feedback(model, tokenizer, input_data=current_refined_version, add_reference=False)
        current_refined_version = refine_cot(model=model, tokenizer=tokenizer, input_data=feedback, num_of_responses=3, add_reference=False)

        # TODO: 增加eval流程，每个iteration都做一次eval
        # FIXME: 也许每次iteration保存一次文件，跑完整个流程再进行eval
        file_name = f"iteration_{i+1}_generated_reasonings.jsonl"
        output_file_path = os.path.join(args.teacher_cot_path, file_name)
        save_jsonl(current_refined_version, output_file_path)
        tqdm.write(f"iteration {i+1} results saved to {output_file_path}")

    final_version = current_refined_version

    return final_version
    