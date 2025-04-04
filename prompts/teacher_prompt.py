def construct_teacher_prompt(example_question, example_cot, current_question,data_type):
    if data_type == 'math':
        return f"""Please study the following example carefully. It demonstrates how to think through a math problem step by step.

Note: The example is only for illustration of the reasoning process — do not reuse its numbers or logic.

Example:

{example_question}

<think>
{example_cot}
</think>

---

Now, solve the following problem using a similar style of reasoning, but **entirely your own steps**.

Problem:
{current_question}
"""
    elif data_type == 'code':
        return f"""Here is an example of a coding question and its reasoning process:

Example Question:
{example_question}

Example Reasoning:
{example_cot}

Now, please reason in a similar step-by-step manner for the following coding problem. Be clear about the thought process before writing the code.

Question:
{current_question}

Reasoning:
"""
