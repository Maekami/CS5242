def construct_teacher_prompt(example_question, example_cot, current_question,data_type):
    if data_type == 'math':
        return f"""Here is an example of a math question and its reasoning process:

Example Question: {example_question}
Example Reasoning:
{example_cot}

Now, please reason in a similar step-by-step manner for the following question.


Question: {current_question}
Reasoning: 
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
