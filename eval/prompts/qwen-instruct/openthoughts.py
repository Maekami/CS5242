def construct_teacher_prompt(example_question, example_cot, current_question):
    """构造 Prompt"""
    return f"""Here is an example of a question and its reasoning process:

Example Question: {example_question}
Example Reasoning:
{example_cot}

Now, please reason in a similar step-by-step manner for the following question.


Question: {current_question}
Reasoning: 
"""