# def construct_answer(example_question, question, comparison, insights):
#     return f"""
# ### Math Problem Solving Task

# Solve the following math problem clearly and step by step.

# You are given:
# - An **example problem** that is similar to the current problem.
# - A **comparison** between the two problems to help identify transferable strategies.
# - Several **potentially helpful insights**.

# Use the example and comparison to guide your reasoning.  
# Only apply insights if they are clearly useful to the current problem.

# ---

# **Example Problem:**  
# {example_question}

# ---

# {comparison}

# ---

# **Potentially helpful insights:**  
# {insights}

# ---

# **Problem:**  
# {question}
# """
def construct_answer(example_question, question, comparison, analyze,analyze_type):
    if analyze_type=='insight':
        return f"""
### Math Problem Solving Task

Solve the following math problem clearly and step by step.

You are given:
- An **example problem** that is similar to the current problem.
- A **comparison** between the two problems to help identify transferable strategies.
- Several **potentially helpful insights**.

Use the example and comparison to guide your reasoning.  
Only apply insights if they are clearly useful to the current problem.

---

**Example Problem:**  
{example_question}

---

{comparison}

---

{analyze}

---

**Problem:**  
{question}
"""
    elif analyze_type=='structure':
        return f"""
### Math Problem Solving Task

Solve the following math problem clearly and step by step.

You are given:
- An **example problem** that is similar to the current problem.
- A **comparison** between the two problems to help identify transferable strategies.
- A **general step-by-step structure** based on how the example problem was solved.

Use the example and comparison to guide your reasoning. 
Follow the structure to solve the current problem. Adapt the steps if needed.

---

**Example Problem:**  
{example_question}

---

{comparison}

---

{analyze}

---

**Problem:**  
{question}
    """

def extract_useful_prompt(example_question, example_cot):
    return f"""
    I have an example question:
{example_question}

And I have a detailed chain of thought (COT) for solving it:
{example_cot}

Your task:
1. Read the question and the detailed COT.
2. Extract only the essential steps needed to solve the question (the minimal reasoning path).
3. Omit any irrelevant or repetitive details, side explorations, or speculation.
4. Present the final answer clearly at the end.

Please provide a concise chain of thought that contains only the necessary logic and calculations to solve the question, followed by the final answer.
"""

def generate_comparison_prompt(example_question, current_question):
    return f"""
You are provided with two math problems:

---

**Example Problem:**  
{example_question}

---

**New Problem:**  
{current_question}

---

Your task is to analyze and compare the two problems. Focus on the following:

- Structural similarities or differences  
- Overlapping or contrasting concepts  
- Reasoning patterns likely required to solve each  

**Do not attempt to solve the new problem.**  
This step is only for analytical comparison.

Begin your response with: `**Comparison**`
"""
def construct_teacher_prompt(example_question, example_cot, current_question, comparison, data_type,analyze_type):
    if analyze_type == 'insight':
        return f"""
You are given one example problem and one new problem. A short comparison between them is also provided.

---

**Example Problem:**  
{example_question}

**Example Solution:**  
{example_cot}

---

**New Problem:**  
{current_question}

---

{comparison}

---

**Your Task:**  
Based on the comparison, find useful ideas or strategies in the example solution that can help solve the new problem.

- Only focus on steps that are likely to transfer.  
- Skip details that are not relevant to the new problem.  
- Do **not** try to solve the new problem.  
- Just extract helpful techniques or methods from the example.

---

**Output Format:**  
Start with: `**Transferable Insights:**`  
Then give 2–5 bullet points with clear and short ideas.
"""
    elif analyze_type == 'structure':
        return f"""
You are given one example problem and one new problem. A short comparison between them is also provided.

---

**Example Problem:**  
{example_question}

**Example Solution:**  
{example_cot}

---

**New Problem:**  
{current_question}

---

{comparison}

---

**Your Task:**  
Look at how the example problem is solved.  
Write a **general step-by-step structure** that follows the same method, but can be used for the new problem.

- Focus on the type and order of steps.  
- Keep each step short and clear.  
- Do **not** copy numbers or calculations.  
- Do **not** solve the new problem.

---

**Output Format:**  
Start with: `**Structure:**`  
Then list numbered steps.

"""

    