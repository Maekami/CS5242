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
def construct_answer(example_question, question, comparison, analyze, analyze_type):
    if analyze_type == 'insight':
        return f"""
### General Reasoning Task

Solve the following question step by step, providing clear and logical reasoning throughout. Ensure your final answer is stated clearly at the end.

You are given:
- An **example question** that is similar to the current question.
- A **comparison** between the two questions to identify transferable reasoning strategies.
- Several **potentially helpful insights** based on the example.

Use the example and comparison to guide your reasoning.  
Only apply insights if they are clearly useful to the current question.

---

**Example Question:**  
{example_question}

---

{comparison}

---

{analyze}

---

**Current Question:**  
{question}
"""
    elif analyze_type == 'structure':
        return f"""
### General Reasoning Task

Solve the following question step by step using a clear and logical approach.

You are given:
- An **example question** that is similar to the current question.
- A **comparison** between the two questions to identify structural reasoning patterns.
- A **general step-by-step structure** derived from the example solution.

Use the structure as a flexible guide to reason through the current question.

---

**Example Question:**  
{example_question}

---

{comparison}

---

{analyze}

---

**Current Question:**  
{question}
"""


def extract_useful_prompt(example_question, example_cot):
    return f"""
I have an example reasoning question:
{example_question}

And a detailed explanation for solving it:
{example_cot}

Your task:
1. Read the question and the explanation carefully.
2. Extract only the essential reasoning steps needed to arrive at the final answer.
3. Remove irrelevant details, side notes, or speculative content.
4. Present a clean, concise, and logical reasoning path followed by a clear final answer.

Please provide the minimal and necessary reasoning steps in a coherent format.
"""


def generate_comparison_prompt(example_question, current_question):
    return f"""
You are provided with two reasoning questions:

---

**Example Question:**  
{example_question}

---

**Current Question:**  
{current_question}

---

Your task is to compare the two questions. Focus on the following:

- Similarities or differences in structure  
- Overlapping or distinct reasoning skills required  
- Types of logic or knowledge involved in each

**Do not attempt to solve the current question.**  
This step is only for analytical comparison.

Begin your response with: `**Comparison**`
"""

def construct_teacher_prompt(example_question, example_cot, current_question, comparison, analyze_type):
    if analyze_type == 'insight':
        return f"""
You are given one example question and one new question. A short comparison between them is also provided.

---

**Example Question:**  
{example_question}

**Example Explanation:**  
{example_cot}

---

**New Question:**  
{current_question}

---

{comparison}

---

**Your Task:**  
Identify useful ideas or reasoning strategies in the example explanation that could help solve the new question.

- Focus on transferable reasoning patterns or approaches.  
- Ignore specific details that don't apply.  
- Do **not** solve the new question.  
- Only extract helpful strategies or techniques.

---

**Output Format:**  
Start with: `**Transferable Insights:**`  
Then list 2–5 bullet points with concise ideas.
"""
    elif analyze_type == 'structure':
        return f"""
You are given one example question and one new question. A short comparison between them is also provided.

---

**Example Question:**  
{example_question}

**Example Explanation:**  
{example_cot}

---

**New Question:**  
{current_question}

---

{comparison}

---

**Your Task:**  
Derive a **general reasoning structure** from the example solution that can be applied to the new question.

- Focus on the type and order of reasoning steps.  
- Do not copy specific content or data.  
- Keep the structure concise and abstract.  
- Do **not** solve the new question.

---

**Output Format:**  
Start with: `**Structure:**`  
Then list numbered steps.
"""