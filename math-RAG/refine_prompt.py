
def construct_selection_prompt(current_question, cot_candidates: list[str]):
    cot_prompts = [f"{i+1}. ## Path {i+1}:\n{cot_candidates[i]}\n\n" for i in range(len(cot_candidates))]
    cot_prompt = "".join(cot_prompts)
    prompt = f"""Pick the **best** reasoning path for the question below. 

**Current Question:**
{current_question}

**Candidate Reasoning Paths:**
{cot_prompt}

---

**Your task:** 
1. You need to choose the best path of reasoning based on your own assessment (e.g., logical soundness, clarity, and correctness).
2. In your response, enclose the Arabic number corresponding to the best path in double curly brackets, e.g., {{{{2}}}}.
"""

    return prompt

def construct_suggestion_prompt(current_question, best_path):
    prompt = f"""Identify **ONE specific weakness** in the reasoning path below.  

**Current Question:**
{current_question}

**Current Best Reasoning Path:**
{best_path}

---

**Your task:**
1. You should try to find **ONE** of these issues:
   - Arithmetic or algebraic mistakes (e.g., incorrect simplifications or incorrect application of operations).
   - Misapplied theorems or incorrect assumptions (e.g., an unjustified jump to a conclusion).
   - Logical inconsistencies (e.g., a contradiction in the reasoning).
   - Misinterpretation of the problem statement or prior steps.
2. In your response, enclose the issue in double curly brackets, e.g., {{{{issue:}}}}.
3. If you think the entire reasoning path is completely correct, you need to output {{{{No obvious issues}}}}.
"""

    return prompt

def construct_refinement_prompt(current_question, best_path, issue):
    prompt = f"""Refine this reasoning path to fix given issue.

**Current Question:**
{current_question}

**Current Best Reasoning Path:**
{best_path}

**Issue to Fix:** 
{issue}  

---

**Your task:**  
1. You should fix the specific issue mentioned above. 
2. You should recheck logic and answer correctness.
"""
# 3. In your response, enclose the refined path in double curly brackets, e.g., {{{{refined:}}}}.
    return prompt