

def construct_evaluation_prompt(example_question, example_cot, current_question, cot_candidates: list[str]):
    """construct evaluation prompt"""
    cot_prompts = [f"{i+1}. **Path {i+1}**: {cot_candidates[i]}\n\n" for i in range(len(cot_candidates))]
    cot_prompt = "".join(cot_prompts)

    if example_question == None or example_cot == None:
        return f"""You are tasked with evaluating multiple reasoning paths for the same question. Below is the question and proposed reasoning paths. 

**Current Question**: {current_question}  

**Candidate Reasoning Paths**:  
{cot_prompt} 

**Instructions**:  
1. **Evaluate**: For each path, assess its logical soundness, clarity, and correctness.  
2. **Compare**: Highlight strengths/weaknesses of each path relative to others.  
3. **Select**: Choose the **best path** and justify why it outperforms the rest.  
4. **Advise**: Provide 2-3 actionable suggestions to further improve the selected best path (e.g., fill gaps, fix errors, enhance clarity).  

Respond **only** in the following format, filling in the brackets with your response and wrapping within the <response></response> structure:
<response>
## Evaluation Summary: [Concise comparison of all paths]
## Best Path: [Select the best path by number, e.g., 2] 
## Optimization Advice:  
1. [Advice 1]  
2. [Advice 2]  
3. [Advice 3]  
</response>
""" 
    else:
        return f"""You are tasked with evaluating multiple reasoning paths for the same question. Below is the question, an example reference, and proposed reasoning paths. 

**Reference Example**:  
- Example Question: {example_question}  
- Example Reasoning: {example_cot}  

**Current Question**: {current_question}  

**Candidate Reasoning Paths**:  
{cot_prompt} 

**Instructions**:  
1. **Evaluate**: For each path, assess its logical soundness, alignment with the example’s structure, clarity, and correctness.  
2. **Compare**: Highlight strengths/weaknesses of each path relative to others.  
3. **Select**: Choose the **best path** and justify why it outperforms the rest.  
4. **Advise**: Provide 2-3 actionable suggestions to further improve the selected best path (e.g., fill gaps, fix errors, enhance clarity).  

Respond **only** in the following format, filling in the brackets with your response and wrapping within the <response></response> structure:
<response>
## Evaluation Summary: [Concise comparison of all paths]
## Best Path: [Select the best path by number, e.g., 2] 
## Optimization Advice:  
1. [Weakness 1: Specific issue in the reasoning]  
2. [Weakness 2: Missing step compared to the example]  
3. [Weakness 3: Ambiguous or illogical statement]  
</response>
"""

def construct_refinement_prompt(example_question, example_cot, current_question, best_cot, advice):
    """construct refinement prompt"""

    if example_question == None or example_cot == None:
        return f"""You are tasked with refining the following reasoning path using the feedback provided. Ensure the optimized version:  
1. Maintains logical rigor and clarity.  
2. Addresses all feedback points.
3. Uses the <response>...</response> structure.

**Current Question**: {current_question} 

**Current Reasoning**:
<response>  
{best_cot}
</response> 

**Feedback Received**:  
{advice}

Respond **only** in the following format, filling in the brackets with your response and wrapping within the <response></response> structure:
<response>
## Optimized reasoning: [Revise the reasoning here, explicitly incorporating feedback]
</response>
"""
    else:
        return f"""You are tasked with refining the following reasoning path using the feedback provided. Ensure the optimized version:
1. Maintains logical rigor and clarity.  
2. Addresses all feedback points.  
3. Aligns with the structure of the reference example.
4. Uses the <response>...</response> structure.

**Reference Example**:  
- Example Question: {example_question}  
- Example Reasoning: {example_cot}  

**Current Question**: {current_question} 

**Initial Best Path**:  
{best_cot}  

**Feedback Received**:  
{advice}

Respond **only** in the following format, filling in the brackets with your response and wrapping within the <response></response> structure:
<response>
## Optimized reasoning: [Revise the reasoning here, explicitly incorporating feedback]
</response>
"""

def construct_refinement_prompt_backup(example_question, example_cot, current_question, best_cot, advice):
    """construct refinement prompt"""

    if example_question == None or example_cot == None:
        return f"""You are tasked with refining the following reasoning path using the feedback provided. Ensure the optimized version:  
1. Maintains logical rigor and clarity.  
2. Addresses all feedback points.
3. Uses the <response>...</response> structure.

**Current Question**: {current_question} 

**Current Reasoning**:
<response>  
{best_cot}
</response> 

**Feedback Received**:  
{advice}

Respond **only** in the following format, filling in the brackets with your response and wrapping within the <response></response> structure:
<response>
## Optimized reasoning: [Revise the reasoning here, explicitly incorporating feedback]
</response>
"""
    else:
        return f"""You are tasked with refining the following reasoning path using the feedback provided. Ensure the optimized version:
1. Maintains logical rigor and clarity.  
2. Addresses all feedback points.  
3. Aligns with the structure of the reference example.
4. Uses the <response>...</response> structure.

**Reference Example**:  
- Example Question: {example_question}  
- Example Reasoning: {example_cot}  

**Current Question**: {current_question} 

**Initial Best Path**:  
{best_cot}  

**Feedback Received**:  
{advice}

Respond **only** in the following format, filling in the brackets with your response and wrapping within the <response></response> structure:
<response>
## Optimized reasoning: [Revise the reasoning here, explicitly incorporating feedback]
</response>
"""