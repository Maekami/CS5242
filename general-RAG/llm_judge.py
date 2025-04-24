from openai import OpenAI
from openai import AsyncOpenAI
import requests
import asyncio

def check_balance(api_key):
    '''查看账户余额'''
    url = "https://api.deepseek.com/user/balance"

    payload={}
    headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {api_key}'
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    print(response.text)

# TODO: 可以进一步修改
def construct_prompt(question, generated_answer, gold_answer):
    '''构造prompt'''
    prompt = f'''Act as an impartial judge to determine semantic equivalence between two answers.

Question:
{question}

Generated answer:
{generated_answer}

Gold (reference) answer:
{gold_answer}

Instruction:
Determine whether the “Generated answer” and the “Gold answer” convey the same meaning, even if they use different wording.  
- If they are semantically equivalent, output exactly:
  True  
- Otherwise, output exactly:
  False  

Your response should be only the single word `True` or `False`, with no additional commentary.
'''
    return prompt

def construct_prompt_batch(question, generated_answer, gold_answer):
    '''批量构造prompt'''
    return [construct_prompt(item1, item2, item3) for item1, item2, item3 in zip(question, generated_answer, gold_answer)]

def check_answer(question, generated_answer, gold_answer, api_key, max_tokens=20, temperature=1, top_p=1):
    '''检查回答与答案是否匹配'''
    # for backward compatibility, you can still use `https://api.deepseek.com/v1` as `base_url`.
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompt = construct_prompt(question, generated_answer, gold_answer)

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are an AI judge of answer quality."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=False
    )

    print(response.choices[0].message.content)

def check_answer_batch(question, generated_answer, gold_answer, api_key, max_tokens=20, temperature=1, top_p=1):
    '''通过使用异步处理来实现批量处理'''
    
    async def call_deepseek(client, prompt, max_tokens, temperature, top_p):
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "You are an AI judge of answer quality."},{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False,
        )
        return response.choices[0].message.content

    async def process(prompts, api_key, max_tokens, temperature, top_p):
        client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        # 并发发送所有请求
        tasks = [call_deepseek(client, prompt, max_tokens, temperature, top_p) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
        
        # TODO: 修改处理返回的回答逻辑
        for prompt, response in zip(prompts, responses):
            print(f"问题: {prompt}\n回答: {response}\n")

    prompts = construct_prompt_batch(question, generated_answer, gold_answer)
    asyncio.run(process(prompts, api_key, max_tokens, temperature, top_p))


######测试样例######
question = [
    "How do covalent bonds form between atoms of the same molecule, such as carbon, given that there is no difference in electronegativity? Provide a detailed explanation of the quantum mechanical principles involved and the role of electron configuration in facilitating bond formation."
] * 6
generated_answer = [
    "Covalent bonds occur because atoms transfer electrons until they achieve noble gas configurations, resulting in oppositely charged ions that attract each other strongly.",
    "Covalent bonds form by the quantum mechanical overlap of atomic orbitals into a lower‑energy bonding orbital, allowing two electrons to be shared between nuclei, which yields a more stable configuration than separate, unpaired electrons in incomplete shells.",
    "Atoms of the same element bond covalently because thermal vibrations cause electrons to randomly jump between atoms, leading to fluctuating charge distributions that transiently hold atoms together.",
    "When two carbon atoms approach, their half‑filled sp³ orbitals overlap and promote electrons to create a shared bonding orbital, enabling electrons to be attracted by both nuclei and lowering the system’s energy relative to isolated atoms.",
    "Covalent bonds form due to increasing entropy as electrons become more disordered between atoms, providing a net thermodynamic driving force for bond formation even without enthalpic stabilization.",
    "Through promotion of electrons into degenerate orbitals and hybridization, atoms develop half‑filled shells whose orbitals overlap to form a bonding molecular orbital that attracts electrons to both nuclei, increasing binding energy over unbonded configurations."
] # 错对错对错对
gold_answer = [
    "Covalent bonds form between atoms of the same molecule due to the attraction of electrons to both nuclei, facilitated by the promotion of electrons and the presence of incomplete shells, resulting in a higher binding energy than the configuration without shared electrons."
] * 6
api_key = "sk-65fedd72d2224ee29292ee07bc6d4c66"

check_answer_batch(question, generated_answer, gold_answer, api_key, max_tokens=20, temperature=0, top_p=1)
# check_answer([],[],[],api_key, max_tokens=20, temperature=0.7, top_p=0.95)
######测试样例######