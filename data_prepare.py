from datasets import load_dataset
import json
import os
import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer



def parse_question(example, data_name=None):
    question = ""
    for key in ["question", "problem", "Question", "input"]:
        if key in example:
            question = example[key]
            break
    
    return question.strip()
def get_repo(data_name):
    if data_name=='math_500':
        return "HuggingFaceH4/MATH-500"
    elif data_name=='aime24':
        return "HuggingFaceH4/aime_2024"
### This code works for Openthought metadata only
def data_prepare(data_type,tokenizer,teacher_name, external_dataset_name, teacher_load_dir, teacher_cot_path, random_t= False, empty_t= False):
    # Step 1: Load reference dataset from GeneralThought
    ds = load_dataset(teacher_name, "metadata", split="train")
    reference_data = [item for item in ds if item.get("domain") == data_type ]
    reference_questions = [item.get("problem", "") for item in reference_data]
    reference_cots = [item.get("deepseek_reasoning", "") for item in reference_data]
    # reference_questions = []
    # reference_cots = []
    # # Step 1.5: Filter those questions and cots which are too long
    # MAX_COT_TOKENS = 15000
    # for item in reference_data:
    #     question = item.get("problem", "")
    #     cot = item.get("deepseek_reasoning", "")
    #     cot_tokens = tokenizer(cot, add_special_tokens=False)["input_ids"]
    #     if len(cot_tokens) <= MAX_COT_TOKENS:
    #         reference_questions.append(question)
    #         reference_cots.append(cot)
    print("Filter over")
    # Step 2: Load external benchmark test questions (e.g., aime, math)
    if external_dataset_name == 'aime24':
        train_type = 'train'
    elif external_dataset_name == 'math_500':
        train_type = 'test'
    external_ds  = load_dataset(get_repo(external_dataset_name), split=train_type)
    test_questions = [{"question": parse_question(example, external_dataset_name)} for example in external_ds]
    

    ### Alternatively, if you want to do a verification test on reference dataset, simply separate your own test data from reference data


    # Step 3: Save test questions and matched CoT folder
    os.makedirs(f'{teacher_cot_path}', exist_ok=True)

    # Step 4: Build FAISS index with TF-IDF embeddings

    # vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
    # reference_vectors = vectorizer.fit_transform(reference_questions)
    # reference_vectors = normalize(reference_vectors).toarray().astype(np.float32)
    embed_model = SentenceTransformer("intfloat/multilingual-e5-large") 
    reference_vectors = embed_model.encode(reference_questions, normalize_embeddings=True)
    # test_vectors = embed_model.encode(test_question_texts, normalize_embeddings=True)
    dim = reference_vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(reference_vectors)

    # Step 5: Encode test questions and search
    test_question_texts = [q["question"] for q in test_questions]
    test_vectors = embed_model.encode(test_question_texts, normalize_embeddings=True)
    # test_vectors = vectorizer.transform(test_question_texts)
    # test_vectors = normalize(test_vectors).toarray().astype(np.float32)
    if not random_t:
        _, indices = index.search(test_vectors, 1)  # top-1 match
    else:
        print('randomly choose teacher')
        indices = np.random.randint(0, len(reference_questions), size=(len(test_question_texts), 1))
    if not empty_t:
        matched_cots = [reference_cots[i[0]] for i in indices]
    else:
        matched_cots = [""] * len(test_questions)
    matched_reference_questions = [reference_questions[i[0]] for i in indices]

    # Step 6: Save matched COTs along with test questions and GT reasoning/solution
    with open(f"{teacher_cot_path}/test_questions_with_matched_cots.jsonl", "w", encoding="utf-8") as f:
        for q, idx, ref_q, cot in zip(test_questions, indices, matched_reference_questions, matched_cots):
            json.dump({
                "test_question": q["question"],
                "matched_reference_index": int(idx[0]),
                "matched_reference_question": ref_q,
                "matched_cot": cot
            }, f, ensure_ascii=False)
            f.write("\n")

    print("✅ All outputs saved in JSONL format under"+teacher_cot_path+"folder.")