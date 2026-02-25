from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from markdownify import markdownify
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os 
import uuid
import re 
import torch

qdrant_client = QdrantClient(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "qdrant_data"))



model = SentenceTransformer("google/embeddinggemma-300m")

documents_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Output")

def chunk_text(text, max_chars=1000, overlap=300):
    paragraphs = re.split(r'\n{2,}', text)
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) > max_chars and current:
            chunks.append(current.strip())
            current = current[-overlap:] + "\n\n" + para
        else:
            current += "\n\n" + para
    if current.strip():
        chunks.append(current.strip())
    return chunks

def embed_documents(documents_path):
    if not qdrant_client.collection_exists("personal_data"):
        qdrant_client.create_collection(
            collection_name="personal_data",
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
    for document in os.listdir(documents_path):
        for page in os.listdir(os.path.join(documents_path, document)):
            try: 
                result_path = os.path.join(documents_path, document, page, "result.mmd")
                with open(result_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    markdown_content = markdownify(content)
                    chunks = chunk_text(markdown_content)

                    for i, chunk in enumerate(chunks):
                        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document}/{page}_{i}"))
                        embedding = model.encode(f"task: search result | query: {chunk}").tolist()
                        point = PointStruct(id=point_id, vector=embedding, payload={"content": chunk})
                        qdrant_client.upsert(collection_name="personal_data", points=[point])

            except Exception as e:
                print(f"Error processing {page}. Error: {e}")

def query_similar_documents(query, top_k=3):
    # EmbeddingGemma uses task prompts for better retrieval
    query_with_prompt = f"task: search result | query: {query}"
    query_embedding = model.encode(query_with_prompt).tolist()
    search_result = qdrant_client.query_points(
        collection_name="personal_data",
        query=query_embedding,
        limit=top_k
    )

    return search_result.points

def free_embedding_model():
    global model
    del model
    model = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def llm_call(query, context): 
    # Unload embedding model and free memory before loading LLM
    free_embedding_model()

    tokenizer = AutoTokenizer.from_pretrained(
        'Nanbeige/Nanbeige4.1-3B',
        use_fast=False,
        trust_remote_code=True
    )
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Set explicit max_memory for device_map

    model = AutoModelForCausalLM.from_pretrained(
        'Nanbeige/Nanbeige4.1-3B',
        quantization_config=quant_config,
        device_map={'': 0},  # Use GPU 0
        trust_remote_code=True,
    )

    instruction = (
        "You are a highly concise AI assistant. Answer the user's question STRICTLY based on the provided context. "
        "Do not include unnecessary explanations. "
        "Mark the very end of your final answer with <END>."
    )

    messages = [
        {'role': 'system', 'content': instruction},
        {'role': 'user', 'content': f'Question: {query}\n\nContext:\n{context}. '}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    prompt += "<think>\n</think>\n"

    input_ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids
    output_ids = model.generate(input_ids.to('cuda'), eos_token_id=166101, max_new_tokens=2048)
    raw_resp = tokenizer.decode(output_ids[0][len(input_ids[0]):], skip_special_tokens=True)
    if "</think>" in raw_resp:
        clean_resp = raw_resp.split("</think>")[1].strip()
    else:
        clean_resp = raw_resp.strip()
        
    print(f"--- Clean Final Answer ---\n{clean_resp}")


    

if __name__ == "__main__":
    if not qdrant_client.collection_exists("personal_data"):
        embed_documents(documents_path)
    question_user = "What are the best practices for my breakfast?"
    results = query_similar_documents(question_user)
    context = ""
    for result in results:
        content = result.payload['content']
        print(f"Retrieved score for point ID {result.id}:\n{result.score}\n")
        context += content + "\n"


    
    llm_call(question_user, context)

    qdrant_client.close()
