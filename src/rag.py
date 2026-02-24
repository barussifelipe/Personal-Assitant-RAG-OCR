from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from markdownify import markdownify
from sentence_transformers import SentenceTransformer
import os 
import uuid
import re 


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
    for page in os.listdir(documents_path):
        try: 
            result_path = os.path.join(documents_path, page, "result.mmd")
            with open(result_path, 'r', encoding='utf-8') as file:
                content = file.read()
                markdown_content = markdownify(content)
                chunks = chunk_text(markdown_content)
                for i, chunk in enumerate(chunks):
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, page + f"_{i}"))
                    embedding = model.encode(f"task: search result | query: {chunk}").tolist()
                    point = PointStruct(id=point_id, vector=embedding, payload={"content": chunk})
                    qdrant_client.upsert(collection_name="personal_data", points=[point])

        except Exception as e:
            print(f"Error processing {page}. Error: {e}")

def query_similar_documents(query, top_k=5):
    # EmbeddingGemma uses task prompts for better retrieval
    query_with_prompt = f"task: search result | query: {query}"
    query_embedding = model.encode(query_with_prompt).tolist()
    search_result = qdrant_client.query_points(
        collection_name="personal_data",
        query=query_embedding,
        limit=top_k
    )
    return search_result.points

if __name__ == "__main__":
    if not qdrant_client.collection_exists("personal_data"):
        embed_documents(documents_path)
    question_user = "Qual foi o QI total?"
    results = query_similar_documents(question_user)
    context = ""
    for result in results:
        content = result.payload['content']
        print(f"Retrieved score for point ID {result.id}:\n{result.score}\n")
        print(f"Content:\n{content}\n{'-'*50}\n")
        context += content + "\n"
    prompt_rag = f"Você é um assistente corporativo útil. Responda à pergunta do usuário usando APENAS o contexto fornecido abaixo. Se a resposta não estiver no contexto, diga que não sabe. \n\n CONTEXTO RECUPERADO: {context}; \n \n PERGUNTA DO USUÁRIO: {question_user}"

    

    qdrant_client.close() 