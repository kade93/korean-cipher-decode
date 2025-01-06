import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def load_lines_from_file(file_path):
    """
    주어진 파일의 각 줄을 리스트로 반환 (개행 제거, 공백인 줄 무시).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

# def build_faiss_index(documents, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
def build_faiss_index(documents, model_name="BM-K/KoSimCSE-roberta-multitask"):
    # model_name = "BM-K/KoSimCSE-roberta-multitask"
    # embed_model = SentenceTransformer(model_name)
    """
    documents: list of strings
    model_name: SentenceTransformer 모델 이름
    returns: (index, embeddings, embed_model)
    """
    embed_model = SentenceTransformer(model_name)
    embeddings = embed_model.encode(documents, show_progress_bar=False)
    embeddings = embeddings.astype('float32')  # FAISS requires float32

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, embeddings, embed_model

def search_similar_titles(query, index, documents, embed_model, top_k=3):
    """
    query: 검색할 문장
    index: FAISS 인덱스
    documents: 원본 제목들
    embed_model: SentenceTransformer 등
    top_k: 몇 개의 결과를 볼지
    """
    q_emb = embed_model.encode([query], show_progress_bar=False).astype('float32')
    distances, indices = index.search(q_emb, top_k)  # shape: (1, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        title = documents[idx]
        dist = distances[0][rank]
        results.append((title, dist))
    return results

def main():
    # 1) ./documents/icopw.txt에서 “제목 리스트” 불러오기
    titles_file = "./documents/icopw.txt"
    movie_titles = load_lines_from_file(titles_file)

    # 2) FAISS 인덱스 빌드
    index, embeddings, embed_model = build_faiss_index(movie_titles)

    # 3) ./inputs/test.txt에서 “사용자 쿼리” 불러오기
    test_file = "./inputs/test.txt"
    test_queries = load_lines_from_file(test_file)

    # 4) 각 쿼리에 대해 Top-3 검색
    for i, query in enumerate(test_queries):
        results = search_similar_titles(query, index, movie_titles, embed_model, top_k=3)

        print(f"\n=== Input #{i+1} ===")
        print(f"Query: {query}")
        print("Top-3 결과:")
        for rank, (title, dist) in enumerate(results, start=1):
            print(f"{rank}. {title} (distance: {dist:.4f})")

if __name__ == "__main__":
    main()