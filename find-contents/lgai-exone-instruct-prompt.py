import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 파일에서 제목 목록 불러오기
def load_titles(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        titles = [line.strip() for line in file.readlines()]
    return titles

# EXAONE 모델을 이용한 검색 함수
def search_with_exaone(model, tokenizer, titles, query):
    # 프롬프트 생성
    prompt = (
        "아래의 제목 목록에서 주어진 문장과 가장 관련성이 높은 제목을 찾아주세요.\n"
        "제목 목록:\n"
        + "\n".join([f"{i+1}. {title}" for i, title in enumerate(titles)])
        + f"\n\n질문: {query}\n"
        "가장 관련성이 높은 제목을 한 문장으로 답해주세요."
    )
    
    # 입력 토큰화
    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    
    # 모델 추론
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # 출력 디코딩
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 가장 관련된 제목 추출
    answer = decoded_output.split("\n")[-1].strip()
    return answer

# 메인 실행 함수
def main():
    # EXAONE 모델과 토크나이저 로드
    model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 제목 파일 경로
    file_path = "documents/icopw.txt"
    
    # 제목 목록 불러오기
    titles = load_titles(file_path)
    
    # 검색 쿼리
    query = "2019년 12월 초 긴급 조치 (이시언 왕지혜) "
    
    # 검색 수행
    result = search_with_exaone(model, tokenizer, titles, query)
    
    # 결과 출력
    print(f"검색 결과: {result}")

if __name__ == "__main__":
    main()