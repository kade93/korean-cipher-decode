import re
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from accelerate import Accelerator

def simple_preprocess(text: str) -> str:
    """
    간단한 규칙 기반 전처리 함수.
    1. 'O'를 숫자 '0'으로 치환
    2. 불필요한 공백 제거
    """
    text = re.sub(r'O', '0', text)
    text = re.sub(r'\s+', '', text)
    return text

# 1. Accelerator 초기화
accelerator = Accelerator()

# 2. 모델 및 토크나이저 로드
#    - KETI-AIR/ke-t5-small: 한국어에 특화된 T5 모델 실패
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 3. 모델 & 토크나이저를 Accelerator로 준비
model, tokenizer = accelerator.prepare(model, tokenizer)

# 4. 변형된 문장 정의
raw_text = "2O19.12월 초 긴 급 ( ㅇㅣㅅㅣ언 왕 ㅈ ㅣ ㅎ ㅖ ) [ 죽 였 ㄷㅏ ㅇㅏㄴㅐ를 ]"

# 5. 간단한 전처리 수행
preprocessed_text = simple_preprocess(raw_text)

# 6. 모델에게 복원 태스크 지시
input_text = (
    "다음 문장을 올바른 한국어 문장으로 복원하세요:\n"
    f"입력: {preprocessed_text}\n"
    "출력:"
)

input_ids = tokenizer(input_text, return_tensors="pt").input_ids
input_length = input_ids.shape[1]  # (배치, 시퀀스 길이 중 시퀀스 길이)

outputs = model.generate(
    input_ids.to(device),
    max_length=input_length + 20,  # 입력 토큰 길이 + 여유분
    min_length=input_length - 10,  # 혹은 적정 수준으로 설정
    num_beams=5,
    early_stopping=True
)

restored_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("복원된 텍스트:", restored_text)
# 9. 결과 디코딩
restored_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("복원된 텍스트:", restored_text)