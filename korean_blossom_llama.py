import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator

# Accelerator 초기화
accelerator = Accelerator()

# 모델 로드
model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 모델 로드 후 pad_token_id 설정
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)
model.config.pad_token_id = tokenizer.pad_token_id  # pad_token_id 설정

# Accelerator를 사용해 모델 준비
model = accelerator.prepare(model)

model.eval()

# 프롬프트 및 입력 텍스트
PROMPT = '''You are a highly skilled Korean text restoration assistant. 
Your task is to restore fragmented or distorted Korean text into its correct and original form.
Follow these general rules to handle various input formats:

### General Restoration Rules:
1. Normalize years and months to their standard Korean format (e.g., "2O19.12월" → "2019년 12월").
2. Combine fragmented or separated Korean characters into meaningful and grammatically correct sentences.
3. Retain any important context such as genres, subtitles, or other metadata, integrating them logically.
4. Remove unnecessary symbols, English letters, or numbers unless they are essential to the meaning of the text.
5. Ensure the output is clean, well-structured, and grammatically correct.

### Examples (Different Types of Inputs):
- Input: "2O19.12월 초 긴 급 ( ㅇㅣㅅㅣ언 왕 ㅈ ㅣ ㅎ ㅖ ) [ 죽 였 ㄷㅏ ㅇㅏㄴㅐ를 ]"
  Output: "2019년 12월 초긴급(이시언 왕지혜)[죽였다 아내를]"

- Input: "②O②O.01월 - 긴장감 최고치 - [ - 블.랙.앤.드.블.루 - ] 숨을곳없다"
  Output: "2020년 1월 긴장감 최고치 [블랙 앤드 블루] 숨을 곳 없다"

- Input: "O3월.판타지 [ 돌아온 마법사 ] 1O8Op.한글자막"
  Output: "3월 판타지 [돌아온 마법사] 1080P 한글자막"

Now, restore the following text:
'''

# 입력 텍스트
input_text = "[코미디] 차인표 떠 따 [ᄎr 인 표] 1080P Dolby 5.1 자막포함"

# 메시지를 단일 문자열로 구성
input_message = f"{PROMPT} Input: {input_text}"

# 입력 데이터 생성 시 attention_mask 포함
input_ids = tokenizer(
    input_message,
    return_tensors="pt",
    padding=True,  # 패딩 활성화
    truncation=True  # 길이 초과 시 잘라냄
).to(accelerator.device)

# 종료 토큰 설정
terminators = [
    tokenizer.eos_token_id,
]

# 텍스트 생성
with torch.no_grad():
    outputs = model.generate(
        input_ids["input_ids"],  # input_ids를 명시적으로 지정
        attention_mask=input_ids["attention_mask"],  # attention_mask 추가
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )

# 결과 디코딩
restored_text = tokenizer.decode(outputs[0][input_ids["input_ids"].shape[-1]:], skip_special_tokens=True)

# 결과 출력
print(f"원본 텍스트: {input_text}")
print(f"복원된 텍스트: {restored_text}")