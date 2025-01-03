from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
from accelerate import Accelerator

# 1. Accelerator 초기화
accelerator = Accelerator()

# 2. 데이터 준비
data = {
    "input_text": [
        "2O19.12월 초 긴 급 ( ㅇㅣㅅㅣ언 왕 ㅈ ㅣ ㅎ ㅖ ) [ 죽 였 ㄷㅏ ㅇㅏㄴㅐ를 ]",
        "2O21년 7월 중 신 (ㅇㅓ ㅂㅜ ㅅㅣ ㄴ ㅅㅣ ㅎㅐ) - ㅅ ㅓ ㄹ. 계. 공 포",
    ],
    "output_text": [
        "2019년 12월 초긴급 (이시언 왕지혜) [죽였다 아내를]",
        "2021년 7월 중신 (어부 신시해) 설계 공포",
    ]
}
eval_data = {
    "input_text": [
        "②O②O.01월 - 긴장감 최고치 - [ - 블.랙.앤.드.블.루 - ] 숨을곳없다",
    ],
    "output_text": [
        "2020년 1월 긴장감 최고치 [블랙 앤드 블루] 숨을 곳 없다",
    ]
}

# 데이터셋 생성
train_dataset = Dataset.from_dict(data)
eval_dataset = Dataset.from_dict(eval_data)

# 3. 모델 및 토크나이저 로드
model_name = "microsoft/GODEL-v1_1-large-seq2seq"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 4. Accelerator로 모델 준비
model = accelerator.prepare(model)

# 5. 데이터 전처리 함수
def preprocess_function(examples):
    inputs = examples["input_text"]
    outputs = examples["output_text"]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(outputs, max_length=128, truncation=True, padding="max_length").input_ids
    model_inputs["labels"] = labels
    return model_inputs

# 데이터 전처리 적용
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# 6. 학습 함수 정의
from torch.utils.data import DataLoader
import torch
from torch.optim import AdamW

# 학습 함수
def train_epoch(model, tokenizer, train_dataset, eval_dataset, accelerator):
    # DataLoader 생성 (배치 크기 지정)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=8)

    # 옵티마이저 설정
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Accelerator로 준비
    train_dataloader, eval_dataloader, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, optimizer
    )

    # 학습 루프
    model.train()
    for batch in train_dataloader:
        # 배치 데이터를 텐서로 변환
        input_ids = torch.stack(batch["input_ids"]).to(accelerator.device)
        attention_mask = torch.stack(batch["attention_mask"]).to(accelerator.device)
        labels = torch.stack(batch["labels"]).to(accelerator.device)

        optimizer.zero_grad()

        # 모델에 입력
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()

    # 평가 루프
    model.eval()
    total_eval_loss = 0
    for batch in eval_dataloader:
        with torch.no_grad():
            # 배치 데이터를 텐서로 변환
            input_ids = torch.stack(batch["input_ids"]).to(accelerator.device)
            attention_mask = torch.stack(batch["attention_mask"]).to(accelerator.device)
            labels = torch.stack(batch["labels"]).to(accelerator.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        total_eval_loss += outputs.loss.item()

    print(f"평가 손실: {total_eval_loss / len(eval_dataloader)}")

train_epoch(model, tokenizer, tokenized_train_dataset, tokenized_eval_dataset, accelerator)

# 7. 학습 후 테스트
def generate(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}  # 입력 텐서를 Accelerator 디바이스로 이동
    outputs = model.generate(
        inputs["input_ids"],
        max_length=256,  # 최대 길이를 늘려 응답 유도
        num_beams=5,     # 빔 탐색을 확장
        no_repeat_ngram_size=2,  # 반복 방지 설정
        early_stopping=False     # 조기 종료 비활성화
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 테스트 입력
test_input = "O3월.판타지 [ 돌아온 마법사 ] 1O8Op.한글자막"
# -> O -> 0 영어 숫자로 치환 (실패)
result = generate(test_input)
print(f"복원된 텍스트: {result}")