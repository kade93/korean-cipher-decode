from donut import DonutModel
from transformers import DonutProcessor

# 모델과 프로세서 로드
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-docvqa")
model = DonutModel.from_pretrained(
    "naver-clova-ix/donut-base-finetuned-docvqa",
    ignore_mismatched_sizes=True  # 크기 불일치를 무시하고 로드
)

# 모델을 평가 모드로 설정
model.eval()

# 테스트용 입력 데이터 (예: 이미지 처리)
from PIL import Image
image = Image.open("webhard_sample.png")  # 테스트 이미지 경로

# 이미지 전처리
pixel_values = processor(image, return_tensors="pt").pixel_values

# 모델 추론
outputs = model.generate(pixel_values)

# 결과 디코딩
result = processor.decode(outputs[0])
print("추출된 텍스트:", result)