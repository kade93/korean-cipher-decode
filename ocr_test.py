import unicodedata
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer

# 모델 및 프로세서 로드
processor = TrOCRProcessor.from_pretrained("ddobokki/ko-trocr")
model = VisionEncoderDecoderModel.from_pretrained("ddobokki/ko-trocr")
tokenizer = AutoTokenizer.from_pretrained("ddobokki/ko-trocr")

# 이미지 로드 (로컬 파일 경로 사용)
image_path = "webhard_sample_cut.png"
img = Image.open(image_path)

# 이미지 전처리
if img.mode != "RGB":
    img = img.convert("RGB")

# 2. 이미지 전처리: 크기 조정
target_size = (384, 384)  # TrOCR 모델이 사용하는 기본 입력 크기
img = img.resize(target_size)

# 3. 모델 입력 처리
pixel_values = processor(img, return_tensors="pt").pixel_values
# 텍스트 생성
generated_ids = model.generate(pixel_values, max_length=64)
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 텍스트 정규화 (NFC 형태로)
generated_text = unicodedata.normalize("NFC", generated_text)

print("OCR 결과:", generated_text)