import re
import torch

# 🤗 Transformers
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast
from accelerate import Accelerator
from transformers import file_utils

# Transformers 캐시 디렉토리 확인 (Check Transformers cache directory)
cache_dir = file_utils.default_cache_path
print(f"Transformers 캐시 경로 (Transformers cache path): {cache_dir}")

###############################################################################
# 0) 한글 자모 리스트 (Korean Jamo Lists for composition)
###############################################################################
CHOSUNG_LIST = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNGSEONG_LIST = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
JONGSEONG_LIST = [""] + list("ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

###############################################################################
# 0-1) 변형 기호 -> 자모 매핑 (Variant symbols to Jamo)
###############################################################################
VARIANT_JAMO_MAP = {
    'ㅇ': ['o','O','0'],
    'ㄹ': ['l','L','|','I'],
    'ㄱ': ['g','G'],
    'ㅅ': ['s','S'],
    'ㅈ': ['j','J','z','Z'],
    'ㅎ': ['h','H'],
    # 모음 예시 (Add more if needed)
    'ㅣ': ['i','I','|'],
    'ㅏ': ['a','A'],
    'ㅓ': ['eo','EO'],
    'ㅗ': ['O','o'],  # 주의: ㅇ(‘O’)과 혼동될 수 있으니 상황에 맞게 조정
}

def replace_variant_jamo(text: str) -> str:
    """
    Replace variant symbols with standard Korean jamo based on VARIANT_JAMO_MAP.
    """
    for jamo, variants in VARIANT_JAMO_MAP.items():
        for v in variants:
            text = text.replace(v, jamo)
    return text

###############################################################################
# 0-2) 자모 합성 (Compose Choseong, Jungseong, and Jongsung into Hangul)
###############################################################################
def compose_jamo(choseong: str, jungseong: str, jongsung: str = "") -> str:
    """
    Compose Korean syllable from choseong, jungseong, and optional jongsung.
    """
    if choseong in CHOSUNG_LIST and jungseong in JUNGSEONG_LIST:
        jongsung_index = JONGSEONG_LIST.index(jongsung) if jongsung in JONGSEONG_LIST else 0
        syllable_code = 0xAC00 + (
            CHOSUNG_LIST.index(choseong) * 21 +
            JUNGSEONG_LIST.index(jungseong)
        ) * 28 + jongsung_index
        return chr(syllable_code)
    return choseong + jungseong + jongsung  # if invalid combination

def auto_compose_decomposed(text: str) -> str:
    """
    Scan text and compose valid sequences of (choseong + jungseong + jongsung) into Hangul syllables.
    """
    chars = list(text)
    output = []
    
    i = 0
    length = len(chars)
    while i < length:
        c = chars[i]
        # check if c is choseong
        if c in CHOSUNG_LIST:
            # check next char is jungseong
            if i + 1 < length and chars[i+1] in JUNGSEONG_LIST:
                choseong = c
                jungseong = chars[i+1]
                # check jongsung
                if i + 2 < length and chars[i+2] in JONGSEONG_LIST[1:]:
                    jongsung = chars[i+2]
                    output.append(compose_jamo(choseong, jungseong, jongsung))
                    i += 3
                else:
                    output.append(compose_jamo(choseong, jungseong))
                    i += 2
            else:
                # just a single consonant
                output.append(c)
                i += 1
        else:
            output.append(c)
            i += 1
    
    return "".join(output)

###############################################################################
# 1) 전처리 함수 (Rule-based preprocessing)
###############################################################################
def rule_based_preprocess(text: str) -> str:
    """
    1) replace_variant_jamo: 변형 기호를 표준 자모로 치환
    2) 특수문자/공백 제거
    3) auto_compose_decomposed: 자모를 완성형 한글로 합성
    """
    # 1) 변형 기호를 한글 자모로 치환
    text = replace_variant_jamo(text)
    
    # 2) 불필요한 기호나 공백 제거 (가볍게)
    text = re.sub(r'[,\[\]\(\)\s]+', ' ', text).strip()
    
    # 3) 자모 합성
    text = auto_compose_decomposed(text)
    
    return text

###############################################################################
# 2) KoBART 모델 로드 & Accelerator 초기화
###############################################################################
accelerator = Accelerator()
model_name = "gogamza/kobart-base-v2"
# model_name = "hyunwoongko/kobart"
# model_name = "hyunwoongko/kobart"

tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name, cache_dir=cache_dir)
model = BartForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)

# Prepare with Accelerator
model, tokenizer = accelerator.prepare(model, tokenizer)

###############################################################################
# 3) AI 추론 함수 (Inference) + 프롬프트 설계
###############################################################################
def interpret_content_info(raw_text: str) -> str:
    # 전처리
    pre_text = rule_based_preprocess(raw_text)
    
    # 프롬프트 설계 (Prompt Engineering)
    prompt = (
        "다음 문장에서 포함된 정보를 바탕으로 내용을 해석하세요:\n"
        "- 날짜: 콘텐츠가 언제 올라왔는지.\n"
        "- 배우: 등장하는 배우의 이름.\n"
        "- 제목: 콘텐츠의 제목.\n\n"
        "예:\n"
        "입력: 2O19.12월 초 긴 급 ( ㅇㅣㅅㅣ언 왕 ㅈ ㅣ ㅎ ㅖ ) [ 죽 였 ㄷㅏ ㅇㅏㄴㅐ를 ]\n"
        "출력: 이 콘텐츠는 2019년 12월 초에 올라온 내용이며, 배우는 이시언과 왕지혜입니다. 제목은 \"죽였다 아내를\"입니다.\n\n"
        f"입력: {pre_text}\n"
        "출력:"
    )
    
    # 토큰화
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(accelerator.device)
    
    # max_length, min_length 설정
    input_length = input_ids.shape[1]
    max_length = input_length + 50
    min_length = max(1, input_length - 20)
    
    print(f"응답 최소길이 (min_length): {min_length}, 최대길이 (max_length): {max_length}")

    # 모델 추론
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        min_length=min_length,
        num_beams=5,
        early_stopping=True,
        repetition_penalty=1.2,    
        no_repeat_ngram_size=2
    )
    
    # 디코딩
    result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result_text

###############################################################################
# 4) 테스트 (Test)
###############################################################################
if __name__ == "__main__":
    # 테스트 문장 1
    test_1 = "2O19.12월 초 긴 급 ( ㅇㅣㅅㅣ언 왕 ㅈ ㅣ ㅎ ㅖ ) [ 죽 였 ㄷㅏ ㅇㅏㄴㅐ를 ]"
    result_1 = interpret_content_info(test_1)
    print("입력 (Input):", test_1)
    print("전처리 후 (After Preprocess):", rule_based_preprocess(test_1))
    print("모델 출력 (Model Output):", result_1)

    print("--------------------------------------------------")

    # 테스트 문장 2
    test_2 = "20o1.0lㅣ.1I S.F (ㄱㅏ.ㅇ ㅏ. ㅂㅜ ㅅㅣ 오.) [--- ㅅㅣ . 공 . 간 . ㅇㅣ . 동 ---]"
    result_2 = interpret_content_info(test_2)
    print("입력 (Input):", test_2)
    print("전처리 후 (After Preprocess):", rule_based_preprocess(test_2))
    print("모델 출력 (Model Output):", result_2)