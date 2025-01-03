## Korean Cipher Decoding by AI Research 

```python3
poetry shell
poetry install
python3 [model_name]_test.py

```
## Main Goal : Find the model that can restore Korean Cipher ( Example : **Lr는 ㄱㅏ 끔 눈물을 흘린ㄷr... ) into the normal Korean**

**Related Media:  ChatGPT o1**

https://www.ytn.co.kr/_ln/0104_202409140314118230* 

[Korean Cipher with OpenAI o1](https://www.youtube.com/watch?v=eZDmDn6Iq9Y&t=59s)

### Test on AI Services

- Test Prompt

```
2O19.12월 초 긴 급 ( ㅇㅣㅅㅣ언 왕 ㅈ ㅣ ㅎ ㅖ ) [ 죽 였 ㄷㅏ ㅇㅏㄴㅐ를 ]
위 텍스트를 정상적인 한글로 복원해봐 ( Restore above text into normal Korean )
```

### Results

| **Model Name** | **Model Size** | **Model Type** | **Result** | **API**  |
| --- | --- | --- | --- | --- |
| **GPT 4o / o1 / o1-mini** | **70B +** | **Multimodal LLM** | **O** | **O** |
| **Gemini 1.5 / 2.0** | **70B +**  | **Multimodal LLM** | **O** | **O** |
| **Clova AI**  | **204B** | **Multimodal LLM** | **O** | **O** |
|  |  |  |  |  |

## Local Model Test ( Ollama, Hugging Face )

- **Spec : RTX A5000 * 6**

### Test Resources  ( Prompt Engineering)

- Type A

```python
PROMPT = '''You are a helpful AI assistant specializing in Korean text restoration. 
Please restore the following fragmented Korean text into its original form:
당신은 한국어 텍스트 복원에 특화된 유능한 AI 어시스턴트입니다. 
아래 주어진 분리된 한국어 텍스트를 원래의 형태로 복원해주세요.'''
input_text = "2O19.12월 초 긴 급 ( ㅇㅣㅅㅣ언 왕 ㅈ ㅣ ㅎ ㅖ ) [ 죽 였 ㄷㅏ ㅇㅏㄴㅐ를 ]"

```

- Type B

```python
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
```

### Results summary (get model from HuggingFace)

| **Model Name** | **Model Size** | **Model Type** | **Result** |
| --- | --- | --- | --- |
| KETI-AIR/ke-t5-small | ~220M | **Text 2 Text Generation** | ❌ |
| ddobokki/ko-trocr | ~370M | **Image-to-Text** | ❌ |
| gogamza/kobart-base-v2 | ~125M | **Text 2 Text Generation** | ❌ |
| hyunwoongko/kobart | ~125M | **Text 2 Text Generation** | ❌ |
| google/flan-t5-base | ~248M | **Text 2 Text Generation** | ❌ |
| naver-clova-ix/donut-base | ~1.1B | **Image-to-Text** | ❌ |
| MLP-KTLim/llama-3-Korean-Bllossom-8B | ~8.03B | **Text Generation** | ❌ |
| LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct | ~2.4B | Text Generation | 🟢 |

### The Best Result ( LGAI-EXAONE - 2.4B model )

| **원본 텍스트** | **복원된 텍스트**	 | **GPU 메모리 사용량 (MB)** |
| --- | --- | --- |
| 2O19.12월 초 긴 급 ( ㅇㅣㅅㅣ언 왕 ㅈ ㅣ ㅎ ㅖ ) [ 죽 였 ㄷㅏ ㅇㅏㄴㅐ를 ] | 2019년 12월 초 긴급 조치 (이시언 왕지혜)로 인해 아내를 죽였다 | Allocated: 4685.94, Reserved: 4844.00 |
| 2019.02월 S.F 판타지 [— ㅅㅣ . 공 . 간 . ㅇㅣ . 동 —]한글자막 초고화질 1080P | 2019년 2월 SF 판타지 [—시간 이동—] 고화질 1080P 자막 초고화질 | Allocated: 4685.94, Reserved: 4880.00 |
| ②O②O.01월 - 긴장감 최고치 - [ - 블.랙.앤.드.블.루 - ] 숨을곳없다 | 월 긴장감 최고치 [-블랙앤드블루] 숨을 곳 없다To make it more coherent:2월 긴장감 최고 [-블랙앤드블루] 숨 쉴 곳 없어 | Allocated: 4685.94, Reserved: 4880.00 |
- **After inference - Allocated: 4685.94 MB, Reserved: 4880.00 MB**

```python
Before inference - Allocated: 4685.94 MB, Reserved: 4844.00 MB
original text: 2O19.12월 초 긴 급 ( ㅇㅣㅅㅣ언 왕 ㅈ ㅣ ㅎ ㅖ ) [ 죽 였 ㄷㅏ ㅇㅏㄴㅐ를 ]
decoded text: |system|]You are EXAONE model from LG AI Research, a helpful assistant specialized in restoring distorted Korean text.
[|user|]Restore the following text into normal Korean: 2O19.12월초긴급(이시언왕지혜)[죽였다아내를]
**[|assistant|]2019년 12월 초 긴급 조치 (이시언 왕지혜)로 인해 아내를 죽였다.** 

original text: 2019.02월 S.F 판타지 [--- ㅅㅣ . 공 . 간 . ㅇㅣ . 동 ---]한글자막 초고화질 1080P
decoded text: [|system|]You are EXAONE model from LG AI Research, a helpful assistant specialized in restoring distorted Korean text.
[|user|]Restore the following text into normal Korean: 2019.02월S.F판타지[---시.공.간.이.동---]한글자막초고화질1080P
**[|assistant|]2019년 2월 SF 판타지 [---시간 이동---] 고화질 1080P 자막 초고화질**
original text: ②O②O.01월 - 긴장감 최고치 - [ - 블.랙.앤.드.블.루 - ] 숨을곳없다
decoded text: [|system|]You are EXAONE model from LG AI Research, a helpful assistant specialized in restoring distorted Korean text.
[|user|]Restore the following text into normal Korean: 2O2O.01월-긴장감최고치-[-블.랙.앤.드.블.루-]숨을곳없다
[|assistant|]The restored text in normal Korean appears as follows:

**2월 긴장감 최고치 [-블랙앤드블루] 숨을 곳 없다**
or
**2월 긴장감 최고 [-블랙앤드블루] 숨 쉴 곳 없어**

After inference - Allocated: 4685.94 MB, Reserved: 4880.00 MB
```

### Test Results

- Detail results
    - MLP-KTLim/llama-3-Korean-Bllossom-8B
    
    ```python
    python3 korean_blossom_llama.py 
    # input_text = "2O19.12월 초 긴 급 ( ㅇㅣㅅㅣ언 왕 ㅈ ㅣ ㅎ ㅖ ) [ 죽 였 ㄷㅏ ㅇㅏㄴㅐ를 ]"
    Loading checkpoint shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,  5.74it/s]
    The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
    Setting `pad_token_id` to `eos_token_id`:128009 for open-end generation.
    The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
    **복원된 텍스트: 2020년 1월 초 긴 휴식(이사윤 왕 자고 말고 쉬었다) [ 죽었다고 들었다는 것 ]**
    ```
    
    ```python
    python3 korean_blossom_llama.py 
    # input_text = "2O19년12월 ((Oi완멕그ㄹ1거)) []ㅡ 닥 . Eㅓ. 슬 . 립 ㅡ[] -초현실능력자- 완벽자체"
    Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,  5.55it/s]
    The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
    Setting `pad_token_id` to `eos_token_id`:128009 for open-end generation.
    The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
    **복원된 텍스트: Output: "2019년 12월 ((Oi완벽한1등)) [이게 다. 여기. 슬프다. 립 안해[] -초현실능력자- 완벽자체"**
    ```
    
    **ex. KETI-AIR ke-t5-small**
    
    ```bash
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    
    # 모델과 토크나이저 로드
    model_name = "KETI-AIR/ke-t5-small"  # Hugging Face에서 제공하는 한국어 T5 모델
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    def restore_text_ai(text):
        # 입력 텍스트를 T5 입력 형식으로 변환
        input_text = f"복원: {text}"
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    
        # 모델 추론
        outputs = model.generate(inputs["input_ids"], max_length=512, num_beams=5, early_stopping=True)
    
        # 결과 디코딩
        restored_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return restored_text
    
    # 테스트
    test_text = "2O19.12월 초 긴 급 ( ㅇㅣㅅㅣ언 왕 ㅈ ㅣ ㅎ ㅖ ) [ 죽 였 ㄷㅏ ㅇㅏㄴㅐ를 ]"
    restored = restore_text_ai(test_text)
    print("복원된 텍스트:", restored)
    ```
    
    **→ Fail, No response**
    
    ---
    
    - **google-t5/t5-small : Fail, No response**
    - **KETI-AIR/ke-t5-small : Fail, Hallucinated**
    
    raw_text = "2O19.12월 초 긴 급 ( ㅇㅣㅅㅣ언 왕 ㅈ ㅣ ㅎ ㅖ ) [ 죽 였 ㄷㅏ ㅇㅏㄴㅐ를 ]”
    
    복원된 텍스트: 마찬가지다原原原 마찬가지입니다原山原原hankookilbohankookilbohankookilbo 민정수석hankookilbohankookilbo 좋겠지만hankookilbohankookilbo 많게는hankookilbohankookilbo 위메프hankookilbohankookilbo accidenthankookilbohankookilbo 특활비hankookilbohankookilbo Stamfordhankookilbohankookilbo 바르셀로나hankookilbohankookilbo크스hankookilbohankookilbo ridehankookilbohankookilbo 특수활동비hankookilbohankookilbo 금고hankookilbohankookilbo 조치했다hankookilbohankookilbo 물가hankookilbohankookilbo WTO Stamfordhankookilbo Orange Stamford Stamfordhankookilbo 환전 Stamfordgnradhankookilbohankookilbo door Stamford Stamford Stamford Bangkokhankookilbohankookilbo workshophankookilbohankookilbo 감사원은hankookilbohankookilbo ship Stamfordhankookilbo 세계랭킹hankookilbohankookilbo 잊혀지hankookilbohankookilbo Numberhankookilbohankookilbo 귓hankookilbohankookilbo LHhankookilbohankookilbo wheneverhankookilbohankookilbo 평창hankookilbohankookilbo 월드hankookilbohankookilbo 모니터hankookilbohankookilbo 왔지만hankookilbohankookilbo 수수료를hankookilbohankookilbo 우리금융지주hankookilbohankookilbo Princetonhankookilbohankookilbo 일으키 Stamfordhankookilbognradhankookilbo Stamford
    
    - **model_name = "hyunwoongko/kobart”, Fail**
    
    ```bash
    You passed along `num_labels=3` with an incompatible id to label map: {'0': 'NEGATIVE', '1': 'POSITIVE'}. The number of labels wil be overwritten to 2.
    응답 최소길이 : 18, 최대 길이 : 58
    입력: 2O19.12월 초 긴 급 ( ㅇㅣㅅㅣ언 왕 ㅈ ㅣ ㅎ ㅖ ) [ 죽 였 ㄷㅏ ㅇㅏㄴㅐ를 ]
    복원 결과: 
     다음 다음 문장을 올바른 한국어로 복원하세요:
    입력: 201912월초긴급이시언어로 복하세요:0
    출력.
    응답 최소길이 : 27, 최대 길이 : 67
    입력: 2019.02월 S.F 판타지 [--- ㅅㅣ . 공 . 간 . ㅇㅣ . 동 ---]한글자막 초고화질 1080P
    복원 결과: 
     다음 다음 문장을 올바른 한국어로 복원하세요:
    입력: 201902월SF판타지---시공간이동---1-한글자막초고고-- 한글 자막 초고 고고화질1080P
    ```
    
    ---
    

---

### If not using AI ?

- Need all type of regex matching dictionary for cipher transformed Korean words.
- sample

```python

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
```

```bash
import re

def restore_text(text):
    # 1. 자음/모음 분리 -> 합치기
    # 자음과 모음이 분리된 경우 유니코드 조합
    def combine_jamo(chars):
        CHO = "ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ"
        JUNG = "ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ"
        JONG = "ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"

        if len(chars) == 2:  # 초성 + 중성
            cho, jung = chars
            return chr(0xAC00 + CHO.index(cho) * 588 + JUNG.index(jung) * 28)
        elif len(chars) == 3:  # 초성 + 중성 + 종성
            cho, jung, jong = chars
            return chr(0xAC00 + CHO.index(cho) * 588 + JUNG.index(jung) * 28 + JONG.index(jong) + 1)
        return ''.join(chars)

    text = re.sub(r'\s+', '', text)  # 공백 제거
    text = re.sub(r'[^\w가-힣]', '', text)  # 특수문자 제거
    text = re.sub(r'[O]', '0', text)  # O → 0
    text = re.sub(r'[lI]', '1', text)  # l/I → 1
    text = re.sub(r'[B]', '8', text)  # B → 8
    
    # 자음/모음 분리된 문자 합치기
    result = []
    temp = []
    for char in text:
        if 'ㄱ' <= char <= 'ㅎ' or 'ㅏ' <= char <= 'ㅣ':  # 한글 자모
            temp.append(char)
            if len(temp) == 3 or (len(temp) == 2 and temp[1] in "ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ"):
                result.append(combine_jamo(temp))
                temp = []
        else:
            if temp:
                result.extend(temp)
                temp = []
            result.append(char)
    if temp:
        result.extend(temp)

    return ''.join(result)

# 테스트
test_text = "2O19.12월 초 긴 급 ( ㅇㅣㅅㅣ언 왕 ㅈ ㅣ ㅎ ㅖ ) [ 죽 였 ㄷㅏ ㅇㅏㄴㅐ를 ]"
restored = restore_text(test_text)
print("복원된 텍스트:", restored)
```