import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import re

SUBSTITUTION_MAP = {
    "ㄴ": ["L"],      # Visually similar vertical shape
    "ㅏ": ["r"],      # Right-facing hook
    "ㅣ": ["i", "I"], # Straight vertical line
    "ㅌ": ["E"],
    "사": ["4"],
    "1": ["①"],
    "2": ["②"],
    "3": ["③"],
    "4": ["④"],
    "5": ["⑤"],
    "6": ["⑥"],
    "7": ["⑦"],
    "8": ["⑧"],
    "9": ["⑨"]
}

def preprocess_text(text):
    # Replace visually similar characters
    for key, values in SUBSTITUTION_MAP.items():
        for value in values:
            text = text.replace(value, key)
    
    # Remove extra spaces
    text = re.sub(r'\s+', '', text)
    return text

def print_gpu_memory_usage(stage):
    # Print GPU memory usage for the given stage
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    print(f"{stage} - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")

# Initialize Accelerator
accelerator = Accelerator()

# Load the model and tokenizer
model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map=None  # Accelerator will handle the device
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare the model with Accelerator
model = accelerator.prepare(model)

# Test input list
test_inputs = [
    "2O19년12월 ((Oi완멕그ㄹ1거)) []ㅡ 닥 . Eㅓ. 슬 . 립 ㅡ[] -초현실능력자- 완벽자체",
    "Lr는 ㄱㅏ 끔 눈물을 흘린ㄷr...",
    "2O19.12월 초 긴 급 ( ㅇㅣㅅㅣ언 왕 ㅈ ㅣ ㅎ ㅖ ) [ 죽 였 ㄷㅏ ㅇㅏㄴㅐ를 ]",
    "2019.02월 S.F 판타지 [--- ㅅㅣ . 공 . 간 . ㅇㅣ . 동 ---]한글자막 초고화질 1080P",
    "②O②O.01월 - 긴장감 최고치 - [ - 블.랙.앤.드.블.루 - ] 숨을곳없다"
]

# Iterate through each test input
for test_input in test_inputs:
    # Apply preprocessing to the input
    preprocessed_input = preprocess_text(test_input)

    # Create concise prompt messages
    messages = [
        {"role": "system", 
         "content": ("You are a model that restores distorted Korean text into its correct form. "
                     "Follow these rules:\n"
                     "1. Replace distorted Korean characters with their original Korean equivalents.\n"
                     "2. Retain original English words and numbers without modification.\n"
                     "3. Replace special symbols and punctuation with appropriate Korean equivalents.\n"
                     "4. Remove unnecessary particles like '은', '는', '이', '가', '로', '에', etc., to make the output concise and title-like.\n"
                     "5. Ensure the output remains meaningful and grammatically correct as a title."
                     "Do not add explanations or commentary.")},
        {"role": "user", "content": f"{preprocessed_input}"}
    ]


    # Tokenize the input
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    # Move input tensors to the Accelerator device
    input_ids = input_ids.to(accelerator.device)

    # Generate the output
    output = model.generate(
        input_ids,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=128,
        do_sample=False
    )

    # Decode the generated output
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)


    # Extract only the assistant's response
    if "[|assistant|]" in decoded_output:
        decoded_output = decoded_output.split("[|assistant|]")[-1].strip()
    
    # Print the results
    print(f"original text: {test_input}")
    print(f"preprocessed text: {preprocessed_input}")
    print(f"decoded text: {decoded_output}")
    print_gpu_memory_usage("After inference")
    print("-" * 50)