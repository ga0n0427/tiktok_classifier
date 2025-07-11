# app.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from flask import Flask, request, jsonify

# --- 모델 로드 및 설정 부분 ---
fine_tuned_model_path = "checkpoint-2030"
base_model_name = "EleutherAI/polyglot-ko-3.8b"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)
    print(f"Tokenizer loaded from fine-tuned model path: {fine_tuned_model_path}")
except Exception:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    print(f"Tokenizer loaded from base model: {base_model_name}")

tokenizer.add_special_tokens({"eos_token": "</s>"})
tokenizer.pad_token = tokenizer.eos_token

print("✅ Tokenizer 준비 완료")

quantization_config = None
if torch.cuda.is_available():
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

print(f"Loading base model '{base_model_name}'...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=quantization_config if torch.cuda.is_available() else None,
    device_map="auto" if torch.cuda.is_available() else None,
    torch_dtype=torch.float16 if torch.cuda.is_available() and quantization_config else torch.float32,
    low_cpu_mem_usage=True
)

base_model.resize_token_embeddings(len(tokenizer))
print("✅ Base Model 로드 완료")

print(f"Loading PEFT adapter from '{fine_tuned_model_path}'...")
model = PeftModel.from_pretrained(base_model, fine_tuned_model_path)
model.eval()
print("✅ 모델 + LoRA 설정 완료")
print(f"Model loaded successfully on {device}!")

# --- Flask 애플리케이션 시작 ---
app = Flask(__name__)

# --- 텍스트 생성 함수  ---
def generate_response(prompt_text, max_new_tokens=100):
    # input_ids와 attention_mask를 함께 생성
    encoded_input = tokenizer(prompt_text, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = encoded_input.input_ids
    attention_mask = encoded_input.attention_mask 

    # 모델 생성 시작 인덱스 (입력 프롬프트 길이)
    input_len = input_ids.shape[1]

    generated_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        repetition_penalty=1.2,
        no_repeat_ngram_size=2,
        num_beams=1
    )

    # 생성된 토큰 중 입력 프롬프트 부분을 제외하고 디코딩
    generated_response_ids = generated_ids[0][input_len:]
    generated_text = tokenizer.decode(generated_response_ids, skip_special_tokens=False)

    # </s> 토큰을 기준으로 자르기 (추가적인 후처리)
    if "</s>" in generated_text:
        final_response = generated_text.split("</s>")[0].strip()
    else:
        final_response = generated_text.strip()

    return final_response

# --- API 엔드포인트 정의  ---
@app.route('/generate', methods=['POST'])
def handle_generate_request():
    data = request.json
    if not data or 'prompt' not in data:
        return jsonify({"error": "Prompt is required in the request body."}), 400

    prompt = data['prompt']
    max_new_tokens = data.get('max_new_tokens', 100)

    print(f"Received prompt: '{prompt}'")
    try:
        response = generate_response(prompt, max_new_tokens)
        print(f"Generated response: '{response}'")
        return jsonify({"generated_text": response})
    except Exception as e:
        print(f"Error during text generation: {e}")
        return jsonify({"error": str(e)}), 500

# --- 서버 실행 부분  ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
