# python_server_utils.py 또는 python_server.py 내 별도 함수들로 정의

from video_processor import extract_frames_opencv
from PIL import Image
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
import os
import requests
# GPT 프롬프트 템플릿
PROMPT_TEMPLATE = """
You are an expert annotator analyzing Korean live commerce broadcasts.

Product categories:
- clothing (shirts, jackets, dresses, pants, vintage,  etc.)
- accessories (sanrio, figures, bags, necklaces, scarves, shoes, hats, acrylic, etc.)
- food (snacks, beverages, ingredients, health food, etc.)
- cosmetics (beauty products, skincare, makeup, personal care, etc.)
- furniture (interior items, bedding, sofas, chairs, home decor, blankets, etc.)
- lifestyle (household items, convenience products, everyday goods, etc.)
- electronics (digital devices, home appliances, electronic gadgets, etc.)
- none (if no clear physical product is identified)

Your task:
- Analyze title, chat messages, and transcript to find the PHYSICAL product being sold
- Chat messages contain usernames - focus only on the product-related content after usernames
- Look for actual merchandise/goods being sold, not services or links
- Output ONLY in English using simple product names
- Use "none" only if absolutely no physical product can be identified

"title": "아쿠아 슈즈/슬리퍼 세일", ...
→ Product: shoes
Title: 오몽이네 🍀 피규어,파츠 키링제작 잡화점🎁
→ Product: figurines, keychains

Title: {title}
Chat: {chat}
Transcript: {whisper}

Product: """

def build_gpt_prompt(title, chat, whisper):
    """GPT 프롬프트 생성"""
    chat_str = "\n".join(chat) if isinstance(chat, list) else str(chat)
    return PROMPT_TEMPLATE.format(title=title, chat=chat_str, whisper=whisper)

def call_gpt_api(client, prompt, model="gpt-4o-mini", max_tokens=32, temperature=0.0):
    """OpenAI GPT API 호출"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()

def call_custom_llm_api(url, prompt_text, max_new_tokens=50, timeout=30):
    """사용자 정의 LLM API (ngrok) 호출"""
    llm_payload = {
        "prompt": f"프롬프트: {prompt_text}응답:",
        "max_new_tokens": max_new_tokens
    }
    response = requests.post(url, json=llm_payload, timeout=timeout)
    response.raise_for_status() # HTTP 오류 발생 시 예외 발생
    return response.json().get("generated_text", "").strip()

# CLIPCrossAttentionClassifier 클래스 정의
class CLIPCrossAttentionClassifier(nn.Module):
    def __init__(self, num_classes=6,
                 clip_model_name="openai/clip-vit-base-patch16"):
        super().__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_model_name)
        self.text_encoder   = CLIPTextModel.from_pretrained(clip_model_name)

        vis_dim  = self.vision_encoder.config.hidden_size
        txt_dim  = self.text_encoder.config.hidden_size

        self.img_proj   = nn.Linear(vis_dim, txt_dim)
        self.cross_attn = nn.MultiheadAttention(txt_dim, 8, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(txt_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

        self.tokenizer  = CLIPTokenizer.from_pretrained(clip_model_name)
        self.processor  = CLIPImageProcessor.from_pretrained(clip_model_name)

    def forward(self, images, text_inputs):
        device = images.device

        # 5D → 4D (배치 내 여러 이미지 처리)
        if images.dim() == 5:
            B, N, C, H, W = images.shape
            images_flat   = images.view(B * N, C, H, W)
        else:
            B, images_flat = images.shape[0], images

        # Vision
        v = self.vision_encoder(pixel_values=images_flat).last_hidden_state[:, 1:, :]
        if images.dim() == 5:
            v = v.view(B, N, -1, v.size(-1)).mean(dim=1)

        v = self.img_proj(v)

        # Text
        tok = self.tokenizer(text_inputs, return_tensors="pt",
                             padding=True, truncation=True).to(device)
        t   = self.text_encoder(**tok).last_hidden_state

        # Cross-Attention
        attn, _ = self.cross_attn(query=t, key=v, value=v)
        pooled  = attn.mean(dim=1)
        return self.classifier(pooled)

def load_clip_model(model_path, num_classes, device):
    """CLIP Cross-Attention 분류 모델 로드"""
    model = CLIPCrossAttentionClassifier(num_classes=num_classes).to(device)
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            # 'module.' 접두사 제거 (nn.DataParallel로 학습된 경우)
            new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
            model.eval() # 평가 모드
            print(f"CLIP Cross-Attention model loaded successfully from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading CLIP model from {model_path}: {e}")
            return None
    else:
        print(f"CLIP model not found at {model_path}. CLIP classification will be skipped.")
        return None

def predict_with_clip(clip_model, image_paths, text_input, categories, device):
    if not clip_model:
        return "clip_model_not_loaded"
    if not image_paths:
        return "no_frames_extracted"
    
    images_for_clip = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
            images_for_clip.append(clip_model.processor(images=img, return_tensors="pt").pixel_values)
        else:
            print(f"Warning: Image file not found at {img_path}")

    if not images_for_clip:
        return "no_valid_images_found"

    images_tensor = torch.cat(images_for_clip, dim=0).to(device)
    
    processed_text_input = [text_input.split('Product: ')[-1].strip() if 'Product:' in text_input else text_input]

    with torch.no_grad():
        try:
            outputs = clip_model(images_tensor.unsqueeze(0), processed_text_input)
            predicted_label_idx = torch.argmax(outputs, dim=1).item()
            return categories[predicted_label_idx]
        except Exception as e:
            return f"clip_inference_error: {str(e)}"
