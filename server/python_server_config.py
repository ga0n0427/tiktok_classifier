# python_server_config.py 또는 python_server.py 상단에 위치

import os
import json
import requests
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
from openai import OpenAI
from PIL import Image
import requests

# --- 기본 경로 설정 ---
BASE_PROJECT_PATH = 'C:\\gaon\\real_tiktok'
VIDEO_DIR = os.path.join(BASE_PROJECT_PATH, 'recordings')
META_DIR = BASE_PROJECT_PATH
IMAGE_OUTPUT_BASE_DIR = os.path.join(BASE_PROJECT_PATH, 'images')
os.makedirs(IMAGE_OUTPUT_BASE_DIR, exist_ok=True)

# --- 모델 및 API 설정 ---
CLIP_MODEL_PATH = os.path.join(BASE_PROJECT_PATH, 'clip_crossattention_best.pth')
# 중요: 이 클래스 순서는 모델 학습 시 사용된 클래스 순서와 정확히 일치해야 합니다.
# num_classes=6에 맞춰 6개 카테고리로 정의. 'none'은 GPT 판단에 의존.
FINAL_CATEGORIES = [
    "식품", "생활_편의", "패션의류", "패션잡화",
    "디지털_인테리어", "화장품_미용"
]
NGROK_URL = ""
API_ENDPOINT = "/generate"
FULL_LLM_API_URL = f"{NGROK_URL}{API_ENDPOINT}"

# --- OpenAI 클라이언트 초기화 ---
open_key = ""
client = OpenAI(api_key=open_key) # 환경 변수 OPENAI_API_KEY가 설정되어 있어야 합니다

# --- PyTorch 디바이스 설정 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MONGO_URI = "" # 실제 MongoDB URI로 변경해주세요

MONGO_DB_NAME = "" # 사용할 데이터베이스 이름

MONGO_COLLECTION_NAME = "tiktok_url" # 사용할 컬렉션 이름