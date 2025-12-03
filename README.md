# TikTok Live Commerce Classifier

이 저장소는 **TikTok 라이브 쇼핑 방송을 자동으로 수집하고**,  
수집된 방송 영상·메타데이터를 기반으로 **상품 카테고리를 분류하는 CLIP 기반 모델**과  
이를 활용한 **Python 서버 파이프라인**으로 구성된 프로젝트입니다.

크게 다음 세 부분으로 나뉩니다.

1. TikTok 크롤링 (Node.js + Puppeteer)
2. 상품 분류 모델 학습 (PyTorch + CLIP)
3. 서버 및 파이프라인 (Flask 서버 + OpenAI/커스텀 LLM + MongoDB)

---

## 📂 Repository Structure

```text
tiktok_classifier/
├─ tiktok/                   # TikTok 라이브 검색/카드 탐색/녹화/메타데이터 수집
├─ train/
│  ├─ model3.py              # CLIP 기반 이미지/텍스트 분류 모델 학습 스크립트
│  └─ finetune.ipynb         # 추가 실험/노트북
├─ server/
│  ├─ server.py              # Flask 서버: 메타데이터 + 영상 -> 분류/LLM 호출
│  ├─ video_processor.py     # OpenCV 기반 프레임 추출
│  ├─ python_server_config.py# 경로/모델/LLM/Mongo 설정
│  └─ python_server_utils.py # CLIP 분류기, GPT/커스텀 LLM 호출 유틸
├─ glot_model/               # 학습된 모델/체크포인트(예: CLIP Cross-Attention)
├─ utils.js                  # 공통 JS 유틸 (delay, Whisper 서버 호출 등)
├─ main.js                   # TikTok 크롤러 엔트리 포인트
├─ whisper_fastapi_server.py # 음성 -> 텍스트(Whisper) 처리 서버
├─ cookies.json              # TikTok 세션 쿠키 (민감정보, 공유 금지)
└─ README.md                 # 프로젝트 설명 문서 (이 파일로 교체 가능)
```

---

# 1. TikTok Crawling (Node.js + Puppeteer)

`tiktok/` 디렉토리는 **TikTok 라이브 쇼핑 방송을 자동으로 수집**하는 크롤러를 담고 있습니다.  
Puppeteer를 사용하여 브라우저를 띄우고, 라이브 검색 페이지에서 실시간 방송 카드를 탐색하고,  
조건에 맞는 방송만 골라 영상 녹화/채팅 수집/메타데이터 저장을 수행합니다.

### 1.1 크롤링 주요 흐름 (`main.js`)

1. `launchBrowser()`  
   - `tiktok/browser.js`  
   - `puppeteer-extra` + `stealth plugin`을 사용해 실제 크롬 브라우저 실행  
   - `cookies.json`을 로드해서 로그인/연령제한 우회 등 세션 적용

2. `openSearchPage(page, url)`  
   - `tiktok/searchPage.js`  
   - 특정 키워드(예: "실시간쇼핑", "악세사리", "틱톡 라이브 쇼핑")로 검색된 라이브 리스트 페이지 오픈  
   - 로딩이 끝날 때까지 대기

3. `extractAllCards(page)`  
   - `tiktok/cardList.js`  
   - 현재 검색 페이지에서 `data-e2e="search_live-item"` / `search-card-desc` 를 기준으로 라이브 카드 리스트 추출  
   - 각 카드에서 `userId` 와 클릭 가능한 `link`를 묶어서 반환

4. 방송 필터링  
   - `broadcastFilter.js`  
   - 시청자 수가 0이거나 비정상인 경우, 카운터를 올리며 스킵  
   - 연속으로 너무 많은 비정상 방송이 나오면 탐색 전략을 조정할 수 있도록 설계

5. 사용자 쿨다운  
   - `userCooldown.js`  
   - 같은 유저의 방송을 짧은 시간 내에 반복 수집하지 않도록 파일 기반 timestamp 관리  
   - `loadCooldownData / saveCooldownData / isUserCooldown / cleanupCooldown` 제공

6. 목표 카드 처리 (`processTargetCard`)  
   - `tiktok/cardHandler.js`  
   - 특정 방송 카드를 클릭해 라이브 방송 방으로 들어감  
   - 시청자 수를 먼저 파싱하고, 유효한 방송만 이후 단계 진행  
   - 병렬로 다음 작업 수행:
     - `recordScreen2`로 화면 녹화 (mp4 저장, `recordings/` 디렉토리 등)
     - `extractChatMessages`로 일정 시간 동안 채팅 수집
     - 방송 제목, 판매자, 채널 URL 등 메타데이터 추출
   - Whisper 서버에 녹화된 영상 경로를 전달하여 음성 → 텍스트(자막) 변환  
     - `utils.js`의 `sendToWhisperServer(filepath)` 사용
   - 최종적으로
     - liveUrl, channelUrl, title, seller, chatMessages, transcript, viewerCount 등
     - JSON 형태로 디스크에 저장

7. 에러 및 자원 정리  
   - 방송 처리 중 오류 발생 시 로그만 남기고 크롤러는 계속 진행  
   - 새로 연 탭은 처리 완료 후 반드시 닫고, 메인 검색 페이지는 유지

> **데이터셋 규모**  
> 위 과정을 통해 구축한 라이브 커머스 데이터셋은  
> **약 1만 개 수준의 방송 샘플(메타데이터 + 영상 + 채팅 + Whisper 자막)** 으로 구성할 수 있도록 설계되었습니다.

---

# 2. 상품 분류 모델 (PyTorch + CLIP)

`train/model3.py`는 **TikTok 라이브 영상에서 추출한 이미지들을 입력으로**  
실제 판매 상품을 카테고리별로 분류하는 **CLIP 기반 분류 모델**을 학습하는 스크립트입니다.

### 2.1 데이터 구조

- `image_root = "real_images_6_cropped"`  
  - 각 방송 단위를 하나의 폴더로 두고, 그 안에 최대 6장의 크롭된 이미지(상품 중심)를 저장  
  - 6장 미만일 경우 마지막 이미지를 반복하여 6장을 맞춤

- `label_root = "real_data3"`  
  - 각 폴더 이름과 동일한 JSON 라벨 파일이 존재  
  - 예시:
    ```json
    {
      "category": "clothing",
      "query": "반팔티 라이브 커머스"
    }
    ```

- 데이터셋 클래스: `ClipImageOnlyDataset`
  - 폴더 리스트를 기준으로 이미지를 로딩하고,
  - `mode`에 따라
    - `"text"`: 텍스트가 있는 샘플만 사용
    - `"both"`: 텍스트 유무 상관 없이 사용
  - `pixel_values` (이미지 텐서), `text` (query 문자열), `label` (카테고리 id) 반환

데이터는 `train/val/test`로 **폴더 단위 랜덤 분할**되며,  
총 샘플 수는 약 **1만개 규모**로 설정해 두고 실험했습니다.

---

## 2.2 CLIPImageClassifier (단순 이미지 분류)

`CLIPImageClassifier`는 **이미지 특징만 사용**하는 베이스라인 모델입니다.

- `CLIPVisionModel`(예: `openai/clip-vit-base-patch16`)을 백본으로 사용
- 각 이미지에서 `[CLS]` 토큰(첫 토큰) 임베딩을 추출
- 하나의 방송에 대해 여러 이미지가 있을 경우 **이미지 임베딩을 평균**하여 하나의 벡터로 압축
- MLP 분류기:
  - 768 → 512 → num_classes
  - ReLU + Dropout(0.2) 포함
- `CrossEntropyLoss`, `AdamW(lr=1e-5)`로 학습

이 모델은 **텍스트 정보 없이도 이미지 만으로 어느 정도 카테고리 분류가 가능한지**를 보는 용도입니다.

---

## 2.3 CLIPCrossAttentionClassifier (이미지-텍스트 Cross-Attention)

실제 프로젝트에서는 **텍스트(제목/쿼리)와 이미지를 함께 활용**하는  
`CLIPCrossAttentionClassifier`를 주요 모델로 사용합니다.

구조 요약:

1. **Vision Encoder**
   - `CLIPVisionModel.from_pretrained(clip_model_name)`
   - 입력: (B, N, C, H, W)
     - B: 배치 크기
     - N: 한 방송에서 사용한 이미지 수(최대 6장)
   - 처리:
     - (B, N, C, H, W) → (B*N, C, H, W) → CLIP 입력
     - 마지막 hidden state에서 `[CLS]`를 제외하고 패치 토큰들만 사용하거나,
     - 패치 차원을 평균해 (B, D) 혹은 (B, P, D) 형태의 시퀀스로 정제
   - 추가 projection:
     - `nn.Linear(vis_dim, 512)`로 임베딩 차원 축소

2. **Text Encoder**
   - `CLIPTextModel` + `CLIPTokenizer`
   - 입력: 라벨 JSON에 들어 있는 `query` 또는 Whisper 텍스트 기반 문장
   - 출력: (B, L, 512) 형태의 토큰 시퀀스 임베딩

3. **Cross-Attention Layer**
   - `nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)`
   - Query: text 임베딩 `t` (질문/설명)
   - Key/Value: vision 임베딩 `v` (이미지 패치)
   - 의미:
     - 텍스트가 "어떤 상품인지"를 물어보는 쿼리 역할
     - 이미지 패치는 상품 후보의 시각적 단서를 제공
     - Cross-Attention을 통해 **텍스트가 강조하는 부분에 집중한 시각 표현**을 형성

4. **Pooling & Classifier**
   - Cross-Attention 출력 (B, L, 512)를 토큰 차원 평균
   - MLP 분류기:
     - 512 → 512 → num_classes
     - ReLU + Dropout(0.2) 포함

5. **학습 설정**
   - 손실 함수: `CrossEntropyLoss`
   - Optimizer: `AdamW`
   - AMP(`torch.cuda.amp`) 및 gradient clipping으로 안정성 확보
   - `EarlyStopping`을 사용해 검증 loss 기준으로 best 모델 선택
   - 최종 weight는 `clip_crossattention_classifier.pth`로 저장

이 구조 덕분에 모델은

- **이미지 단독**이 아닌 **텍스트 조건부(제목/쿼리/자막)**로 상품을 분류할 수 있고,
- 시각·언어 정보가 서로 보완하며 **복잡한 라이브 커머스 장면에서도 더 정확한 카테고리 예측**이 가능합니다.

---

# 3. 서버 파이프라인 (Python Flask + CLIP + LLM)

`server/` 디렉토리는 **영상/메타데이터를 입력받아 최종 카테고리/설명을 반환하는 서버 파트**를 담당합니다.

### 3.1 주요 파일

- `server.py`
  - Flask 기반 HTTP 서버
  - 클라이언트로부터 `meta_filepath` 등을 입력받아 해당 방송의 JSON/영상 정보를 읽음
  - `video_processor.extract_frames_opencv`로 mp4 파일에서 프레임 추출
  - 추출된 이미지들을 CLIP 기반 분류기로 전달하여 상품 카테고리 예측
  - OpenAI GPT 또는 커스텀 LLM API(ngrok 주소)를 호출해 요약/설명 문장 생성
  - 최종 결과를 MongoDB에 저장 (`MONGO_URI`, `MONGO_DB_NAME`, `MONGO_COLLECTION_NAME`)

- `video_processor.py`
  - OpenCV 기반 프레임 추출
  - 중심 영역을 기준으로 일정 크기로 crop
  - `num_frames`개만 균등 간격으로 추출하여 이미지 저장

- `python_server_config.py`
  - BASE_PROJECT_PATH, VIDEO_DIR, META_DIR
  - CLIP 모델 경로, LLM API URL, OpenAI key
  - MongoDB 설정 (URI, DB, 컬렉션 이름 등)
  - `DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")`

- `python_server_utils.py`
  - GPT prompt 템플릿 (`PROMPT_TEMPLATE`)
  - OpenAI `gpt-4o-mini` 호출 함수
  - 커스텀 LLM API(`/generate`) 호출 함수
  - 서버 측에서 사용하는 CLIP Cross-Attention 분류기 정의
  - 저장된 이미지들을 로드하고 모델에 넣어 최종 카테고리 문자열 반환

---

# 4. 실행 예시

> ⚠️ 실제 경로, 환경변수, API 키, MongoDB URI 등은 개인 환경에 맞게 수정해야 합니다.

### 4.1 TikTok 크롤링 (Node.js)

```bash
# Node.js 의존성 설치
npm install

# 크롤러 실행
node main.js
```

- `cookies.json`에 TikTok 로그인 쿠키를 저장해 두어야 안정적인 수집이 가능합니다.
- 수집된 영상은 `recordings/`, 메타데이터는 별도의 JSON 파일로 저장됩니다.

### 4.2 모델 학습 (Python)

```bash
cd train
python model3.py
```

- `real_images_6_cropped/`, `real_data3/` 경로에 맞게 데이터 구성  
- 학습이 끝나면 `clip_crossattention_classifier.pth`가 저장됩니다.

### 4.3 서버 실행

```bash
cd server
python server.py
```

- 설정: `python_server_config.py`에서 BASE_PROJECT_PATH, NGROK_URL, OPENAI API 키, MONGO_URI 등을 수정
- 클라이언트는 meta 파일 경로를 전달하여 방송 분석 결과를 받을 수 있습니다.

---




