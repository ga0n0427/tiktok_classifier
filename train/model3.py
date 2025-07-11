import os, json, torch, random
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchvision import transforms
from transformers import (
    CLIPTextModel, CLIPVisionModel,
    CLIPTokenizer, CLIPImageProcessor
)
from PIL import Image

# ──────────────────────── 0. 환경 설정 ────────────────────────
image_root  = "real_images_6_cropped"
label_root  = "real_data3"
batch_size  = 8
epochs      = 20
num_workers = 0
grad_accum  = 4
device      = "cuda" if torch.cuda.is_available() else "cpu"

# ───────────────────── 1. 데이터셋 정의 ───────────────────────
class ClipImageOnlyDataset(Dataset):
    """
    mode = "both"   : 텍스트 O/X  모두 (2× len)
    mode = "text"   : 텍스트 O 만
    mode = "notext" : 텍스트 X 만
    """
    def __init__(self, image_root, label_root, mode="both"):
        self.image_root, self.label_root, self.mode = image_root, label_root, mode

        folders = set(os.listdir(image_root))
        labels  = {f.replace(".json", "") for f in os.listdir(label_root)
                   if f.endswith(".json")}
        self.folder_list = sorted(list(folders & labels))

        self.label2id = {
            "식품": 0, "생활_편의": 1, "패션의류": 2, "패션잡화": 3,
            "디지털_인테리어": 4, "화장품_미용": 5,
        }
        self.id2label = {v: k for k, v in self.label2id.items()}

        self.processor = CLIPImageProcessor.from_pretrained(
            "openai/clip-vit-base-patch16"
        )
        # ✅ 증강(augment) 완전히 제거
        self.augment = None

    def __len__(self):
        return len(self.folder_list) * 2 if self.mode == "both" else len(self.folder_list)

    def __getitem__(self, idx):
        # ── 텍스트 사용여부 결정 ────────────────────────────
        if self.mode == "both":
            base_idx, use_text = idx // 2, idx % 2 == 0
        elif self.mode == "text":
            base_idx, use_text = idx, True
        else:   # "notext"
            base_idx, use_text = idx, False

        folder  = self.folder_list[base_idx]
        img_dir = os.path.join(self.image_root, folder)

        # ── 이미지 최대 6장 로드 (6장 미만이면 마지막 이미지 복제) ──
        img_files = sorted([f for f in os.listdir(img_dir)
                            if f.lower().endswith((".jpg", ".png"))])
        if not img_files:
            raise RuntimeError(f"[{folder}] 이미지가 없습니다")

        while len(img_files) < 6:
            img_files.append(img_files[-1])
        img_files = img_files[:6]

        images = [Image.open(os.path.join(img_dir, f)).convert("RGB")
                  for f in img_files]
        # (증강 없음) images 그대로 사용
        pixel_values = self.processor(images=images,
                                      return_tensors="pt")["pixel_values"]

        # ── 라벨 & 텍스트 ──────────────────────────────────
        with open(os.path.join(self.label_root, f"{folder}.json"), encoding="utf-8") as fp:
            data = json.load(fp)

        label = torch.tensor(self.label2id[data["category"]], dtype=torch.long)
        text  = data["query"] if use_text else ""

        return pixel_values, text, label
    
# ───────────────── 2. 데이터·로더 준비 (중복 없는 버전) ──────────────────
# ① 폴더 리스트 한 번만 생성
base_ds = ClipImageOnlyDataset(image_root, label_root, mode="text")   # 텍스트 O 전용

total_folders = len(base_ds)            # 폴더 개수
val_len  = max(1, int(total_folders * 0.1))
test_len = max(1, int(total_folders * 0.2))
train_len = total_folders - val_len - test_len

g = torch.Generator().manual_seed(0)
train_idx, val_idx, test_idx = torch.utils.data.random_split(
    range(total_folders), [train_len, val_len, test_len], generator=g
)

# ② 인덱스를 재사용해 서로 다른 mode 의 데이터셋 생성
def make_subset(indices, mode):
    ds = ClipImageOnlyDataset(image_root, label_root, mode)   # 새 Dataset
    ds.folder_list = [ds.folder_list[i] for i in indices]     # 폴더 중복 X
    return ds

train_set = make_subset(train_idx.indices, mode="both")   # 텍스트 O / X 모두
val_set   = make_subset(val_idx.indices,   mode="text")   # 텍스트 O만
test_set  = make_subset(test_idx.indices,  mode="text")   # 텍스트 O만

# ③ DataLoader
train_loader = DataLoader(train_set, batch_size=batch_size,
                          shuffle=True,  num_workers=num_workers)
val_loader   = DataLoader(val_set,   batch_size=batch_size,
                          shuffle=False, num_workers=num_workers)
test_loader  = DataLoader(test_set,  batch_size=batch_size,
                          shuffle=False, num_workers=num_workers)

class CLIPImageClassifier(nn.Module):
    def __init__(self, num_classes=6, clip_model_name="openai/clip-vit-base-patch16"):
        super().__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_model_name)
        vis_dim = self.vision_encoder.config.hidden_size  # 768

        self.classifier = nn.Sequential(
            nn.Linear(vis_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

        self.processor = CLIPImageProcessor.from_pretrained(clip_model_name)

    def forward(self, images, text_inputs=None):
        device = images.device

        # (B, N, C, H, W) → (B*N, C, H, W)
        if images.dim() == 5:
            B, N, C, H, W = images.shape
            images_flat = images.view(B * N, C, H, W)
        else:
            B, images_flat = images.shape[0], images

        # Vision → patch (B*N, P, 768) → [CLS] (B*N, 768)
        v = self.vision_encoder(pixel_values=images_flat).last_hidden_state[:, 0, :]
        if images.dim() == 5:
            v = v.view(B, N, -1).mean(dim=1)  # (B, 768)

        logits = self.classifier(v)  # (B, num_classes)
        return logits

# ────────────────── 3. 모델 정의 (512-dim Cross-Attn) ─────────────────
class CLIPCrossAttentionClassifier(nn.Module):
    def __init__(self, num_classes=6,
                 clip_model_name="openai/clip-vit-base-patch16"):
        super().__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained(clip_model_name)
        self.text_encoder   = CLIPTextModel.from_pretrained(clip_model_name)

        vis_dim  = self.vision_encoder.config.hidden_size   # 768
        txt_dim  = self.text_encoder.config.hidden_size     # 512

        self.img_proj   = nn.Linear(vis_dim, txt_dim)       # 768→512
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

        # 5D → 4D
        if images.dim() == 5:
            B, N, C, H, W = images.shape
            images_flat   = images.view(B * N, C, H, W)
        else:
            B, images_flat = images.shape[0], images

        # Vision → patch (B’ , P , 768)  ※ B’ = B×N 또는 B
        v = self.vision_encoder(pixel_values=images_flat).last_hidden_state[:, 1:, :]
        if images.dim() == 5:
            v = v.view(B, N, -1, v.size(-1)).mean(dim=1)     # (B, P, 768)

        v = self.img_proj(v)                                 # (B, P, 512)  ★ 768→512

        # Text
        tok = self.tokenizer(text_inputs, return_tensors="pt",
                             padding=True, truncation=True).to(device)
        t   = self.text_encoder(**tok).last_hidden_state      # (B, L, 512)

        # Cross-Attention
        attn, _ = self.cross_attn(query=t, key=v, value=v)    # (B, L, 512)
        pooled  = attn.mean(dim=1)                            # (B, 512)
        return self.classifier(pooled)                        # (B, num_classes
    
model = CLIPCrossAttentionClassifier().to(device)
train_loader = DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,          # 단일 프로세스
    pin_memory=True         # OK
    # prefetch_factor, persistent_workers  제거!
)

common_kwargs = dict(batch_size=batch_size,
                     shuffle=False,
                     num_workers=0,      # ← 동일하게 0
                     pin_memory=True)
val_loader  = DataLoader(val_set,  **common_kwargs)
test_loader = DataLoader(test_set, **common_kwargs)

criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-5)

# ❸ AMP 스케일러
scaler = torch.cuda.amp.GradScaler()

from tqdm.auto import tqdm   # pip install tqdm

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    torch.set_grad_enabled(train)

    total_loss, correct, total = 0., 0, 0
    if train: optimizer.zero_grad(set_to_none=True)

    bar = tqdm(loader, desc="Train" if train else "Eval", leave=False)

    for step, (imgs, texts, labels) in enumerate(bar):
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(images=imgs, text_inputs=texts)
            loss   = criterion(logits, labels) / grad_accum

        if train:
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * grad_accum * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

        # tqdm 왼쪽 상태줄 업데이트
        bar.set_postfix(loss=loss.item()*grad_accum, acc=100*correct/total)

    return total_loss/total, 100.*correct/total
BEST_MODEL_PATH = "clip_crossattention_best.pth"
# ───────────────────── 5. 학습 루프 ───────────────────────────
class EarlyStopping:
    def __init__(self, patience=3, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.best = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best is None or val_loss < self.best - self.delta:
            self.best = val_loss
            self.counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)  # ✅ 저장
            print(f"💾 Best model saved (val_loss: {val_loss:.4f})")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

if __name__ == "__main__":
    early_stopper = EarlyStopping()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(train_loader, train=True)
        va_loss, va_acc = run_epoch(val_loader,   train=False)
        print(f"[{epoch:02d}/{epochs}] Train {tr_loss:.4f}/{tr_acc:5.2f}% │ "
              f"Val {va_loss:.4f}/{va_acc:5.2f}%")
        early_stopper(va_loss, model)  # ✅ model 전달
        if early_stopper.early_stop:
            print("⛔ Early Stopping"); break

    # ───────────────────── 6. 테스트 & 저장 ──────────────────────
    #model.load_state_dict(torch.load(BEST_MODEL_PATH))
    te_loss, te_acc = run_epoch(test_loader, train=False)
    print(f"\n🧪 Test Loss {te_loss:.4f} | Acc {te_acc:5.2f}%")

    torch.save(model.state_dict(), "clip_crossattention_classifier.pth")
    print("✅ 모델 저장 완료")
