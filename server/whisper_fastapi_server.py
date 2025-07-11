from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from faster_whisper import WhisperModel
import os
import traceback
#uvicorn main:app --reload --host 0.0.0.0 --port 8000
app = FastAPI()
model = WhisperModel("large")  # 또는 "medium", "large-v2"

class TranscribeRequest(BaseModel):
    filepath: str

@app.post("/process_metadata")
def transcribe_audio(req: TranscribeRequest):
    path = req.filepath
    print(f"Transcribing file: {path}")

    # ❌ 파일이 존재하지 않음
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="❌ 파일이 존재하지 않습니다")

    # ❌ 용량이 너무 작거나 비어있음
    if os.path.getsize(path) < 1000:
        raise HTTPException(status_code=400, detail="❌ 파일 크기가 너무 작습니다 (무음 혹은 손상 가능성)")

    try:
        segments, info = model.transcribe(path, beam_size=5, language="ko")

        transcript = " ".join([seg.text.strip() for seg in segments])
        if not transcript.strip():
            raise HTTPException(status_code=204, detail="⚠️ 전사된 텍스트가 없습니다 (무음 또는 인식 실패)")

        return {
            "language": info.language,
            "transcript": transcript
        }

    except IndexError:
        raise HTTPException(status_code=422, detail="❌ 오디오 스트림을 찾을 수 없습니다 (파일 손상 또는 무효)")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"❌ Whisper 처리 중 오류: {str(e)}")
