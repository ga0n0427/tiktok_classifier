from flask import Flask, request, jsonify
import os
import json
import requests
from datetime import datetime, timezone
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, PyMongoError

# 위에서 정의한 설정값들을 임포트
from python_server_config import (
    BASE_PROJECT_PATH, VIDEO_DIR, META_DIR, IMAGE_OUTPUT_BASE_DIR,
    CLIP_MODEL_PATH, FINAL_CATEGORIES, NGROK_URL, API_ENDPOINT,
    FULL_LLM_API_URL, client, DEVICE,
    MONGO_URI, MONGO_DB_NAME, MONGO_COLLECTION_NAME # MongoDB 설정 임포트
)

# 위에서 정의한 유틸리티 함수들을 임포트
from python_server_utils import (
    build_gpt_prompt, call_gpt_api, call_custom_llm_api,
    load_clip_model, predict_with_clip
)
from video_processor import extract_frames_opencv

app = Flask(__name__)

# --- 서버 시작 시 CLIP 모델 로드 ---
clip_classifier_model_global = load_clip_model(
    CLIP_MODEL_PATH,
    num_classes=len(FINAL_CATEGORIES),
    device=DEVICE
)

# --- MongoDB 클라이언트 초기화 ---
mongo_client = None
try:
    mongo_client = MongoClient(MONGO_URI)
    # 실제 연결 테스트 (선택 사항이지만 연결 오류를 일찍 감지하는 데 도움 됨)
    mongo_client.admin.command('ping')
    print("MongoDB에 성공적으로 연결되었습니다!")
except ConnectionFailure as e:
    print(f"MongoDB 연결 실패: {e}")
    # 프로덕션 환경에서는 앱을 종료하거나 오류를 로깅하고 적절히 처리
except Exception as e:
    print(f"MongoDB 클라이언트 초기화 중 예상치 못한 오류 발생: {e}")

# --- 모델 로드 및 MongoDB 초기화 끝 ---


@app.route('/process_metadata', methods=['POST'])
def process_metadata():
    data = request.json
    if not data or 'meta_filepath' not in data:
        return jsonify({"status": "error", "message": "'meta_filepath' not provided"}), 400

    meta_filepath = os.path.join(META_DIR, data['meta_filepath'])
    print(f"Flask가 메타데이터 파일을 찾을 경로: {meta_filepath}") # 이 라인 출력 확인
    
    try:
        with open(meta_filepath, 'r', encoding='utf-8') as f:
            full_meta_data = json.load(f)
        
        meta_data = full_meta_data.get("data", {})
        if not meta_data:
            print(f"오류: 메타데이터 파일 '{meta_filepath}'에서 'data' 필드를 찾을 수 없거나 비어 있습니다.")
            return jsonify({"status": "error", "message": "Metadata 'data' field is missing or empty"}), 500

    except FileNotFoundError:
        print(f"오류: Flask가 '{meta_filepath}' 파일을 찾을 수 없습니다.") # 이 라인 출력 확인
        return jsonify({"status": "error", "message": "Metadata file not found"}), 404
    except json.JSONDecodeError:
        print(f"오류: '{meta_filepath}' 파일의 JSON 형식이 올바르지 않습니다.")
        return jsonify({"status": "error", "message": "Invalid JSON in metadata file"}), 500
    except Exception as e:
        print(f"서버 처리 중 예상치 못한 오류 발생: {str(e)}")
        return jsonify({"status": "error", "message": f"Server processing error: {str(e)}"}), 500

    # 이 부분이 실행되면 제대로 데이터를 읽어온 것입니다.
    title = meta_data.get("title", "(제목 없음)")
    chat_messages = meta_data.get("chatMessages", [])
    whisper_transcript = meta_data.get("transcript", "")
    

    # 라이브 커머스 여부 판단을 위한 원시 콘텐츠 생성
    raw_content_is_live_commerce = f"제목: {title}\n"
    
    if chat_messages:
        chat_content = '\n'.join(chat_messages) + '\n'
        raw_content_is_live_commerce += f"📌 채팅: {chat_content}"
    
    if whisper_transcript:
        transcript_content = whisper_transcript + '\n'
        raw_content_is_live_commerce += f"📌 자막: {transcript_content}"
        
    raw_content_is_live_commerce += "이 방송은 제품을 판매하는 라이브커머스인가요?\n[답]:"

    is_live_commerce = False
    llm_generated_text = ""
    try:
        llm_generated_text = call_custom_llm_api(FULL_LLM_API_URL, raw_content_is_live_commerce)
        if llm_generated_text.lower().startswith("네") or llm_generated_text.lower().startswith("예"):
            is_live_commerce = True
            
    except requests.exceptions.ConnectionError as e:
        return jsonify({"status": "error", "message": f"Failed to connect to LLM API: {e}"}), 500
    except requests.exceptions.Timeout:
        return jsonify({"status": "error", "message": "LLM API request timed out"}), 504
    except requests.exceptions.HTTPError as e:
        return jsonify({
            "status": "error",
            "message": f"LLM API call failed with status {e.response.status_code}",
            "llm_response": e.response.text
        }), e.response.status_code
    except Exception as e:
        return jsonify({"status": "error", "message": f"Unexpected error: {str(e)}"}), 500

    extracted_frames_paths = []
    gpt_product_category = "none"
    clip_predicted_category = "none"
    mongo_insert_status = "not_attempted"

    if is_live_commerce:
        video_filename = meta_data.get("filename")
        if video_filename:
            video_filepath = os.path.join(VIDEO_DIR, video_filename)
            extracted_frames_paths = extract_frames_opencv(video_filepath, IMAGE_OUTPUT_BASE_DIR, num_frames=6)
        
        try:
            gpt_prompt_product = build_gpt_prompt(title, chat_messages, whisper_transcript)
            gpt_product_category = call_gpt_api(client, gpt_prompt_product)
            
            # gpt_product_category가 'none'이면 빈 문자열로 대체
            clip_input_text = gpt_product_category if gpt_product_category.lower() != 'none' else ""

            if clip_classifier_model_global and extracted_frames_paths:
                try:
                    import time
                    start = time.time()
                    clip_predicted_category = predict_with_clip(
                        clip_classifier_model_global,
                        extracted_frames_paths,
                        clip_input_text, # 결정된 텍스트 입력 사용
                        FINAL_CATEGORIES,
                        DEVICE
                    )
                    end = time.time()
                    duration = end-start
                    print('걸린 초:', duration)
                    print(clip_predicted_category)
                except Exception as clip_e:
                    print(f"Error during CLIP prediction: {clip_e}")
                    clip_predicted_category = f"clip_prediction_error: {str(clip_e)}"
                    
            else:
                # CLIP 모델이 로드되지 않았거나 프레임이 추출되지 않은 경우
                if not clip_classifier_model_global:
                    clip_predicted_category = "clip_model_not_loaded"
                elif not extracted_frames_paths:
                    clip_predicted_category = "no_frames_extracted"
                print(f"CLIP classification skipped. Reason: {clip_predicted_category}")
                
        except Exception as e:
            gpt_product_category = f"gpt_error: {str(e)}"
            clip_predicted_category = "skipped_due_to_gpt_error"
        
        # --- 라이브 커머스일 경우 MongoDB에 저장 ---
        if mongo_client:
            try:
                db = mongo_client[MONGO_DB_NAME]
                collection = db[MONGO_COLLECTION_NAME]

                # 몽고DB 저장용 데이터 구성 (이미지 JSON 구조와 유사하게)
                # _id는 MongoDB가 자동으로 생성하므로 명시적으로 넣지 않습니다.
                # dates 필드는 현재 시간으로 설정
                current_time_utc = datetime.now(timezone.utc)
                
                # Constructing the document with only the desired fields
                tiktok_doc = {
                    "liveUrl": meta_data.get("liveUrl"), 
                    "channelUrl": meta_data.get("channelUrl"),
                    "title": title,
                    "thumbnail": meta_data.get("thumbnail"), 
                    "seller": meta_data.get("seller"), 
                    "platform": meta_data.get("platform", "tiktok"), 
                    "category": clip_predicted_category if clip_predicted_category != "none" else gpt_product_category, 
                    "dates": [current_time_utc.isoformat(timespec='milliseconds').replace('+00:00', 'Z')], 
                }
                
                result = collection.insert_one(tiktok_doc)
                mongo_insert_status = f"success_id_{result.inserted_id}"
                print(f"MongoDB에 문서 삽입 성공: {result.inserted_id}")

            except ConnectionFailure as e:
                mongo_insert_status = f"mongo_connection_error: {e}"
                print(f"MongoDB 삽입 실패 (연결 오류): {e}")
            except PyMongoError as e:
                mongo_insert_status = f"mongo_error: {e}"
                print(f"MongoDB 삽입 실패 (PyMongo 오류): {e}")
            except Exception as e:
                mongo_insert_status = f"mongo_unexpected_error: {e}"
                print(f"MongoDB 삽입 중 예상치 못한 오류 발생: {e}")
        else:
            mongo_insert_status = "mongo_client_not_initialized"
            print("MongoDB 클라이언트가 초기화되지 않아 데이터 삽입을 건너뜀.")

    return jsonify({
        "status": "success",
        "message": "Metadata processed, LLM API called, video frames and product category processed",
        "is_live_commerce": is_live_commerce,
        "llm_generated_text": llm_generated_text,
        "gpt_product_category": gpt_product_category,
        "clip_predicted_category": clip_predicted_category,
        "extracted_frames": extracted_frames_paths,
        "mongo_insert_status": mongo_insert_status # MongoDB 삽입 결과 추가
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)