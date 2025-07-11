// utils.js

/**
 * 비동기 지연 함수
 * @param {number} ms - 밀리초 단위의 지연 시간
 * @returns {Promise<void>}
 */
export const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

import axios from "axios";

export async function sendToWhisperServer(filepath) {
  try {
    const res = await axios.post("http://127.0.0.1:5000/process_metadata", {
      filepath
    }, {
      // 🚀 타임아웃을 3초 (3000 밀리초)로 설정
      timeout: 3000
    });

    return res.data?.transcript || null;
  } catch (e) {
    console.warn("⚠️ Whisper 요청 실패:", e.response?.data?.detail || e.message);
    return null;
  }
}
