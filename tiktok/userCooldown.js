import fs from "fs";

const COOLDOWN_FILE = "./cooldown_user_ids.json";
const COOLDOWN_DURATION_MS = 5 * 60 * 60 * 1000; // 5시간

// 🔁 JSON 파일 불러오기
export function loadCooldownData() {
  try {
    if (!fs.existsSync(COOLDOWN_FILE)) return {};
    const data = JSON.parse(fs.readFileSync(COOLDOWN_FILE, "utf8"));
    return data;
  } catch (e) {
    console.warn("⚠️ 쿨다운 파일 불러오기 실패:", e.message);
    return {};
  }
}

// 💾 JSON 파일 저장하기
export function saveCooldownData(data) {
  try {
    fs.writeFileSync(COOLDOWN_FILE, JSON.stringify(data, null, 2), "utf8");
  } catch (e) {
    console.error("❌ 쿨다운 저장 실패:", e.message);
  }
}

// ✅ 유저 사용 가능 여부
export function isUserCooldown(userId, cooldownMap) {
  const lastTime = cooldownMap[userId];
  if (!lastTime) return false;

  const now = Date.now();
  return now - lastTime < COOLDOWN_DURATION_MS;
}

// 🧼 만료된 유저 정리
export function cleanupCooldown(cooldownMap) {
  const now = Date.now();
  for (const [userId, ts] of Object.entries(cooldownMap)) {
    if (now - ts >= COOLDOWN_DURATION_MS) {
      delete cooldownMap[userId];
    }
  }
  return cooldownMap;
}
