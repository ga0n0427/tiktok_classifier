// tiktok/broadcastFilter.js

let invalidCount = 0;

/**
 * 방송이 유효하지 않은 경우 판별
 * - viewerCount가 0이거나 NaN인 경우 비정상으로 간주
 */
export function isInvalidBroadcast(viewerCount) {
  return !viewerCount || isNaN(viewerCount);
}

/**
 * 비정상 방송일 경우 카운터 증가
 * 정상 방송이면 카운터 초기화
 */
export function updateInvalidCounter(isInvalid) {
  if (isInvalid) {
    invalidCount++;
  } else {
    invalidCount = 0;
  }
  return invalidCount;
}

/**
 * 현재 검색 URL을 스킵할 조건 (2회 연속 비정상)
 */
export function shouldSkipCurrentURL() {
  return invalidCount >= 2;
}

/**
 * 검색어 전환 시 카운터 초기화
 */
export function resetInvalidCounter() {
  invalidCount = 0;
}
