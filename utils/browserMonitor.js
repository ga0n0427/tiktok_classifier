// utils/browserMonitor.js

/**
 * 브라우저 또는 페이지가 비정상적으로 종료되었는지 확인하는 유틸리티 함수
 * @param {Browser} browser Puppeteer Browser 인스턴스
 * @param {Page} page Puppeteer Page 인스턴스
 * @returns {boolean} true = 비정상 상태, false = 정상
 */
export function isBrowserOrPageDead(browser, page) {
  if (!browser || !browser.isConnected()) {
    console.warn("🔌 브라우저 연결 끊김 감지됨.");
    return true;
  }
  if (!page || page.isClosed()) {
    console.warn("📴 페이지 닫힘 감지됨.");
    return true;
  }
  return false;
}

/**
 * 페이지에 DOM 접근 가능 여부 확인 (Watchdog 감시용)
 * 실패 시 예외 발생
 */
export async function pingPage(page) {
  await page.evaluate(() => document.body?.innerText?.length);
}

/**
 * Watchdog 실행 함수 - 페이지/브라우저가 멈추면 강제종료 시도
 * @param {Page} pageToWatch 감시할 Puppeteer 페이지
 * @param {Browser} browserToMonitor 감시할 Puppeteer 브라우저
 * @param {number} intervalMs 간격(ms)
 * @returns interval ID
 */
export function startWatchdog(pageToWatch, browserToMonitor, intervalMs = 30000) {
  let watchdogId = setInterval(async () => {
    try {
      await pingPage(pageToWatch);
    } catch (e) {
      console.warn("🐶 Watchdog: 응답 없음 - 브라우저 종료 시도");
      try {
        const proc = browserToMonitor.process?.();
        if (proc?.pid) {
          console.warn(`🐶 Watchdog: SIGKILL (PID: ${proc.pid})`);
          proc.kill("SIGKILL");
        } else {
          await browserToMonitor.close();
        }
      } catch (killErr) {
        console.error("🐶 Watchdog 종료 오류:", killErr.message);
      }
      clearInterval(watchdogId);
    }
  }, intervalMs);

  return watchdogId;
}

/**
 * Watchdog 종료
 */
export function stopWatchdog(id) {
  if (id) {
    clearInterval(id);
    console.log("🧹 Watchdog 중단 완료");
  }
}
