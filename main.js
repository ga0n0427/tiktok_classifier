import { launchBrowser } from "./tiktok/browser.js";
import { openSearchPage } from "./tiktok/searchPage.js";
import { extractAllCards } from "./tiktok/cardList.js";
import { extractMetadata } from "./tiktok/metadataExtractor.js";
import { processTargetCard } from "./tiktok/cardHandler.js";
import { delay } from "./utils.js";
import {
  isInvalidBroadcast,
  updateInvalidCounter,
  resetInvalidCounter
} from "./tiktok/broadcastFilter.js";
import {
  isBrowserOrPageDead,
  startWatchdog,
  stopWatchdog
} from "./utils/browserMonitor.js";
import {
  loadCooldownData,
  saveCooldownData,
  isUserCooldown,
  cleanupCooldown
} from "./tiktok/userCooldown.js";


const SEARCH_URLS_ORIGINAL = [ // 원본 URL 목록 (const로 유지)
  "https://www.tiktok.com/search/live?q=%EC%8B%A4%EC%8B%9C%EA%B0%84%EC%87%BC%ED%95%91&t=1748002487625",
  "https://www.tiktok.com/search/live?q=%EC%95%85%EC%84%B8%EC%82%AC%EB%A6%AC&t=1748002530399",
  "https://www.tiktok.com/search/live?q=%ED%99%94%EC%9E%A5%ED%92%88&t=1748002598716",
  "https://www.tiktok.com/search/live?q=%EC%87%BC%ED%95%91&t=1747896121725",
  "https://www.tiktok.com/search/live?q=틱톡%20쇼핑",
  "https://www.tiktok.com/search/live?q=틱톡%20라이브%20쇼핑",
  "https://www.tiktok.com/search/live?q=쇼핑라이브실시간",
  "https://www.tiktok.com/search/live?q=라이브%20실시간%20쇼핑",
  "https://www.tiktok.com/search/live?q=의류%20판매%20방송",
  "https://www.tiktok.com/search/live?q=틱톡%20라이브%20옷%20판매",
  "https://www.tiktok.com/search/live?q=%EB%B7%B0%ED%8B%B0&t=1748001014147",
  "https://www.tiktok.com/search/live?q=%EC%8B%9D%ED%92%88%ED%8C%90%EB%A7%A4%20%EB%9D%BC%EC%9D%B4%EB%B8%8C%EB%B0%A9%EC%86%A1&t=1748001112316",
];

// 💡 Fisher-Yates 셔플 알고리즘
function shuffleArray(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]]; // 요소 교환
  }
  return array;
}

// 전역 유지 (이제 사용하지 않을 수도 있음)
// let currentUrlIndex = 0; // 이 변수는 더 이상 필요 없을 가능성이 높습니다.

async function run() {
  let cooldownMap = loadCooldownData(); // 🔥 시작 시 불러오기

  while (true) {
    // 🚀 매 루프 재시작 시 URL 목록 섞기
    // SEARCH_URLS_ORIGINAL 배열을 복사하여 섞고 사용
    const currentSearchUrls = shuffleArray([...SEARCH_URLS_ORIGINAL]);

    const { browser, page: initialPage } = await launchBrowser();
    let page = initialPage;
    let watchdogId = startWatchdog(page, browser);

    try {
      // 섞인 currentSearchUrls 배열을 순회
      for (let i = 0; i < currentSearchUrls.length; i++) { // currentUrlIndex 대신 i 사용
        const url = currentSearchUrls[i]; // 섞인 배열에서 URL 가져오기
        console.log(`🔍 검색 URL 접속: ${url}`);
        resetInvalidCounter();

        if (isBrowserOrPageDead(browser, page)) throw new Error("브라우저 또는 페이지 비정상 상태");

        await openSearchPage(page, url);

        // ✅ userId와 cardHandle을 함께 추출
        const userCardPairs = await extractAllCards(page, 3); // returns [ [userId, cardHandle], ... ]

        for (let j = 0; j < userCardPairs.length; j++) { // 내부 루프 변수 j로 변경
          const [userId, cardHandle] = userCardPairs[j];

          // ⏳ 쿨다운 검사
          if (isUserCooldown(userId, cooldownMap)) {
            console.log(`⏳ ${userId}는 쿨다운 중 → 스킵`);
            continue;
          }

          // 등록
          cooldownMap[userId] = Date.now();
          saveCooldownData(cooldownMap);

          console.log(`🔄 ${j + 1}번째 카드 (${userId}) 처리 중...`);

          const imageUrl = await extractMetadata(cardHandle);
          const result = await processTargetCard(page, cardHandle, imageUrl);

          if (!result || isInvalidBroadcast(result.viewerCount)) {
            const count = updateInvalidCounter(true);
            if (count >= 2) {
              console.warn("🚫 비정상 방송 2회 감지 → 다음 URL로 이동합니다");
              await page.close();
              page = await browser.newPage();
              break;
            }
          } else {
            updateInvalidCounter(false);
            // 👉 여기서 서버 전송 등 추가 가능
          }

          if (isBrowserOrPageDead(browser, page)) throw new Error("카드 처리 중 페이지 종료됨");

          await delay(2000);
        }

        // ⏲️ URL 처리 후 쿨다운 cleanup
        cooldownMap = cleanupCooldown(cooldownMap);
        saveCooldownData(cooldownMap);

        await delay(3000);
      }

      console.log("✅ 전체 URL 1회 순회 완료 → 처음부터 재시작 (무작위 순서)");
      stopWatchdog(watchdogId);
      await browser.close();
      await delay(5000);

    } catch (err) {
      console.error("💥 예외 발생:", err.message);
      stopWatchdog(watchdogId);
      try {
        await browser.close();
      } catch (e) {
        console.warn("⚠️ 브라우저 종료 실패:", e.message);
      }

      console.log("⏱️ 5초 후 전체 루프 재시작...");
      await delay(5000);
    }
  }
}


async function startMainLoop() {
  try {
    await run();
  } catch (e) {
    console.error("❌ 치명적인 에러 발생:", e.message);
    await delay(5000);
    await startMainLoop(); // 재귀 재시작
  }
}

startMainLoop();