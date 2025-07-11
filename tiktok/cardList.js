import { delay } from "../utils.js";

export async function extractAllCards(page, maxScroll = 2) {
  const userCardPairs = [];

  for (let i = 0; i < maxScroll; i++) {
    const liveItems = await page.$$('div[data-e2e="search_live-item"]');
    const descItems = await page.$$('div[data-e2e="search-card-desc"]');

    const count = Math.min(liveItems.length, descItems.length);
    console.log(`🎯 감지된 카드: ${count}쌍 (live=${liveItems.length}, desc=${descItems.length})`);

    for (let j = 0; j < count; j++) {
      const liveCard = liveItems[j];
      const descCard = descItems[j];

      try {
        const userId = await descCard.$eval(
          'p[data-e2e="search-card-user-unique-id"]',
          el => el.innerText.trim()
        );
        const link = await liveCard.$('a[href^="https://www.tiktok.com"]');

        if (userId && link) {
          userCardPairs.push([userId, link]); // 💡 최종 [userId, 클릭 가능한 <a>]
        }
      } catch (err) {
        console.warn("⚠️ userId 또는 링크 추출 실패:", err.message);
      }
    }

    await page.evaluate(() => window.scrollBy(0, window.innerHeight));
    await delay(2000);
  }

  return userCardPairs; // [ [userId, linkHandle], ... ]
}
