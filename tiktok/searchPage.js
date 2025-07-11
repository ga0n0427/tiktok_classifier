import { delay } from "../utils.js";

export async function openSearchPage(page, url) {
  console.log(`🌐 검색 URL 접속: ${decodeURIComponent(url)}`);
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 60000 });
  await delay(5000); // 로딩 대기
}
