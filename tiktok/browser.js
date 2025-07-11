import puppeteer from "puppeteer-extra";
import StealthPlugin from "puppeteer-extra-plugin-stealth";
import fs from "fs";

puppeteer.use(StealthPlugin());

export async function launchBrowser() {
  const browser = await puppeteer.launch({
    headless: false,
    executablePath: "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    args: [
      "--window-size=1920,1080",
      "--disable-features=site-per-process",
    ],
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1920, height: 1080 });

  // 쿠키 적용
  try {
    const cookies = JSON.parse(fs.readFileSync("cookies.json", "utf8"));
    await page.setCookie(...cookies);
    console.log("🍪 쿠키 적용 완료.");
  } catch (e) {
    console.warn("⚠️ 쿠키 적용 실패:", e.message);
  }

  return { browser, page };
}
