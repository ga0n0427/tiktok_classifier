export async function extractMetadata(cardElement) {
  try {
    const imageUrl = await cardElement.$eval('img', img => img.src);
    return imageUrl;
  } catch (e) {
    console.warn("⚠️ 메타데이터 추출 실패:", e.message);
    return {};
  }
}
export async function extractChatMessages(newPage, durationMs = 3000) {
  const chat = new Set();
  const start = Date.now();
  while (Date.now() - start < durationMs) {
    try {
      const messages = await newPage.evaluate(() => Array.from(
        document.querySelectorAll('div[data-e2e="chat-message"]')
      ).map(el => el.textContent?.trim()).filter(Boolean));
      messages.forEach(m => chat.add(m));
    } catch {}
    await new Promise(r => setTimeout(r, 1000));
  }
  return chat;
}