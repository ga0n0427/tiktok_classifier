//cardHandler.js
import { delay, sendToWhisperServer } from "../utils.js";
import { recordScreen2 } from "./recorder.js";
import { extractChatMessages } from './metadataExtractor.js'; // metadataExtractor.js에서 가져옴
import fs from "fs";
import path from "path";
import fetch from "node-fetch";

export async function processTargetCard(mainPage, cardElement, imageUrl) {
    const browser = mainPage.browser();
    const existingPages = await browser.pages();

    await cardElement.click();
    console.log("🖱️ 카드 클릭 완료");

    await delay(3000); // 새 탭이 열리거나 같은 탭에서 진행될 수 있으므로 대기

    const pagesAfter = await browser.pages();
    const newPage = pagesAfter.find(p => !existingPages.includes(p));
    const workingPage = newPage || mainPage;

    const thumbnail = imageUrl;
    const timestamp = Date.now().toString();
    const jsonPath = path.join("metadata", `${timestamp}.json`);

    try {
        await workingPage.bringToFront();

        try {
            await workingPage.waitForNavigation({ waitUntil: "domcontentloaded", timeout: 1500 });
        } catch {
            console.warn("⚠️ Navigation 타임아웃, 강제 진행");
        }

        // --- 시청자 수 추출 로직 (먼저 실행) ---
        let viewerCount = null;
        try {
            const viewerCountSelector = '#tiktok-live-main-container-id > div.tiktok-1mlfv7.e1q7cfv80 > div.tiktok-i9gxme.e1q7cfv81 > div > div.tiktok-j6lnca.e1ghyro61 > div.tiktok-ztkwg2.e1ghyro62 > div.tiktok-tk1gy.e1f21nov0 > div.tiktok-79elbk.e1w7pjwm0 > div > div > div.tiktok-10mejol.erheenz9 > div.tiktok-15iq4kv.e8in9to0 > div';
            const countTextElement = await workingPage.$(viewerCountSelector);
            let viewerCountText = null;

            if (countTextElement) {
                viewerCountText = await countTextElement.evaluate(el => el.textContent.trim());
            }

            if (viewerCountText) {
                const cleanedText = viewerCountText.replace(/[^\d.,]/g, '');
                const parsedFloat = parseFloat(cleanedText.replace(/,/g, ''));
                viewerCount = Math.floor(parsedFloat);
                console.log("👀 시청자 수:", viewerCount);
            } else {
                console.warn("시청자 수 텍스트를 찾을 수 없습니다. (방송 처리 건너뜀)");
                return null; // 시청자 수 없으면 여기서 즉시 함수 종료
            }
        } catch (error) {
            console.warn("시청자 수 추출 중 오류 발생:", error.message);
            return null; // 오류 발생 시 즉시 함수 종료
        }
        // --- 시청자 수 추출 로직 끝 ---

        // 시청자 수 추출 성공 시에만 녹화와 채팅 수집 시작
        console.log("🎥 화면 녹화 & 💬 채팅 수집 시작...");
        const recordAndChatPromise = Promise.all([
            recordScreen2(`${timestamp}.mp4`, 10, "recordings"), // 10초 녹화
            extractChatMessages(workingPage, 10000) // 10초 동안 채팅 수집
        ]);

        // 나머지 메타데이터 추출은 계속 병렬로 진행
        const metadataExtractionPromise = (async () => {
            let liveUrl = "";
            let title = "(제목 없음)";
            let seller = "(채널 없음)";
            let channelUrl = "";

            try { liveUrl = workingPage.url(); } catch (error) { console.warn("URL 추출 실패:", error.message); }
            try { title = await workingPage.$eval('div[data-e2e="user-profile-live-title"]', el => el.getAttribute("title")); } catch (error) { console.warn("제목 추출 실패:", error.message); }
            try { seller = await workingPage.$eval('h1[data-e2e="user-profile-nickname"]', el => el.textContent.trim()); } catch (error) { console.warn("판매자 추출 실패:", error.message); }
            try { channelUrl = await workingPage.$eval('div.tiktok-6gei9z.erheenz4 a', el => el.href); } catch (error) { console.warn("채널 URL 추출 실패:", error.message); }

            return { liveUrl, title, seller, channelUrl };
        })();

        // 모든 상위 레벨의 병렬 작업을 기다립니다. (녹화+채팅, 메타데이터)
        const [[_, chatSetResult], { liveUrl, title, seller, channelUrl }] = await Promise.all([
            recordAndChatPromise,
            metadataExtractionPromise
        ]);

        const chatMessages = Array.from(chatSetResult);
        console.log("✅ 모든 데이터 및 미디어 수집 완료");
        console.log("라이브 URL:", liveUrl);
        console.log("제목:", title);
        console.log("판매자:", seller);
        console.log("채널 URL:", channelUrl);
        console.log("수집된 채팅 메시지 수:", chatMessages.length);


        // 🎙️ Whisper로 음성 전송 (선택 사항, 필요시 주석 해제)
        const videoPath = path.join("recordings", `${timestamp}.mp4`);
        const transcript = await sendToWhisperServer(videoPath);
        const filename = `${timestamp}.mp4`;
        const result = {
            data: {
                liveUrl,
                channelUrl,
                title,
                thumbnail,
                seller,
                platform: "tiktok",
                chatMessages,
                transcript,
                filename,
            },
            viewerCount
        };

        fs.writeFileSync(jsonPath, JSON.stringify(result, null, 2));
        console.log("💾 메타데이터 저장 완료:", jsonPath);
        console.log("시청자 수:", viewerCount);
        console.log("📦 방송 정보:", result.data);
        console.log("1123")
        try {
        const response = await fetch("http://localhost:8000/process_metadata", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ meta_filepath: jsonPath })
        });
        console.log("22")
        const serverResult = await response.json();
        console.log("✅ 서버 처리 결과:", serverResult);
        } catch (err) {
        console.warn("❌ Flask 서버와 통신 중 오류 발생:", err.message);
        }
        return result;

    } catch (err) {
        console.warn("❌ 방송 처리 중 오류:", err.message);
        return null;
    } finally {
        if (newPage && !newPage.isClosed()) {
            await newPage.close();
            console.log("🧹 새 탭 닫기 완료");
        } else if (newPage) {
            console.log("🧹 새 탭이 이미 닫혀 있었음");
        }
        // mainPage는 닫지 않음
    }
}