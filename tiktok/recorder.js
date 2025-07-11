// recorder.js
import { spawn } from "child_process";
import fs from "fs";
import path from "path";
export function recordScreen(filename, durationSec, VIDEO_DIR = "recordings", OFFSET_X = 0, OFFSET_Y = 0, shouldSave = true) {
  const filePath = shouldSave
    ? path.join(VIDEO_DIR, filename)
    : process.platform === "win32" ? "NUL" : "/dev/null";

  const ffArgs = [
    "-y",
    "-f", "gdigrab",
    "-framerate", "8",
    "-offset_x", OFFSET_X, "-offset_y", OFFSET_Y,
    "-video_size", "1920x1080", "-i", "desktop",
    "-an",
    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
    "-movflags", "+faststart",
    "-t", durationSec.toString(),
    filePath
  ];

  return new Promise((resolve, reject) => {
    const p = spawn("ffmpeg", ffArgs);
    p.stderr.on("data", d => process.stdout.write(d.toString()));
    p.on("close", code => {
      if (code === 0 && shouldSave) {
        resolve(filePath);
      } else if (code === 0 && !shouldSave) {
        resolve(null); // 저장 안 한 경우에도 성공 처리
      } else {
        reject(new Error("ffmpeg exited with code " + code));
      }
    });
  });
}

export function recordScreen2(filename, durationSec, VIDEO_DIR = "recordings2", OFFSET_X = 0, OFFSET_Y = 0) {
  const filePath = path.join(VIDEO_DIR, filename);
  const ffArgs = [
    "-y",
    "-f", "dshow", "-i", 'audio=CABLE Output(VB-Audio Virtual Cable)',
    "-f", "gdigrab",
    "-framerate", "8",
    "-offset_x", OFFSET_X, "-offset_y", OFFSET_Y,
    "-video_size", "1920x1080", "-i", "desktop",
    "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
    "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
    "-t", durationSec.toString(),
    filePath
  ];

  return new Promise((resolve, reject) => {
    const p = spawn("ffmpeg", ffArgs);
    p.stderr.on("data", d => process.stdout.write(d.toString()));
    p.on("close", code => {
      if (code === 0 && fs.existsSync(filePath)) {
        resolve(filePath);
      } else {
        reject(new Error("ffmpeg exited with code " + code));
      }
    });
  });
}