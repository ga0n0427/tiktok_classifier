import os
import cv2

def extract_frames_opencv(video_path, output_base_dir, num_frames=6, target_size=(576, 576)):
    """
    OpenCV를 사용하여 비디오 파일에서 지정된 개수만큼의 프레임을 추출합니다.
    프레임은 항상 중앙을 기준으로 지정된 target_size (정사각형 또는 사각형) 크기로 자릅니다.
    별도 리사이즈 과정은 없습니다.
    """
    if not os.path.exists(video_path):
        return []

    extracted_frame_paths = []

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_duration = total_frames / fps if fps > 0 else 0

        if video_duration == 0:
            cap.release()
            return []

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(output_base_dir, video_name)
        os.makedirs(output_dir, exist_ok=True)

        interval_time = video_duration / (num_frames + 1)
        crop_w, crop_h = target_size

        for i in range(num_frames):
            time_to_extract = interval_time * (i + 1)
            cap.set(cv2.CAP_PROP_POS_MSEC, time_to_extract * 1000)
            ret, frame = cap.read()

            if ret:
                h, w, _ = frame.shape
                center_x, center_y = w // 2, h // 2

                # 좌표 계산 (경계 초과 방지)
                x1 = max(center_x - crop_w // 2, 0)
                y1 = max(center_y - crop_h // 2, 0)
                x2 = min(center_x + crop_w // 2, w)
                y2 = min(center_y + crop_h // 2, h)

                cropped_frame = frame[y1:y2, x1:x2]

                output_path = os.path.join(output_dir, f"{video_name}_frame_{i+1}.jpg")
                cv2.imwrite(output_path, cropped_frame)
                extracted_frame_paths.append(output_path)

        cap.release()
        return extracted_frame_paths

    except Exception as e:
        return []

# 테스트를 위한 코드 (직접 실행 시)
if __name__ == '__main__':
    import subprocess

    test_video = os.path.join(os.getcwd(), 'test.mp4')
    test_output_dir = os.path.join(os.getcwd(), 'test_extracted_frames')

    # 테스트 비디오 파일이 없으면 생성 (dummy file for testing)
    if not os.path.exists(test_video):
        try:
            subprocess.run(['ffmpeg', '-f', 'lavfi', '-i', 'color=c=black:s=640x480:r=30', '-t', '10', '-c:v', 'libx264', '-vf', 'format=yuv420p', test_video], check=True, capture_output=True, text=True)
        except Exception as e:
            pass

    if os.path.exists(test_video):
        extracted_frames = extract_frames_opencv(test_video, test_output_dir, num_frames=6)
        # print(f"Extracted frames: {extracted_frames}") # 필요시 주석 해제하여 확인