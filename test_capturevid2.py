import cv2
from facenet_pytorch import MTCNN
import torch
from datetime import datetime
import os

# Định nghĩa thiết bị
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Đường dẫn lưu ảnh
IMG_PATH = './data/test_images/'
usr_name = input("Input your name: ").strip()
USR_PATH = os.path.join(IMG_PATH, usr_name)
os.makedirs(USR_PATH, exist_ok=True)

# Số lượng ảnh cần chụp
total_images = 100
leap = 1

# Khởi tạo MTCNN
mtcnn = MTCNN(margin=20, keep_all=False, select_largest=True, post_process=False, device=device)

# Bắt đầu chụp ảnh bằng camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting to capture images. Press 'ESC' to quit.")

while cap.isOpened() and total_images > 0:
    isSuccess, frame = cap.read()
    if isSuccess:
        # Phát hiện khuôn mặt và landmarks
        boxes, _, landmarks = mtcnn.detect(frame, landmarks=True)

        if boxes is not None and landmarks is not None and leap % 2 == 1:  # Lọc ảnh qua leap để giảm tốc độ xử lý
            for box, landmark in zip(boxes, landmarks):
                # Tạo bounding box cắt khuôn mặt
                x1, y1, x2, y2 = [int(coord) for coord in box]
                cropped_face = frame[y1:y2, x1:x2]

                # Kiểm tra kích thước hợp lệ trước khi lưu
                if cropped_face.size > 0:
                    # Lưu ảnh
                    filename = os.path.join(USR_PATH, f"{datetime.now():%Y-%m-%d-%H-%M-%S}.jpg")
                    cv2.imwrite(filename, cropped_face)
                    total_images -= 1
                    print(f"Captured image {total_images} remaining.")

                if total_images == 0:
                    break

        leap += 1

        # Hiển thị khung hình hiện tại
        cv2.imshow('Face Capturing', frame)

        # Dừng chương trình nếu nhấn 'ESC'
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        print("Không thể đọc hình ảnh từ webcam.")
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
print(f"Finished capturing. Images saved in folder: {USR_PATH}")
