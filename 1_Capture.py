import cv2
from facenet_pytorch import MTCNN
import torch
from datetime import datetime
import os

# Kiểm tra thiết bị
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Cấu hình đường dẫn lưu ảnh
IMG_PATH = './data/test_images/'
usr_name = input("Input your name: ")
USR_PATH = os.path.join(IMG_PATH, usr_name)

# Tạo thư mục nếu chưa tồn tại
if not os.path.exists(USR_PATH):
    os.makedirs(USR_PATH)

# # input bang video
video_path = input("Input the path to your video file: ")  # Nhập đường dẫn video
cap = cv2.VideoCapture(video_path)  # Thay camera bằng video
# input bang camera
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Kiểm tra camera
if not cap.isOpened():
    print("Cannot access the camera. Exiting...")
    exit()

# Hiển thị thông báo
print("Camera is now active. The live feed is displayed. Press 'ESC' to exit.")

# Cấu hình MTCNN
mtcnn = MTCNN(margin=20, keep_all=False, select_largest=True, post_process=False, device=device)

# Biến đếm ảnh và tần suất lưu
count = 100
leap = 1

# Cửa sổ luôn hiển thị trên đầu
WINDOW_NAME = 'Face Capturing'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)

while cap.isOpened():
    # Đọc khung hình từ camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Hiển thị khung hình camera
    cv2.imshow(WINDOW_NAME, frame)

    # Chụp và lưu ảnh khuôn mặt nếu tìm thấy
    if mtcnn(frame) is not None and leap % 2 == 1 and count > 0:  # Chụp ảnh mỗi 2 frame
        timestamp = str(datetime.now())[:-7].replace(":", "-").replace(" ", "-")
        path = os.path.join(USR_PATH, f"{timestamp}-{count}.jpg")
        mtcnn(frame, save_path=path)
        print(f"Saved: {path}")
        count -= 1
    leap += 1

    # Thoát nếu nhấn ESC
    if cv2.waitKey(1) & 0xFF == 27:
        print("Exiting...")
        break

    # Thoát khi đủ ảnh
    if count <= 0:
        print("Collected 100 images. Exiting...")
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()