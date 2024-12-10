import cv2
from facenet_pytorch import MTCNN
import torch
from datetime import datetime
import os
import numpy as np
import math

# Định nghĩa thiết bị
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Đường dẫn ảnh
IMG_PATH = './data/test_images/'
count = 100
usr_name = input("Input your name: ")
USR_PATH = os.path.join(IMG_PATH, usr_name)
os.makedirs(USR_PATH, exist_ok=True)
leap = 1

# Hàm tính khoảng cách Euclidean
def euclidean_dist(a, b):
    x1, y1, x2, y2 = a[0], a[1], b[0], b[1]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Hàm align (căn chỉnh khuôn mặt)
def align(img, landmarks):
    left_eye_x, left_eye_y = landmarks[0]
    right_eye_x, right_eye_y = landmarks[1]
    left_eye = (left_eye_x, left_eye_y)
    right_eye = (right_eye_x, right_eye_y)

    # Tính góc quay
    if left_eye_y > right_eye_y:
        optimal_eye = (right_eye_x, left_eye_y)
        rot = -1  # Ngược chiều kim đồng hồ
    else:
        optimal_eye = (left_eye_x, right_eye_y)
        rot = 1  # Cùng chiều kim đồng hồ

    # Tính khoảng cách Euclidean và góc quay
    a = euclidean_dist(left_eye, optimal_eye)
    b = euclidean_dist(right_eye, optimal_eye)
    c = euclidean_dist(left_eye, right_eye)

    cos_a = (b * b + c * c - a * a) / (2 * b * c)
    angle_a = np.arccos(cos_a)

    angle = (angle_a * 180) / math.pi
    if rot == -1:
        angle = 90 - angle

    # Quay ảnh bằng warpAffine
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, rot * angle, 1.0)
    rotated_img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated_img

# Hàm cắt và căn chỉnh khuôn mặt
def extract_and_align_face(img, box, landmarks, margin=20):
    face_size = 160  # Kích thước khuôn mặt sau cắt
    box = [
        max(0, int(box[0]) - margin),
        max(0, int(box[1]) - margin),
        min(img.shape[1], int(box[2]) + margin),
        min(img.shape[0], int(box[3]) + margin),
    ]  # Thêm khoảng margin xung quanh box
    cropped_face = img[box[1]:box[3], box[0]:box[2]]  # Cắt khuôn mặt
    aligned_face = align(cropped_face, landmarks)  # Căn chỉnh khuôn mặt
    resized_face = cv2.resize(aligned_face, (face_size, face_size))  # Resize về 160x160
    return resized_face

# Khởi tạo MTCNN
mtcnn = MTCNN(margin=20, keep_all=False, select_largest=True, post_process=False, device=device)

# Bắt đầu chụp ảnh bằng camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened() and count:
    isSuccess, frame = cap.read()
    if isSuccess:
        boxes, _, landmarks = mtcnn.detect(frame, landmarks=True)
        if boxes is not None and landmarks is not None and leap % 2:
            for box, landmark in zip(boxes, landmarks):
                # Cắt và căn chỉnh khuôn mặt
                aligned_face = extract_and_align_face(frame, box, landmark)
                # Lưu ảnh khuôn mặt đã căn chỉnh
                path = os.path.join(USR_PATH, f"{datetime.now():%Y-%m-%d-%H-%M-%S}_{count}.jpg")
                cv2.imwrite(path, aligned_face)
                count -= 1
                if count == 0:
                    break
        leap += 1

        # Hiển thị ảnh trên cửa sổ
        cv2.imshow('Face Capturing', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Nhấn ESC để thoát
            break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
