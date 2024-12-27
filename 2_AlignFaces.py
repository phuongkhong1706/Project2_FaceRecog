import os
import cv2
import mediapipe as mp
import numpy as np

# Đường dẫn đến thư mục chứa ảnh và thư mục sẽ lưu ảnh đã căn chỉnh
input_folder = r"C:\Users\FUJITSU\OneDrive\Desktop\Project2\Project2_FaceRecog\data\test_images\20216200"  # Đảm bảo thư mục này tồn tại
output_folder = r"C:\Users\FUJITSU\OneDrive\Desktop\ameo"  # Đảm bảo thư mục này tồn tại

# Tạo thư mục đầu ra nếu chưa có
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Khởi tạo MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils


# Hàm căn chỉnh khuôn mặt
def align_face(image, landmarks):
    # Lấy điểm mốc của mắt và mũi (một số điểm quan trọng để căn chỉnh)
    left_eye = landmarks[33]  # Điểm mắt trái
    right_eye = landmarks[263]  # Điểm mắt phải
    nose = landmarks[1]  # Điểm mũi

    # Tính toán độ dịch chuyển và xoay ảnh sao cho mắt ngang với trục x
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))

    # Căn chỉnh ảnh theo góc
    center = (image.shape[1] // 2, image.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_img = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return rotated_img


# Duyệt qua tất cả ảnh trong thư mục đầu vào
for filename in os.listdir(input_folder):
    img_path = os.path.join(input_folder, filename)

    if os.path.isfile(img_path):
        # Đọc ảnh
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Phát hiện khuôn mặt trong ảnh
        results = face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Lấy các điểm mốc khuôn mặt
                landmarks = [(lm.x * img.shape[1], lm.y * img.shape[0]) for lm in face_landmarks.landmark]

                # Căn chỉnh khuôn mặt
                aligned_face = align_face(img, landmarks)

                # Lưu ảnh đã căn chỉnh
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, aligned_face)
                print(f"Đã lưu ảnh căn chỉnh: {filename}")
        else:
            print(f"Không phát hiện khuôn mặt trong ảnh: {filename}")