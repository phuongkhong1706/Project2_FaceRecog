import os
import glob
import numpy as np
import face_recognition
from PIL import Image

IMG_PATH = './data/test_images'
DATA_PATH = './data'

# Tạo thư mục lưu trữ nếu chưa tồn tại
os.makedirs(DATA_PATH, exist_ok=True)

embeddings = []
names = []

for usr in os.listdir(IMG_PATH):
    user_images = glob.glob(os.path.join(IMG_PATH, usr) + '/*.jpg')
    user_encodings = []
    for file in user_images:
        try:
            # Đọc và chuyển đổi hình ảnh
            image = face_recognition.load_image_file(file)
            face_locations = face_recognition.face_locations(image)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            if face_encodings:
                user_encodings.append(face_encodings[0])  # Lấy encoding đầu tiên trong ảnh
        except Exception as e:
            print(f"Lỗi xử lý file {file}: {e}")
            continue

    if len(user_encodings) > 0:
        # Tính trung bình các encoding của người dùng
        avg_encoding = np.mean(user_encodings, axis=0)
        embeddings.append(avg_encoding)
        names.append(usr)

# Lưu kết quả
embeddings = np.array(embeddings)
names = np.array(names)

np.save(os.path.join(DATA_PATH, "face_encodings.npy"), embeddings)
np.save(os.path.join(DATA_PATH, "usernames.npy"), names)

print(f'Update Completed! There are {names.shape[0]} people in FaceLists.')
