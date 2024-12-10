import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import numpy as np
from PIL import Image
from torchvision import transforms

# Định nghĩa hàm align
def align(img, landmarks, desired_size=160, padding=0.1):
    left_eye = landmarks[0]
    right_eye = landmarks[1]

    delta_x = right_eye[0] - left_eye[0]
    delta_y = right_eye[1] - left_eye[1]
    angle = np.arctan2(delta_y, delta_x) * 180 / np.pi

    eye_dist = np.sqrt(delta_x ** 2 + delta_y ** 2)
    scale = desired_size * (1 - 2 * padding) / eye_dist

    eyes_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)

    mat = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    mat[0, 2] += desired_size / 2 - eyes_center[0]
    mat[1, 2] += desired_size / 2 - eyes_center[1]

    aligned_img = cv2.warpAffine(img, mat, (desired_size, desired_size), flags=cv2.INTER_LINEAR)
    return aligned_img

# Khởi tạo
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = InceptionResnetV1(pretrained="casia-webface", classify=False).to(device)
model.eval()
mtcnn = MTCNN(keep_all=True, device=device)

# Load danh sách khuôn mặt
embeddings = torch.load('./data/faceslist.pth')
names = np.load('./data/usernames.npy')

# Hàm tính khoảng cách
def match(face, embeddings, threshold=3):
    face_emb = model(trans(face).to(device).unsqueeze(0))
    distances = torch.sqrt(torch.sum((embeddings - face_emb) ** 2, dim=1))
    min_dist = torch.min(distances).item()
    if min_dist > threshold:
        return "Unknown"
    return names[torch.argmin(distances)]

# Nhận diện khuôn mặt
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        boxes, probs, landmarks = mtcnn.detect(frame, landmarks=True)
        if landmarks is not None:
            for i, landmark in enumerate(landmarks):
                aligned_face = align(frame, landmark)
                name = match(Image.fromarray(aligned_face), embeddings)
                # Hiển thị
                cv2.putText(frame, name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
