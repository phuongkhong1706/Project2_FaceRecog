import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import math
import time

frame_size = (640, 480)
IMG_PATH = './data/test_images'
DATA_PATH = './data'

# Hàm tính khoảng cách Euclidean
def euclidean_dist(a, b):
    x1, y1, x2, y2 = a[0], a[1], b[0], b[1]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# Hàm căn chỉnh khuôn mặt (Aligner)
def align(img, landmarks):
    left_eye_x, left_eye_y = landmarks[0][0], landmarks[0][1]
    right_eye_x, right_eye_y = landmarks[1][0], landmarks[1][1]
    left_eye = (left_eye_x, left_eye_y)
    right_eye = (right_eye_x, right_eye_y)

    # Tính điểm mắt tối ưu cần xoay tới
    if left_eye_y > right_eye_y:
        optimal_eye = (right_eye_x, left_eye_y)
        rot = -1  # Ngược chiều kim đồng hồ
    else:
        optimal_eye = (left_eye_x, right_eye_y)
        rot = 1  # Cùng chiều kim đồng hồ

    # Tính khoảng cách Euclidean
    a = euclidean_dist(left_eye, optimal_eye)
    b = euclidean_dist(right_eye, optimal_eye)
    c = euclidean_dist(left_eye, right_eye)

    # Dùng định lý cosine
    cos_a = (b * b + c * c - a * a) / (2 * b * c)
    angle_a = np.arccos(cos_a)
    angle = (angle_a * 180) / math.pi

    if rot == -1:
        angle = 90 - angle

    # Xoay ảnh
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, rot * angle, 1.0)
    rotated_img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    return rotated_img

def trans(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    return transform(img)

# Load danh sách khuôn mặt và tên
def load_faceslist():
    if device == 'cpu':
        embeds = torch.load(DATA_PATH + '/faceslistCPU.pth')
    else:
        embeds = torch.load(DATA_PATH + '/faceslist.pth')
    names = np.load(DATA_PATH + '/usernames.npy')
    return embeds, names

# Suy luận để nhận diện khuôn mặt
def inference(model, face, local_embeds, threshold=3):
    embeds = []
    embeds.append(model(trans(face).to(device).unsqueeze(0)))
    detect_embeds = torch.cat(embeds)
    norm_diff = detect_embeds.unsqueeze(-1) - torch.transpose(local_embeds, 0, 1).unsqueeze(0)
    norm_score = torch.sum(torch.pow(norm_diff, 2), dim=1)
    min_dist, embed_idx = torch.min(norm_score, dim=1)
    if min_dist * power > threshold:
        return -1, -1
    else:
        return embed_idx, min_dist.double()

# Hàm trích xuất khuôn mặt (sửa lại để thêm align)
def extract_face(box, img, landmarks, margin=20):
    face_size = 160
    img_size = frame_size
    margin = [
        margin * (box[2] - box[0]) / (face_size - margin),
        margin * (box[3] - box[1]) / (face_size - margin),
    ]  # Tạo margin bao quanh box cũ
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, img_size[0])),
        int(min(box[3] + margin[1] / 2, img_size[1])),
    ]
    img = img[box[1]:box[3], box[0]:box[2]]

    # Thực hiện align nếu landmarks tồn tại
    if landmarks is not None:
        aligned_face = align(img, landmarks)
    else:
        aligned_face = img

    face = cv2.resize(aligned_face, (face_size, face_size), interpolation=cv2.INTER_AREA)
    face = Image.fromarray(face)
    return face


# Phần chính (Main)
if __name__ == "__main__":
    prev_frame_time = 0
    new_frame_time = 0
    power = pow(10, 6)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = InceptionResnetV1(
        classify=False,
        pretrained="casia-webface"
    ).to(device)
    model.eval()

    mtcnn = MTCNN(thresholds=[0.7, 0.7, 0.8], keep_all=True, device=device)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    embeddings, names = load_faceslist()

    while cap.isOpened():
        isSuccess, frame = cap.read()
        if isSuccess:
            boxes, _, all_landmarks = mtcnn.detect(frame, landmarks=True)
            if boxes is not None:
                for i, box in enumerate(boxes):
                    bbox = list(map(int, box.tolist()))
                    landmarks = all_landmarks[i] if all_landmarks is not None else None
                    face = extract_face(bbox, frame, landmarks)
                    idx, score = inference(model, face, embeddings)
                    if idx != -1:
                        frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)
                        score = torch.Tensor.cpu(score[0]).detach().numpy() * power
                        frame = cv2.putText(frame, names[idx] + '_{:.2f}'.format(score), (bbox[0], bbox[1]),
                                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2, cv2.LINE_8)
                    else:
                        frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)
                        frame = cv2.putText(frame, 'Unknown', (bbox[0], bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 1,
                                            (0, 255, 0), 2, cv2.LINE_8)

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = str(int(fps))
            cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
