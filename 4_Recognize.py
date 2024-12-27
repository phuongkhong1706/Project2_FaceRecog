import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import math
import pandas as pd
from datetime import datetime
import openpyxl

# Cài đặt kích thước frame
FRAME_SIZE = (640, 480)
IMG_PATH = './data/test_images'
DATA_PATH = './data'
OUTPUT_FILE = "C://Users//FUJITSU//OneDrive//Desktop//Diemdanh.xlsx"  # Đường dẫn file Excel

# Cài đặt kích thước khung giữa màn hình
CENTER_BOX_SIZE = (300, 300)  # Tăng kích thước khung


def euclidean_dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def align(img, landmarks):
    left_eye = tuple(landmarks[0])
    right_eye = tuple(landmarks[1])

    if left_eye[1] > right_eye[1]:
        optimal_eye = (right_eye[0], left_eye[1])
        rotation_direction = -1
    else:
        optimal_eye = (left_eye[0], right_eye[1])
        rotation_direction = 1

    a = euclidean_dist(left_eye, optimal_eye)
    b = euclidean_dist(right_eye, optimal_eye)
    c = euclidean_dist(left_eye, right_eye)

    cos_angle = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
    angle = np.arccos(cos_angle) * 180 / np.pi

    if rotation_direction == -1:
        angle = 90 - angle

    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, rotation_direction * angle, 1.0)
    aligned_img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    return aligned_img


def trans(img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    return transform(img)


def load_faceslist(device):
    embeds = torch.load(f"{DATA_PATH}/faceslist{'CPU' if device == 'cpu' else ''}.pth")
    names = np.load(f"{DATA_PATH}/usernames.npy")
    return embeds, names


def inference(model, face, local_embeds, device, threshold=3):
    face_embed = model(trans(face).to(device).unsqueeze(0))
    norm_diff = face_embed.unsqueeze(-1) - torch.transpose(local_embeds, 0, 1).unsqueeze(0)
    norm_score = torch.sum(norm_diff ** 2, dim=1)
    min_dist, embed_idx = torch.min(norm_score, dim=1)

    if min_dist.item() > threshold:
        return -1, min_dist.item()
    else:
        return embed_idx.item(), min_dist.item()


def extract_face(box, img, landmarks, margin=20):
    face_size = 160
    img_h, img_w = FRAME_SIZE

    margin = [
        margin * (box[2] - box[0]) / (face_size - margin),
        margin * (box[3] - box[1]) / (face_size - margin),
    ]
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, img_w)),
        int(min(box[3] + margin[1] / 2, img_h)),
    ]
    face_img = img[box[1]:box[3], box[0]:box[2]]

    if landmarks is not None:
        aligned_face = align(face_img, landmarks)
    else:
        aligned_face = face_img

    resized_face = cv2.resize(aligned_face, (face_size, face_size), interpolation=cv2.INTER_AREA)
    return Image.fromarray(resized_face)


def save_to_excel(name):
    # Lấy thời gian hiện tại
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Tạo hoặc cập nhật file Excel
    try:
        df = pd.read_excel(OUTPUT_FILE)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['MSSV', 'Date-Time'])

    new_entry = {'MSSV': name, 'Date-Time': current_time}
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Đã lưu thông tin nhận diện: {name}, {current_time}")


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = InceptionResnetV1(classify=False, pretrained="casia-webface").to(device)
    model.eval()

    mtcnn = MTCNN(thresholds=[0.7, 0.7, 0.8], keep_all=True, device=device)
    embeddings, names = load_faceslist(device)

    root = tk.Tk()
    root.title("Face Recognition")

    canvas = tk.Canvas(root, width=FRAME_SIZE[0], height=FRAME_SIZE[1])
    canvas.pack()

    label = tk.Label(root, text="Vui lòng đưa mặt vào khung", font=("Arial", 16))
    label.pack()

    restart_button = tk.Button(root, text="Khởi động lại", font=("Arial", 12), command=lambda: restart())
    restart_button.pack()

    cap = cv2.VideoCapture(0)
    recognized = False

    def restart():
        nonlocal recognized
        recognized = False
        label.config(text="Vui lòng đưa mặt vào khung")
        update_frame()  # Tiếp tục luồng cập nhật

    def update_frame():
        nonlocal recognized

        ret, frame = cap.read()
        if ret:
            # Vẽ khung chính giữa
            center_x, center_y = FRAME_SIZE[0] // 2, FRAME_SIZE[1] // 2
            box_x1 = center_x - CENTER_BOX_SIZE[0] // 2
            box_y1 = center_y - CENTER_BOX_SIZE[1] // 2
            box_x2 = center_x + CENTER_BOX_SIZE[0] // 2
            box_y2 = center_y + CENTER_BOX_SIZE[1] // 2
            frame = cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (255, 0, 0), 2)

            boxes, _, landmarks = mtcnn.detect(frame, landmarks=True)
            if boxes is not None and not recognized:
                for i, box in enumerate(boxes):
                    bbox = list(map(int, box.tolist()))
                    face_landmarks = landmarks[i] if landmarks is not None else None

                    # Kiểm tra khuôn mặt có nằm trong khung không
                    if bbox[0] >= box_x1 and bbox[1] >= box_y1 and bbox[2] <= box_x2 and bbox[3] <= box_y2:
                        face = extract_face(bbox, frame, face_landmarks)
                        idx, score = inference(model, face, embeddings, device)

                        if idx != -1:
                            recognized = True
                            name = names[idx]
                            print(f"Nhận diện được: {name}")
                            label.config(text=f"Nhận diện: {name}")
                            save_to_excel(name)  # Ghi vào file Excel
                            break
                        else:
                            frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                            frame = cv2.putText(frame, "Unknown", (bbox[0], bbox[1] - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            label.config(text="Không nhận diện được")

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(image=img)
            canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            canvas.image = img_tk

        if not recognized:  # Chỉ tiếp tục gọi update_frame nếu chưa nhận diện
            root.after(10, update_frame)

    update_frame()
    root.mainloop()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()