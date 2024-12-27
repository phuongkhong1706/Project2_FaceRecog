import tkinter as tk
from datetime import datetime
import cv2
import face_recognition
import numpy as np
import pandas as pd
from PIL import Image, ImageTk

# Cài đặt thông số
FRAME_SIZE = (640, 480)
CENTER_BOX_SIZE = (250, 300)
DATA_PATH = './data'
OUTPUT_FILE = "C://Users//FUJITSU//OneDrive//Desktop//Diemdanh.xlsx"


def load_faceslist():
    """Tải embeddings và tên người dùng từ file dữ liệu."""
    try:
        encodings = np.load(f"{DATA_PATH}/face_encodings.npy", allow_pickle=True)
        names = np.load(f"{DATA_PATH}/usernames.npy", allow_pickle=True)
        return encodings, names
    except FileNotFoundError:
        print("Không tìm thấy file dữ liệu khuôn mặt.")
        return [], []


def save_to_excel(name):
    """Lưu kết quả nhận diện vào file Excel."""
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        df = pd.read_excel(OUTPUT_FILE)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['MSSV', 'Date-Time'])
    new_entry = {'MSSV': name, 'Date-Time': current_time}
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Lưu kết quả: {name} tại {current_time}")


def main():
    known_encodings, known_names = load_faceslist()

    # Giao diện Tkinter
    root = tk.Tk()
    root.title("Nhận diện khuôn mặt")
    canvas = tk.Canvas(root, width=FRAME_SIZE[0], height=FRAME_SIZE[1])
    canvas.pack()
    label = tk.Label(root, text="Vui lòng đưa mặt vào khung", font=("Arial", 16))
    label.pack()

    # Mở camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])

    recognized = False

    def update_frame():
        nonlocal recognized
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc từ camera.")
            return

        frame = cv2.flip(frame, 1)  # Lật khung hình
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Vẽ khung trung tâm
        center_x, center_y = FRAME_SIZE[0] // 2, FRAME_SIZE[1] // 2
        box_x1, box_y1 = center_x - CENTER_BOX_SIZE[0] // 2, center_y - CENTER_BOX_SIZE[1] // 2
        box_x2, box_y2 = center_x + CENTER_BOX_SIZE[0] // 2, center_y + CENTER_BOX_SIZE[1] // 2
        frame = cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (255, 0, 0), 2)

        # Nhận diện khuôn mặt
        face_locations = face_recognition.face_locations(frame_rgb)
        face_encodings = face_recognition.face_encodings(frame_rgb, face_locations)

        if face_locations and not recognized:
            for face_encoding, face_location in zip(face_encodings, face_locations):
                top, right, bottom, left = face_location
                if box_x1 <= left and box_y1 <= top and right <= box_x2 and bottom <= box_y2:
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else -1
                    if matches and matches[best_match_index]:
                        recognized = True
                        name = known_names[best_match_index]
                        label.config(text=f"Nhận diện: {name}")
                        save_to_excel(name)
                        break
                    else:
                        label.config(text="Không nhận diện được")

        # Hiển thị khung hình lên giao diện
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        canvas.image = img

        if not recognized:
            root.after(30, update_frame)

    update_frame()
    root.mainloop()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
