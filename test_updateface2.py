import os
import glob
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np

# Khởi tạo MTCNN
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

# Khởi tạo model nhận diện khuôn mặt
model = InceptionResnetV1(pretrained='vggface2').eval()

# Đường dẫn ảnh
IMG_PATH = './data/test_images'
DATA_PATH = './data'

embeddings = []
names = []

for usr in os.listdir(IMG_PATH):
    embeds = []
    for file in glob.glob(os.path.join(IMG_PATH, usr) + '/*.jpg'):
        img = Image.open(file)
        img_array = np.array(img)

        # Phát hiện landmarks
        boxes, probs, landmarks = mtcnn.detect(img_array, landmarks=True)

        if landmarks is not None:
            for landmark in landmarks:
                aligned_face = align(img_array, landmark)  # Landmark từ MTCNN
                aligned_face = Image.fromarray(aligned_face)

                embeds.append(model(trans(aligned_face).to(device).unsqueeze(0)))
        else:
            print(f"No landmarks detected for {file}. Skipping...")

    if len(embeds) > 0:
        embedding = torch.cat(embeds).mean(0, keepdim=True)
        embeddings.append(embedding)
        names.append(usr)

embeddings = torch.cat(embeddings)
names = np.array(names)

torch.save(embeddings, DATA_PATH + "/faceslist.pth")
np.save(DATA_PATH + "/usernames.npy", names)
