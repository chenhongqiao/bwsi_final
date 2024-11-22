import numpy as np
import os
from PIL import ImageGrab
import cv2
import time
import datetime
import torch
import random
from torchvision import models, transforms
from torch import nn

random.seed(42)

input_size = 224
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

data_transform = transforms.Compose(
    [
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

Emotion = {
    0: "Anger",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}


def start_emotion_detection(ns, camera_index=1, bbox=(300, 300, 500, 500)):
    ns.emotion = "Neutral"

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 8)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model.load_state_dict(
        torch.load(dir_path + "/ft_weights.pt", map_location=torch.device("cpu"))
    )
    model.eval()

    emotion_ave = []
    embedding_average = []
    last_time = datetime.datetime.now()

    video_capture = cv2.VideoCapture(camera_index)

    while True:
        # grabbing the image from the screen
        screen = ImageGrab.grab(bbox=bbox).convert("RGB")

        # Converts image to tensor and predict the emotion
        x = data_transform(screen)
        output = model(x.unsqueeze(0))
        _, pred = torch.max(output, 1)

        # need to get the mode of emotion within 1 second to reduce the randomness of the output
        if (time.time() - last_time.timestamp()) > 2:
            if emotion_ave:  # Check if the list is not empty
                max_emotion_label = max(set(emotion_ave), key=emotion_ave.count)
                embedding_tensor = torch.mean(torch.stack(embedding_average), dim=0)
                ns.emotion = Emotion[max_emotion_label]
                ns.emotion_embedding = embedding_tensor.detach()
            else:
                print("No emotions detected in the last 1 second.")

            # Reset the emotion_ave list after processing
            emotion_ave = []
            embedding_average = []

            last_time = datetime.datetime.now()
        else:
            emotion_ave.append(int(pred))
            embedding_average.append(output)

        processed_img = cv2.cvtColor(np.array(screen), cv2.COLOR_BGR2RGB)
        cv2.putText(
            processed_img,
            ns.emotion,
            (0, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
        )

        processed_img = cv2.resize(processed_img, dsize=processed_img.shape[1::-1])

        # processed_img_color = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        # combined_img = np.hstack(
        #    (cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), processed_img_color)
        # )

        cv2.imshow("Camera Feed", np.array(processed_img))

        # Show image and prediction
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break

    video_capture.release()


if __name__ == "__main__":
    start_emotion_detection(camera_index=1, bbox=(300, 300, 500, 500))
