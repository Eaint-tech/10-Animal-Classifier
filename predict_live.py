# predict_live.py

import torch
import cv2
from torchvision import transforms, models
import torch.nn as nn

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load class names
with open('class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Load the same architecture and load trained weights
model = models.resnet18(weights=None)  # DON'T use pretrained=True here
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load('animal_cnn.pth', map_location=device))  # Load weights
model = model.to(device)
model.eval()

# Transform (NO augmentations here — only normalization and resize!)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # Same as training
                         [0.229, 0.224, 0.225])
])

# Predict from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert and preprocess
    image = transform(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs, 1)
        label = class_names[pred.item()]

    # Display label
    cv2.putText(frame, f"Prediction: {label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
