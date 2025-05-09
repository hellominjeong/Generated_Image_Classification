import os
import random
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from datetime import datetime

# ==== 설정 ====
root_dir = "/mnt/data1/pair/mj228/tomesd/Generated_Image_Classification_Min-jeong-LEE/data"
target_split = random.choice(['train', 'validation'])
target_class = random.choice(['real', 'fake'])
img_dir = os.path.join(root_dir, target_split, target_class)

# 이미지 무작위 선택
img_name = random.choice(os.listdir(img_dir))
img_path = os.path.join(img_dir, img_name)
print(f"✅ 선택된 이미지: {img_path}")

# ==== 전처리 ====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
original_img = Image.open(img_path).convert('RGB')
input_tensor = transform(original_img).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

# ==== 모델 로딩 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("best_resnet_model.pth", map_location=device))
model.to(device)
model.eval()

# ==== Grad-CAM hook ====
features = None
gradients = None

def forward_hook(module, input, output):
    global features
    features = output.detach()

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0].detach()

target_layer = model.layer4[1].conv2
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# ==== 예측 및 역전파 ====
output = model(input_tensor)
pred_class = output.argmax(dim=1).item()
print(f"모델 예측 클래스: {pred_class} ({'real' if pred_class == 1 else 'fake'})")

model.zero_grad()
class_score = output[0, pred_class]
class_score.backward()

# ==== Grad-CAM 생성 ====
weights = gradients.mean(dim=[2, 3], keepdim=True)
cam = (weights * features).sum(dim=1).squeeze()
cam = F.relu(cam).cpu().numpy()
cam = cv2.resize(cam, (224, 224))
cam = (cam - cam.min()) / (cam.max() - cam.min())

# ==== 시각화 ====
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255
original = np.array(original_img.resize((224, 224))).astype(np.float32) / 255
overlay = heatmap + original
overlay = overlay / overlay.max()

# show
plt.imshow(overlay)
plt.title(f"Grad-CAM: Pred = {'real' if pred_class else 'fake'} | GT = {target_class}")
plt.axis('off')
plt.show()

# 저장
save_dir = "./gradcam_outputs"
os.makedirs(save_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(save_dir, f"gradcam_{target_class}_{pred_class}_{timestamp}.png")

plt.imsave(save_path, overlay)
print(f"✅ Grad-CAM 이미지 저장 완료: {save_path}")
