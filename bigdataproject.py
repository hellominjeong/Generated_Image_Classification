# import os
# import shutil
# import random

# # 원본 이미지 경로
# fake_src = "/mnt/data1/pair/mj228/tomesd/tome_images_stable_diffusion"
# real_src = "/mnt/data1/pair/mj228/tomesd/resized_sampled_imagenet3"

# # 대상 디렉토리
# train_fake_dst = "/mnt/data1/pair/mj228/tomesd/Generated_Image_Classification_Min-jeong-LEE/data/train/fake"
# val_fake_dst = "/mnt/data1/pair/mj228/tomesd/Generated_Image_Classification_Min-jeong-LEE/data/validation/fake"
# train_real_dst = "/mnt/data1/pair/mj228/tomesd/Generated_Image_Classification_Min-jeong-LEE/data/train/real"
# val_real_dst = "/mnt/data1/pair/mj228/tomesd/Generated_Image_Classification_Min-jeong-LEE/data/validation/real"

# # 이미지 복사 함수
# def split_and_copy(src_dir, train_dst, val_dst, seed=42):
#     images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#     assert len(images) >= 2000, f"📛 {src_dir} 에 2000장 미만의 이미지가 있습니다 (현재 {len(images)}장)."

#     random.seed(seed)
#     random.shuffle(images)

#     train_images = images[:1600]
#     val_images = images[1600:2000]

#     for img in train_images:
#         shutil.copy(os.path.join(src_dir, img), os.path.join(train_dst, img))
#     for img in val_images:
#         shutil.copy(os.path.join(src_dir, img), os.path.join(val_dst, img))

#     print(f"✅ {src_dir} → {train_dst} (1600장), {val_dst} (400장) 복사 완료")

# # 실행
# split_and_copy(fake_src, train_fake_dst, val_fake_dst)
# split_and_copy(real_src, train_real_dst, val_real_dst)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 데이터 경로 설정
train_dir = "/mnt/data1/pair/mj228/tomesd/Generated_Image_Classification_Min-jeong-LEE/data/train"  # 훈련 데이터 경로
val_dir = "/mnt/data1/pair/mj228/tomesd/Generated_Image_Classification_Min-jeong-LEE/data/validation" # 검증 데이터 경로

# 하이퍼파라미터 설정
batch_size = 32
num_epochs = 10
learning_rate = 0.003
num_classes = 2  # fake와 real 두 가지 클래스

# 데이터 전처리: 이미지 크기 조정 및 정규화
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet 입력 크기
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet 데이터셋의 평균 및 표준편차로 정규화
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# 데이터셋 로드
image_datasets = {
    'train': datasets.ImageFolder(train_dir, data_transforms['train']),
    'val': datasets.ImageFolder(val_dir, data_transforms['val'])
}

# DataLoader 설정
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
    'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False)
}

# ResNet 모델 불러오기 (pre-trained)
model = models.resnet18(pretrained=True)

# 마지막 레이어를 재설정하여 2개의 클래스를 분류할 수 있게 설정
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# 모델을 GPU로 옮기기 (가능한 경우)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = model.to(device)

# 손실 함수 및 최적화 알고리즘 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 및 검증 함수
def train_model_with_history(model, dataloaders, criterion, optimizer, num_epochs=10):
    best_acc = 0.0
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 각 epoch에서 훈련 및 검증 단계 실행
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # 모델을 훈련 모드로 설정
            else:
                model.eval()   # 모델을 평가 모드로 설정

            running_loss = 0.0
            running_corrects = 0

            # 데이터에 대한 반복문
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 옵티마이저 초기화
                optimizer.zero_grad()

                # 순전파
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 역전파 및 최적화 (훈련 단계에서만)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 기록 저장
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

            # 최적의 모델을 저장
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

    print(f'Best val Acc: {best_acc:.4f}')

    # 최적의 가중치로 모델 로드
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

# 학습 및 검증에 대한 손실과 정확도 그래프 그리기
def plot_loss_accuracy(train_loss, val_loss, train_acc, val_acc, num_epochs):
    epochs_range = range(1, num_epochs+1)

# 모델 학습
model2, train_loss_history, val_loss_history, train_acc_history, val_acc_history = train_model_with_history(
    model, dataloaders, criterion, optimizer, num_epochs=num_epochs)

torch.save(model2.state_dict(), 'best_resnet_model.pth')



# email: mj_lee@korea.ac.kr
# if you have any questions please contact me by email.