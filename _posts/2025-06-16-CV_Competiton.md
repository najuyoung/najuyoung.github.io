---
title: (AI Boot Camp) CV and Competition info
date: 2025-06-16 12:00:01 +0900
categories: [ BOOTCAMP, KERNEL_ACADEMY ]
tags: [ 'AIBootCamp' ]
toc: true
comments: false
mermaid: true
math: true
---

# 📄 Document Type Classification 대회: 실무 문서 분류 AI 모델 개발 가이드

## 📦 사용하는 python package

```python
# 핵심 라이브러리
torch==2.1.0
torchvision==0.16.0
timm==0.9.10
albumentations
pandas
numpy
scikit-learn
PIL
tqdm

# 실험 관리 (선택사항)
wandb
```


## 🚀 TL;DR

- **Document Type Classification**은 17종의 문서 이미지를 자동으로 분류하는 Computer Vision 대회다[^1_1]
- 금융, 의료, 보험 등 실제 산업 현장에서 필요한 문서 디지털화 작업을 AI로 자동화하는 것이 목표다[^1_1]
- 총 1,570장의 학습 데이터와 3,140장의 테스트 데이터로 구성되며, 테스트 데이터에는 현실적인 노이즈가 포함되어 있다[^1_1]
- **ResNet34** 모델과 **Albumentations** 라이브러리를 사용한 베이스라인 코드를 제공한다[^1_2]
- 평가 지표는 **Macro F1 Score**를 사용하여 클래스 불균형 문제를 고려한다[^1_1][^1_3]
- EDA, 데이터 증강, 앙상블 등 다양한 성능 향상 기법을 적용할 수 있다[^1_1]


## 📓 실습 Jupyter Notebook

실습 코드는 다음 깃허브 링크에서 확인할 수 있습니다:

- **베이스라인 코드**: `Image-Classification-beiseurain-kodeu-haeseol.ipynb`
- **대회 소개 자료**: `Image-Classification-daehoe-sogae.pdf`

---

## 🎯 대회 개요

### Document Type Classification이란?

**Document Type Classification**은 다양한 문서 이미지를 입력받아 해당 문서의 타입을 자동으로 분류하는 컴퓨터 비전 태스크다[^1_4][^1_5]. 이는 실제 산업 현장에서 아날로그 문서 데이터의 디지털화를 위해 필수적인 기술이다[^1_6].

### 분류 대상 문서 타입 (17종)

이번 대회에서 분류해야 하는 문서 타입은 다음과 같다[^1_1]:

- **금융 관련**: 계좌번호, 진료비영수증, 약제비 영수증, 진료비 납입 확인서
- **신분증명**: 여권, 운전면허증, 주민등록증
- **차량 관련**: 자동차 번호판, 자동차 계기판, 자동차 등록증
- **의료 관련**: 처방전, 통원/진료 확인서, 입퇴원 확인서, 진단서, 소견서
- **기타**: 이력서, 건강보험 임신출산 진료비 지급 신청서

> 실제 산업 현장에서는 문서 데이터가 금융, 의료, 보험, 물류 등 모든 도메인에 존재하며, 많은 회사들이 아날로그 데이터의 디지털화를 통한 디지털 혁신을 추진하고 있다.
{: .prompt-tip}

## 📊 데이터셋 구성

### 학습 데이터

- **총 1,570장**의 문서 이미지
- **17개 클래스**로 구성
- 각 클래스별로 **46~100장**의 이미지 포함[^1_1]


### 테스트 데이터

- **총 3,140장**의 문서 이미지
- 실제 현실 세계의 노이즈를 반영한 다양한 **augmentation**이 적용됨[^1_1]
- 구겨진 문서, 물에 젖은 문서, 빛번짐 등의 현실적인 왜곡 포함[^1_1]


### 데이터 구조

```
datasets_fin/
├── train.csv          # 학습 이미지 이름과 클래스 매핑
├── meta.csv           # 클래스 이름과 인덱스 매핑  
├── sample_submission.csv  # 제출용 템플릿
├── train/             # 학습 이미지 폴더
└── test/              # 테스트 이미지 폴더
```


## 📈 평가 지표: Macro F1 Score

### F1 Score의 개념

**F1 Score**는 정밀도(Precision)와 재현율(Recall)의 조화평균으로 계산되는 분류 성능 지표다[^1_7][^1_8]. 클래스 불균형 문제가 있을 때 모델의 성능을 정확하게 평가할 수 있다[^1_9].

$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$

### Macro F1 Score 계산 방법

**Macro F1 Score**는 각 클래스별로 개별적으로 계산된 F1 Score의 단순 평균이다[^1_3][^1_7]. 이는 클래스 빈도에 관계없이 모든 클래스를 동등하게 취급한다[^1_8].

```python
# Macro F1 Score 계산 예시
from sklearn.metrics import f1_score

def macro_f1_score(y_true, y_pred, n_classes):
    f1_scores = []
    
    for c in range(n_classes):
        y_true_c = (y_true == c)
        y_pred_c = (y_pred == c)
        f1_c = f1_score(y_true_c, y_pred_c)
        f1_scores.append(f1_c)
    
    return np.mean(f1_scores)
```


### Confusion Matrix 이해

**Confusion Matrix**는 실제 클래스와 예측 클래스를 비교하여 모델의 성능을 시각화하는 도구다[^1_10]. 다음 네 가지 요소로 구성된다[^1_10]:

- **TP (True Positive)**: 실제 positive를 positive로 예측 (정답)
- **FP (False Positive)**: 실제 negative를 positive로 예측 (오답)
- **FN (False Negative)**: 실제 positive를 negative로 예측 (오답)
- **TN (True Negative)**: 실제 negative를 negative로 예측 (정답)


## 🔧 베이스라인 모델 구현

### 모델 아키텍처: ResNet34

**ResNet (Residual Network)**은 깊은 신경망에서 발생하는 기울기 소실 문제를 해결하기 위해 개발된 CNN 아키텍처다[^1_11]. **Skip Connection**을 통해 기울기가 쉽게 전파될 수 있도록 설계되었다[^1_11].

```python
import timm
import torch.nn as nn

# ResNet34 모델 로드
model = timm.create_model(
    'resnet34',
    pretrained=True,
    num_classes=17  # 17개 문서 클래스
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)
```


### 데이터 전처리 및 증강

**Albumentations** 라이브러리를 사용하여 이미지 전처리와 데이터 증강을 수행한다[^1_12]. 이는 70개 이상의 다양한 증강 기법을 제공하며 PyTorch와 완벽하게 호환된다[^1_12].

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 학습용 변환
trn_transform = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# 테스트용 변환
tst_transform = A.Compose([
    A.Resize(height=224, width=224),
    A.Normalize(mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```


### 데이터셋 클래스 구현

```python
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, csv, path, transform=None):
        self.df = pd.read_csv(csv).values
        self.path = path
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        name, target = self.df[idx]
        img = np.array(Image.open(os.path.join(self.path, name)))
        
        if self.transform:
            img = self.transform(image=img)['image']
        
        return img, target
```


## 🚀 모델 학습 및 추론

### 학습 루프 구현

```python
def train_one_epoch(loader, model, optimizer, loss_fn, device):
    model.train()
    train_loss = 0
    preds_list = []
    targets_list = []
    
    pbar = tqdm(loader)
    for image, targets in pbar:
        image = image.to(device)
        targets = targets.to(device)
        
        model.zero_grad(set_to_none=True)
        
        preds = model(image)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())
        targets_list.extend(targets.detach().cpu().numpy())
        
        pbar.set_description(f"Loss: {loss.item():.4f}")
    
    # 성능 지표 계산
    train_loss /= len(loader)
    train_acc = accuracy_score(targets_list, preds_list)
    train_f1 = f1_score(targets_list, preds_list, average='macro')
    
    return {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_f1": train_f1,
    }
```


### 추론 및 결과 저장

```python
# 모델을 평가 모드로 설정
model.eval()
preds_list = []

for image, _ in tqdm(tst_loader):
    image = image.to(device)
    
    with torch.no_grad():
        preds = model(image)
        preds_list.extend(preds.argmax(dim=1).detach().cpu().numpy())

# 결과 저장
pred_df = pd.DataFrame(tst_dataset.df, columns=['ID', 'target'])
pred_df['target'] = preds_list
pred_df.to_csv("submission.csv", index=False)
```


## 🎨 성능 향상 기법

### EDA (Exploratory Data Analysis)

**탐색적 데이터 분석**을 통해 데이터의 특성을 파악하고 모델 개선 방향을 설정할 수 있다[^1_1]:

- **이미지 시각화**: 학습 데이터는 clean하고 테스트 데이터는 noisy한 특성 파악
- **회전 문제**: 테스트 이미지에 회전된 경우가 많아 rotation augmentation 필요
- **이질적 이미지**: 차량 관련 이미지들이 다른 문서들과 상이한 특성을 보임
- **크기 분포**: 문서 이미지의 크기 분포를 파악하여 적절한 resize 전략 수립


### 데이터 증강 기법

**Document-specific augmentation**을 위해 **Augraphy** 라이브러리 활용을 고려할 수 있다[^1_1]. 이는 문서 이미지에 특화된 다양한 증강 기법을 제공한다[^1_1]:

- **BleedThrough**: 종이 뒷면 내용이 비치는 효과
- **BadPhotoCopy**: 복사기 품질 저하 효과
- **BookBinding**: 책 제본으로 인한 왜곡
- **ColorShift**: 색상 변화 효과
- **InkMottling**: 잉크 번짐 효과


### 앙상블 기법

**Ensemble Methods**는 여러 모델의 예측을 결합하여 단일 모델보다 더 나은 성능을 달성하는 기법이다[^1_13]. 다음과 같은 방법들을 적용할 수 있다[^1_1]:

- **Model Ensemble**: 다양한 아키텍처 조합 (ResNet, EfficientNet, ViT)
- **Data Ensemble**: 다양한 이미지 크기 및 증강 기법 적용
- **Seed Ensemble**: 동일한 모델을 다른 random seed로 학습
- **Soft Voting**: 각 모델의 확률값을 평균하여 최종 예측

```python
# 앙상블 예시
def ensemble_predict(models, data_loader, device):
    all_preds = []
    
    for model in models:
        model.eval()
        preds = []
        
        with torch.no_grad():
            for images, _ in data_loader:
                images = images.to(device)
                outputs = model(images)
                preds.extend(torch.softmax(outputs, dim=1).cpu().numpy())
        
        all_preds.append(np.array(preds))
    
    # Soft voting
    ensemble_preds = np.mean(all_preds, axis=0)
    final_preds = np.argmax(ensemble_preds, axis=1)
    
    return final_preds
```


## 🔬 실험 관리: Weights \& Biases

### W\&B 활용법

**Weights \& Biases**는 머신러닝 실험을 체계적으로 관리하고 시각화하는 플랫폼이다[^1_14]. 다음과 같은 기능을 제공한다[^1_14]:

- **실험 추적**: 손실값, 정확도, 하이퍼파라미터 자동 기록
- **실시간 모니터링**: 학습 과정을 실시간으로 시각화
- **비교 분석**: 여러 실험 결과를 한눈에 비교
- **협업 도구**: 팀원들과 실험 결과 공유

```python
import wandb

# W&B 초기화
wandb.init(
    project="document-classification",
    config={
        "learning_rate": 1e-3,
        "batch_size": 32,
        "epochs": 10,
        "model": "resnet34"
    }
)

# 학습 중 메트릭 로깅
for epoch in range(epochs):
    metrics = train_one_epoch(train_loader, model, optimizer, loss_fn, device)
    
    wandb.log({
        "epoch": epoch,
        "train_loss": metrics["train_loss"],
        "train_acc": metrics["train_acc"],
        "train_f1": metrics["train_f1"]
    })
```


## 🎯 최적화 전략

### 하이퍼파라미터 튜닝

- **Learning Rate**: Cosine Annealing 스케줄러 사용
- **Batch Size**: GPU 메모리에 따라 조정
- **Image Size**: 224x224에서 시작하여 점진적으로 증가
- **Augmentation**: 테스트 데이터 특성에 맞는 증강 기법 선택


### 모델 선택 기준

- **경량 모델**: MobileNetV2, EfficientNet-B0
- **중간 성능**: ResNet50, EfficientNet-B3
- **고성능 모델**: EfficientNet-B7, Vision Transformer

> 문서 분류 태스크의 특성상 정확한 텍스트 정보 인식보다는 문서의 전체적인 레이아웃과 구조를 파악하는 것이 중요하다. 따라서 적절한 receptive field를 가진 CNN 모델이 효과적일 수 있다.
{: .prompt-tip}

## 🏆 최종 제출 및 검증

### 교차 검증

```python
from sklearn.model_selection import StratifiedKFold

# 5-Fold 교차 검증
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Training Fold {fold+1}")
    
    # 폴드별 학습 및 검증
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    
    # 모델 학습 및 평가
    model = train_model(train_dataset, val_dataset)
    
    # 검증 점수 기록
    val_score = evaluate_model(model, val_dataset)
    print(f"Fold {fold+1} Validation F1: {val_score:.4f}")
```


### 최종 제출 파일 생성

```python
# 최종 예측 결과 생성
final_predictions = ensemble_predict(best_models, test_loader, device)

# 제출 파일 생성
submission_df = pd.read_csv("sample_submission.csv")
submission_df['target'] = final_predictions
submission_df.to_csv("final_submission.csv", index=False)

print("제출 파일이 생성되었습니다!")
print(f"예측 결과 분포:\n{pd.Series(final_predictions).value_counts()}")
```


---

## 💡 핵심 인사이트

**Document Type Classification**은 단순한 이미지 분류를 넘어 실제 산업 현장의 문제를 해결하는 실용적인 AI 기술이다[^1_6]. 성공적인 모델 개발을 위해서는 다음과 같은 요소들이 중요하다:

- **도메인 특화 전처리**: 문서 이미지의 특성을 고려한 적절한 전처리 및 증강
- **클래스 불균형 대응**: Macro F1 Score를 고려한 학습 전략 수립
- **현실적 노이즈 처리**: 테스트 데이터의 노이즈에 강건한 모델 설계
- **체계적 실험 관리**: W\&B 등을 활용한 효율적인 실험 추적

이러한 접근 방식을 통해 금융, 의료, 보험 등 다양한 산업 분야에서 활용 가능한 실용적인 문서 분류 시스템을 구축할 수 있다.

<div style="text-align: center">⁂</div>

[^1_1]: Image-Classification-daehoe-sogae.pdf

[^1_2]: Image-Classification-beiseurain-kodeu-haeseol.ipynb

[^1_3]: https://data-minggeul.tistory.com/11

[^1_4]: https://paperswithcode.com/task/document-image-classification

[^1_5]: https://www.docsumo.com/blogs/ocr/document-classification

[^1_6]: https://www.linkedin.com/pulse/next-generation-document-classification-exploring-vision-srinivas-rqftc

[^1_7]: https://velog.io/@e1kim/분류평가지표-Precision-Recall-F1-Macro-Micro-score

[^1_8]: https://velog.io/@nata0919/분류-성능-평가-지표-F1-Score-F-Beta-Score-Macro-F1-정리

[^1_9]: https://dacon.io/en/forum/408130

[^1_10]: https://datasciencedojo.com/blog/confusion-matrix/

[^1_11]: https://cs231n.stanford.edu/reports/2017/pdfs/12.pdf

[^1_12]: https://github.com/De30/albumentations

[^1_13]: https://www.ultralytics.com/glossary/ensemble

[^1_14]: https://www.appvizer.com/artificial-intelligence/monitoringofexperiments/weights-biases

[^1_15]: https://www.altexsoft.com/blog/document-classification/

[^1_16]: https://wikidocs.net/22891

[^1_17]: https://blog.csdn.net/gitblog_00060/article/details/141013644

[^1_18]: https://www.mathworks.com/help/images/get-started-with-image-preprocessing-and-augmentation-for-deep-learning.html

[^1_19]: https://www.youtube.com/watch?v=OP8AozaEuLM

[^1_20]: https://affine.ai/learn-how-to-classify-documents-using-computer-vision-and-nlp/

[^1_21]: https://faiiry9.tistory.com/92

[^1_22]: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

