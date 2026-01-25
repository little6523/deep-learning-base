# 1. 숫자 필기 데이터
import matplotlib.pyplot as plt # 시각화를 위한 맷플롯립
from sklearn.datasets import load_digits
digits = load_digits() # 1979개의 이미지 로드

print(digits.images[0])
print(digits.target[0])
print('전체 샘플의 수 : {}'.format(len(digits.images)))

## 전체 샘플 중에서 상위 5개 샘플 시각화
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:5]):
    plt.subplot(2, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('sample: %i' % label)

plt.show()

for i in range(5):
    print(i, '번 인덱스 샘플의 레이블 : ', digits.target[i])

print(digits.data[0])

X = digits.data
Y = digits.target

# 2. 다층 퍼셉트론 분류기 만들기
import torch
import torch.nn as nn
from torch import optim

## 모델 정의: 순차적인 레이어 구조
model = nn.Sequential(
    nn.Linear(64, 32), # 입력층: 64, 첫 번째 은닉층: 32
    nn.ReLU(),                            # 활성화 함수: ReLu
    nn.Linear(32, 16), # 입력층: ReLu
    nn.ReLU(),                            # 활성화 함수: ReLu
    nn.Linear(16, 10)  # 두 번째 은닉층: 16, 출력층: 10 (클래스의 개수)
)

# 입력 데이터 X와 레이블 Y를 텐서로 변환
X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.int64)

loss_fn = nn.CrossEntropyLoss() # 비용 함수는 소프트맥스 함수를 포함

optimizer = optim.Adam(model.parameters())

losses = []

for epoch in range(100):
    optimizer.zero_grad()      # 옵티마이저의 기울기 초기화
    y_pred = model(X)          # 순전파 연산으로 예측값 계산
    loss = loss_fn(y_pred, Y)  # 손실 함수로 비용 계산
    loss.backward()            # 역전파 연산으로 기울기 계산
    optimizer.step()           # 옵티마이저를 통해 파라미터 업데이트

    if epoch % 10 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, 100, loss.item()
        ))

    losses.append(loss.item())

plt.plot(losses)
plt.show()