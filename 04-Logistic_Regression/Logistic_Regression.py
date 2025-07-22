# 1. 시그모이드 함수(Sigmoid Function)

# %matplotlib inline => 코랩에서 matplotlib 사용
# import numpy as np
# import matplotlib.pyplot as plt
#
# def sigmoid(x):
#     return 1/(1+np.exp(-x))
#
# # 1-1. W가 1이고 b가 0인 그래프
# x = np.arange(-5.0, 5.0, 0.1)
# y = sigmoid(x)
#
# plt.plot(x, y, 'g')
# plt.plot([0, 0],[1.0,0.0], ':') # 가운데 점선 추가
# plt.title('Sigmoid Function')
# plt.show()
#
# # 1-2. W값의 변화에 따른 경사도의 변화
# x = np.arange(-5.0, 5.0, 0.1)
# y1 = sigmoid(0.5*x)
# y2 = sigmoid(x)
# y3 = sigmoid(2*x)
#
# plt.plot(x, y1, 'r', linestyle='--') # W의 값이 0.5일 때
# plt.plot(x, y2, 'g') # W의 값이 1일 때
# plt.plot(x, y3, 'b', linestyle='--') # W의 값이 2일 때
# plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
# plt.title('Sigmoid Function')
# plt.show()
#
# # 1-3. b값의 변화에 따른 좌, 우 이동
# x = np.arange(-5.0, 5.0, 0.1)
# y1 = sigmoid(x+0.5)
# y2 = sigmoid(x+1)
# y3 = sigmoid(x+1.5)
#
# plt.plot(x, y1, 'r', linestyle='--') # x + 0.5
# plt.plot(x, y2, 'g') # x + 1
# plt.plot(x, y3, 'b', linestyle='--') # x + 1.5
# plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
# plt.title('Sigmoid Function')
# plt.show()

# 1-4. 시그모이드 함수를 이용한 분류
# 시그모이드 함수의 출력값은 0과 1사이
# 임계값을 0.5라고 가정 => 출력값이 0.5 이상이면 1(true), 이하면 0(false)로 판단할 수 있음

# 2. 파이토치로 로지스틱 회귀 구현하기
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

print(x_train.shape)
print(y_train.shape)

W = torch.zeros((2, 1), requires_grad=True) # 크기는 2 x 1
b = torch.zeros(1, requires_grad=True)

# 시그모이드 직접 구현
hypothesis = 1 / (1 + torch.exp(-x_train.matmul(W) + b))

print(hypothesis)

# 시그모이드 간단 구현
hypothesis = torch.sigmoid(x_train.matmul(W) + b)

print(hypothesis)
print(y_train)

loss = -(y_train[0] * torch.log(hypothesis[0]) + (1 - y_train[0]) * torch.log(1 - hypothesis[0]))
print(loss)

losses = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis))
print(losses)

cost = losses.mean()
print(cost)

F.binary_cross_entropy(hypothesis, y_train)

optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # Cost 계산
    hypothesis = torch.sigmoid(x_train.matmul(W) + b)
    cost = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis)).mean()

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print(hypothesis)

prediction = hypothesis >= torch.FloatTensor([0.5])
print(prediction)

print(W)
print(b)