# 1. 파이토치로 선형 회귀 구현하기
# 1-1. 기본 셋팅
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 코드를 재실행해도 같은 결과가 나오도록 시드 설정
# torch.manual_seed(1)

# 1-2. 변수 선언
# x_train = torch.FloatTensor(([1], [2], [3]))
# y_train = torch.FloatTensor(([2], [4], [6]))

# print(x_train)
# print(x_train.shape)

# print(y_train)
# print(y_train.shape)

# 1-3. 가중치와 편향의 초기화
# 가중치 W를 0으로 초기화하고 학습을 통해 값이 변경되는 변수임을 명시
# W = torch.zeros(1, requires_grad=True)
# 가중치 W를 출력
# print(W)

# b = torch.zeros(1, requires_grad=True)
# print(b)

# 1-4. 가설 세우기
# hypothesis = x_train * W + b
# print(hypothesis)

# 1-5. 비용 함수 선언하기 => ((0 - 2)^2 + (0 - 4)^2 + (0 - 6)^2) / 3 = 18.66666.....
# cost = torch.mean((hypothesis - y_train) ** 2)
# print(cost)

# 1-6. 경사 하강법 구현하기
# optimizer = optim.SGD([W, b], lr=0.01)

# gradient를 0으로 초기화
# optimizer.zero_grad()

# 비용 함수를 미분하여 gradient 계산
# cost.backward()

# W와 b를 업데이트
# optimizer.step()

## 전체 코드
# 코드를 재실행해도 같은 결과가 나오도록 시드 설정
# torch.manual_seed(1)
#
# # 데이터
# x_train = torch.FloatTensor(([1], [2], [3]))
# y_train = torch.FloatTensor(([2], [4], [6]))
#
# # 모델 초기화
# W = torch.zeros(1, requires_grad=True)
# b = torch.zeros(1, requires_grad=True)
#
# # optimizer 설정
# optimizer = optim.SGD([W, b], lr=0.01)
#
# # 반복할 횟수 설정
# nb_epochs = 2000
# for epoch in range(nb_epochs + 1):
#
#     # H(x) 계산
#     hypothesis = x_train * W + b
#
#     # cost 계산
#     cost = torch.mean((hypothesis - y_train) ** 2)
#
#     # cost로 H(x) 개선
#     optimizer.zero_grad()
#     cost.backward()
#     optimizer.step()
#
#     # 100번마다 로그 출력
#     if epoch % 100 == 0:
#         print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
#             epoch, nb_epochs, W.item(), b.item(), cost.item()
#         ))

# 2. optimizer.zero_grad()가 필요한 이유
# import torch
# w = torch.tensor(2.0, requires_grad=True)
# print(w)
#
# nb_epochs = 20
# for epoch in range(nb_epochs + 1):
#
#   z = 2 * w
#
#   z.backward()
#   print('수식을 w로 미분한 값 : {}'.format(w.grad))

# 3. torch.manual_seed()를 하는 이유
# import torch
#
# torch.manual_seed(3)
# print('랜덤 시드가 3일 때')
# for i in range(1, 3):
#     print(torch.rand(3))
#
# torch.manual_seed(5)
# print('랜덤 시드가 5일 때')
# for i in range(1, 3):
#     print(torch.rand(1))
#
# torch.manual_seed(3)
# print('랜덤 시드가 다시 3일 때')
# for i in range(1,3):
#   print(torch.rand(1))

# 4. 자동 미분(Autograd) 실습하기
import torch

w = torch.tensor(2.0, requires_grad=True)

y = w ** 2
z = 2 * y + 5

z.backward()

print('수식을 w로 미분한 값 : {}'.format(w.grad))