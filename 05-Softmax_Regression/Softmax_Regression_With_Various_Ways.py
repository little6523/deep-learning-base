# 1. 소프트맥스 회귀의 비용 함수 구현'

# import torch
# import torch.nn.functional as F
#
# torch.manual_seed(1)
#
# # 1-1. 파이토치로 소프트맥스의 비용 함수 구현하기 (로우-레벨)
# z = torch.FloatTensor([1, 2, 3])
#
# hypothesis = F.softmax(z, dim=0)
# print(hypothesis)
# print(hypothesis.sum())
#
# z = torch.rand(3, 5, requires_grad=True)
#
# # 3개의 샘플에 대해서 5개의 클래스 중 어떤 클래스가 정답인지 예측한 결과
# hypothesis = F.softmax(z, dim=1)
# print(hypothesis)
#
# # 0이상 5미만의 정수 중에서 무작위로 뽑아 크기가 3인 1차원 텐서 생성
# y = torch.randint(5, (3,)).long()
# print(y)
#
# # 모든 원소가 0의 값을 가진 3 x 5 텐서 생성
# y_one_hot = torch.zeros_like(hypothesis)
# y_one_hot.scatter_(1, y.unsqueeze(1), 1)
#
# print(y.unsqueeze(1))
#
# print(y_one_hot)
#
# cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
# print(cost)
#
# # 1-2. 파이토치로 소프트맥스의 비용 함수 구현하기 (하이-레벨)
# # F.softmax() + torch.log() = F.log_softmax()
# # Low Level
# print(torch.log(F.softmax(z, dim=1)))
#
# # High Level
# print(F.log_softmax(z, dim=1))
#
# # F.log_softmax() + F.nll_loss() = F.cross_entropy()
# # Low Level
# # 첫번째 수식
# print((y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean())
#
# # 두번째 수식
# print((y_one_hot * -F.log_softmax(z, dim=1)).sum(dim=1).mean())
#
# # High level
# # 세번째 수식
# # nll = Negative Log Likelihood
# print(F.nll_loss(F.log_softmax(z, dim=1), y))
#
# # 네번째 수식
# print(F.cross_entropy(z, y))

# 2. 소프트맥스 회귀 구현하기

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train = [[1, 2, 1, 1],
           [2, 1, 3, 2],
           [3, 1, 3, 4],
           [4, 1, 5, 5],
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

print(x_train.shape)
print(y_train.shape)

y_one_hot = torch.zeros(8, 3)
y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
print(y_one_hot.shape)

# 소프트맥스 회귀 구현하기(로우-레벨)
# 모델 초기화
# W = torch.zeros((4, 3), requires_grad=True)
# b = torch.zeros((1, 3), requires_grad=True)
#
# # optimizer 설정
# optimizer = optim.SGD([W, b], lr=0.1)
#
# nb_epochs = 1000
# for epoch in range(nb_epochs + 1):
#
#     # 가설
#     hypothesis = F.softmax(x_train.matmul(W) + b, dim=1)
#
#     # 비용 함수
#     cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
#
#     # cost로 H(x) 개선
#     optimizer.zero_grad()
#     cost.backward()
#     optimizer.step()
#
#     # 100번마다 로그 출력
#     if epoch % 100 == 0:
#         print('Epoch {:4d}/{} Cost: {:.6f}'.format(
#             epoch, nb_epochs, cost.item()
#         ))

# 소프트맥스 회귀 구현하기(하이-레벨)
# W = torch.zeros((4, 3), requires_grad=True)
# b = torch.zeros((1, 3), requires_grad=True)
#
# # optimizer 설정
# optimizer = optim.SGD([W, b], lr=0.1)
#
# nb_epochs = 1000
# for epoch in range(nb_epochs + 1):
#
#     # cost 계산
#     z = x_train.matmul(W) + b
#     cost = F.cross_entropy(z, y_train)
#
#     # cost로 H(x) 개선
#     optimizer.zero_grad()
#     cost.backward()
#     optimizer.step()
#
#     # 100번마다 로그 출력
#     if epoch % 100 == 0:
#         print('Epoch {:4d}/{} Cost: {:.6f}'.format(
#             epoch, nb_epochs, cost.item()
#         ))

# 소프트맥스 회귀 nn.Module로 구현하기
# 모델을 선언 및 초기화. 4개의 특성을 가지고 3개의 클래스로 분류. input_dim=4, output_dim=3.
model = nn.Linear(4, 3)

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.cross_entropy(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))

# 소프트맥스 회귀 클래스로 구현하기
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3) # Output = 3

    def forward(self, x):
        return self.linear(x)

model = SoftmaxClassifierModel()

# optimizer 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.cross_entropy(prediction, y_train)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # 20번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))
