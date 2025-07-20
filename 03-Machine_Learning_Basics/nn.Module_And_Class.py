# 1. 단순 선형 회귀 구현하기
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# torch.manual_seed(1)
#
# # 데이터
# x_train = torch.FloatTensor([[1], [2], [3]])
# y_train = torch.FloatTensor([[2], [4], [6]])
#
# # 모델을 선언 및 초기화. 단순 선형 회귀이므로 input_dim=1, output_dim=1.
# # 하나의 입력 x에 대해서 하나의 출력 y를 가짐
# model = nn.Linear(1, 1)
#
# print(list(model.parameters()))
#
# # optimizer 설정. 경사 하강법 SGD를 사용하고 learning rate를 의미하는 lr은 0.01
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#
# # 전체 훈련 데이터에 대해 경사 하강법을 2,000회 반복
# nb_epochs = 2000
# for epoch in range(nb_epochs + 1):
#
#     # H(x) 계산
#     prediction = model(x_train)
#
#     # cost 계산
#     cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수
#
#     # cost로 H(x) 개선하는 부분
#     # gradient를 0으로 초기화
#     optimizer.zero_grad()
#
#     # 비용 함수를 미분하여 gradient 계산
#     cost.backward()
#
#     # W와 b를 업데이트
#     optimizer.step()
#
#     if epoch % 100 == 0:
#         # 100번마다 로그 출력
#         print('Epoch {:4d}/{} Cost: {:.6f}'.format(
#             epoch, nb_epochs, cost.item()
#         ))
#
# # 임의의 입력 4를 선언
# new_var = torch.FloatTensor([[4.0]])
#
# # 입력한 값 4에 대해서 예측값 y를 리턴받아서 pred_y에 저장
# pred_y = model(new_var)
#
# # y = 2x 이므로 입력이 4라면 y가 8에 가까운 값이 나와야 제대로 학습이 된 것
# print('훈련 후 입력이 4일 때의 예측값 :', pred_y)
#
# print(list(model.parameters()))

# 2. 다중 선형 회귀 구현하기
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# torch.manual_seed(1)
#
# # 데이터
# x_train = torch.FloatTensor([[73, 80, 75],
#                              [93, 88, 93],
#                              [89, 91, 90],
#                              [96, 98, 100],
#                              [73, 66, 70]])
# y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
#
# model = nn.Linear(3, 1)
#
# print(list(model.parameters()))
#
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
#
# nb_epochs = 2000
# for epoch in range(nb_epochs + 1):
#
#     # H(x) 계산
#     prediction = model(x_train)
#
#     # cost 계산
#     cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수
#
#     # cost로 H(x) 개선하는 부분
#     # gradient를 0으로 초기화
#     optimizer.zero_grad()
#
#     # 비용 함수를 미분하여 gradient 계산
#     cost.backward()
#
#     # W와 b를 업데이트
#     optimizer.step()
#
#     if epoch % 100 == 0:
#         # 100번마다 로그 출력
#         print('Epoch {:4d}/{} Cost: {:.6f}'.format(
#             epoch, nb_epochs, cost.item()
#         ))
#
# # 임의의 입력 [73, 80, 75]를 선언
# new_var = torch.FloatTensor([[73, 80, 75]])
#
# # 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
# pred_y = model(new_var)
# print('훈련 후 입력이 73, 80, 75일 때의 예측값 :', pred_y)
#
# print(list(model.parameters()))

# 3. 모델을 클래스로 구현하기
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# model = nn.Linear(1, 1)
#
# class LinearRegressionModel(nn.Module): # torch.nn.Module을 상속받는 파이썬 클래스
#     def __init__(self):
#         super().__init__()
#         self.linear = nn.Linear(1, 1) # 단순 선형 회귀이므로 input_dim=1, output_dim = 1
#
#         def forward(self, x):
#             return self.linear(x)
#
# model = LinearRegressionModel()

# 4. 단순 선형 회귀 클래스로 구현하기
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# torch.manual_seed(1)
#
# # 데이터
# x_train = torch.FloatTensor([[1], [2], [3]])
# y_train = torch.FloatTensor(([2], [4], [6]))
#
# class LinearRegressionModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = nn.Linear(1, 1)
#
#     def forward(self, x):
#         return self.linear(x)
#
# model = LinearRegressionModel()
#
# # optimizer 설정. 경사 하강법 SGD를 사용하고 learning rate를 의미하는 1r은 0.01
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
#
# # 전체 훈련 데이터에 대해 경사 하강법을 2,000회 반복
# nb_epochs = 2000
# for epoch in range(nb_epochs + 1):
#
#     # H(x) 계산
#     prediction = model(x_train)
#
#     # cost 계산
#     cost = F.mse_loss(prediction, y_train)
#
#     # cost로 H(x) 개선
#     # gradient를 0으로 초기화
#     optimizer.zero_grad()
#
#     # 비용 함수를 미분하여 gradient 계산
#     cost.backward()
#
#     # W와 b를 업데이트
#     optimizer.step()
#
#     if epoch % 100 == 0:
#         print('Epoch {:4d}/{} Cost: {:.6f}'.format(
#             epoch, nb_epochs, cost.item()
#         ))

# 5. 다중 선형 회귀 클래스로 구현하기
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# 데이터
# 데이터
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)

model = MultivariateLinearRegressionModel()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 2000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train)

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()

    # 비용 함수를 미분하여 gradient 계산
    cost.backward()

    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
        # 100번마다 로그 출력
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, cost.item()
        ))