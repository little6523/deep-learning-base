# 1. 데이터에 대한 이해
# 3개의 퀴즈 점수로부터 최종 점수 예측하는 모델 만들기
# H(x) = w1x1 + w2x2 + w3x3 + b

# 2. 파이토치 구현
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# 훈련 데이터
# x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
# x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
# x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])
# y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
#
# # 가중치 w와 편향 b 초기화
# w1 = torch.zeros(1, requires_grad=True)
# w2 = torch.zeros(1, requires_grad=True)
# w3 = torch.zeros(1, requires_grad=True)
# b = torch.zeros(1, requires_grad=True)
#
# optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)
#
# nb_epochs = 1000
# for epoch in range(nb_epochs + 1):
#
#     # H(x) 계산
#     hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b
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
#         print('Epoch {:4d}/{} w1: {:.3f} w2: {:.3f} w3: {:.3f} b: {:.3f} Cost: {:.6f}'.format(
#             epoch, nb_epochs, w1.item(), w2.item(), w3.item(), b.item(), cost.item()
#         ))

# 3. 벡터와 행렬 연산으로 바꾸기
# x의 개수가 1000개이면, 위의 방식대로 한다면 1000개의 변수 선언 필요 => 매우 비효율적
# => 행렬 곱셈 연산(또는 벡터의 내적) 사용

x_train  =  torch.FloatTensor([[73,  80,  75],
                               [93,  88,  93],
                               [89,  91,  80],
                               [96,  98,  100],
                               [73,  66,  70]])
y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])

# 모델 초기화
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([W, b], lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    # 편향 b는 브로드캐스팅되어 각 샘플에 더해짐
    hypothesis = x_train.matmul(W) + b

    # cost 계산
    cost = torch.mean((hypothesis - y_train) ** 2)

    # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
        epoch, nb_epochs, hypothesis.squeeze().detach(), cost.item()
    ))

with torch.no_grad():
    new_input = torch.FloatTensor([[75, 85, 72]])
    prediction = new_input.matmul(W) + b
    print('Predicted value for input {}: {}'.format(new_input.squeeze().tolist(), prediction.item()))