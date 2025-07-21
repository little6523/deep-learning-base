# 1. 데이터 로드하기(Data Load)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from torch.utils.data import TensorDataset # 텐서데이터셋
# from torch.utils.data import DataLoader # 데이터로더
#
# x_train  =  torch.FloatTensor([[73,  80,  75],
#                                [93,  88,  93],
#                                [89,  91,  90],
#                                [96,  98,  100],
#                                [73,  66,  70]])
# y_train  =  torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])
#
# dataset = TensorDataset(x_train, y_train)
#
# # shuffle=True를 선택하면 Epoch마다 데이터셋을 섞어 데이터가 학습되는 순서를 변경
# # 모델이 데이터의 순서에 익숙해지는 것을 방지하여 학습
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
#
# model = nn.Linear(3, 1)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
#
# nb_epochs = 20
# for epoch in range(nb_epochs + 1):
#     for batch_idx, samples in enumerate(dataloader):
#         # print(batch_idx)  # 배치 번호
#         # print(samples)    # 배치로 선택된 데이터. batch_size = 2 이므로, x_train과 그에 매핑되는 y_train의 값 2개를 각각 뽑음
#         x_train, y_train = samples
#
#         # H(x) 계산
#         prediction = model(x_train)
#
#         # cost 계산
#         cost = F.mse_loss(prediction, y_train)
#
#         # cost로 H(x) 계산
#         optimizer.zero_grad()
#         cost.backward()
#         optimizer.step()
#
#         print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
#             epoch, nb_epochs, batch_idx + 1, len(dataloader),
#             cost.item()
#         ))
#
# # 임의의 입력 [73, 80, 75]
# new_var = torch.FloatTensor([[73, 80, 75]])
#
# # 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
# pred_y = model(new_var)
# print('훈련 후 입력이 73, 80, 75일 때의 예측값 :', pred_y)

# 2. 커스텀 테이터셋(Custom Dataset)
# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self):
#     # 데이터셋의 전처리를 해주는 부분
#
#     def __len__(self):
#     # 데이터셋의 길이. 즉, 총 샘플의 수를 적어주는 부분
#
#     def __getitem__(self, item):
#     # 데이터셋에서 특정 1개의 샘플을 가져오는 함수

# 3. 커스텀 데이터셋(Custom Dataset)으로 선형 회귀 구현하기
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[73, 80, 75],
                       [93, 88, 93],
                       [89, 91, 90],
                       [96, 98, 100],
                       [73, 66, 70]]
        self.y_data = [[152], [185], [180], [196], [142]]

    # 총 데이터의 개수를 리턴
    def __len__(self):
        return len(self.x_data)

    # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y

dataset = CustomDataset()
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = torch.nn.Linear(3,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        # print(batch_idx)
        # print(samples)
        x_train, y_train = samples

        # H(x) 계산
        prediction = model(x_train)

        # cost 계산
        cost = F.mse_loss(prediction, y_train)

        # cost로 H(x) 계산
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, batch_idx + 1, len(dataloader),
            cost.item()
        ))

# 임의의 입력 [73, 80, 75]를 선언
new_var = torch.FloatTensor([[73, 80, 75]])

# 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var)
print('훈련 후 입력이 73, 80, 75일 때의 예측값 :', pred_y)