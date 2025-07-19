import numpy as np

# 1. 넘파이로 텐서 만들기(벡터와 행렬 만들기)

# 1-1. 1D with Numpy
t = np.array([0., 1., 2., 3., 4., 5., 6.])
# 파이썬으로 설명하면 List를 생성해서 np.array로 1차원 array로 변환함.
print(t)

# ndim: 차원
# shape: 크기 (ex: (7,) => (1,7) => 1x7 크기의 벡터)
print('Rank of t: ', t.ndim)
print('Shape of t: ', t.shape)

# 1-2. Numpy 기초 이해하기
print('t[0] t[1] t[-1] = ', t[0], t[1], t[-1])

print('t[2:5] t[4:-1] = ', t[2:5], t[4:-1])

print('t[:2] t[3:] = ', t[:2], t[3:])

# 1-3. 2D with Numpy
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
print(t)

print('Rank  of t: ', t.ndim)
print('Shape of t: ', t.shape)

# 2. 파이토치 텐서 선언하기(PyTorch Tensor Allocation)
import torch

# 2-1. 1D PyTorch
t = torch.FloatTensor([0., 1., 2., 3., 4., 5., 6.])
print(t)

print(t.dim())  # rank. 즉, 차원
print(t.shape)  # shape
print(t.size()) # shape

print(t[0], t[1], t[-1])  # 인덱스로 접근
print(t[2:5], t[4:-1])    # 슬라이싱
print(t[:2], t[3:])       # 슬라이싱

# 2-2. 2D with PyTorch
t = torch.FloatTensor([[1., 2., 3.],
                       [4., 5., 6.],
                       [7., 8., 9.],
                       [10., 11., 12.]
                      ])
print(t)

print(t.dim())  # rank. 즉, 차원
print(t.size()) # shape

print(t[:, 1]) # 첫번째 차원을 전체 선택한 상황에서 두번째 차원의 첫번째 것만 가져온다.
print(t[:, 1].size()) # ↑ 위의 경우의 크기

print(t[:, :-1]) # 첫번째 차원을 전체 선택한 상황에서 두번째 차원에서는 맨 마지막에서 첫번째를 제외하고 다 가져온다.

# 2-3. 브로드캐스팅(Broadcasting): 크기가 다른 행렬 또는 텐서 간에 사칙 연산을 수행할 수 있도록 하는 기능
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2)

# Vector + scalar
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3]) # [3] -> [3, 3]
print(m1 + m2)

# 2 x 1 Vector + 1 x 2 Vector
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2)

# 브로드캐스팅 과정에서 실제로 두 텐서가 어떻게 변경되는지 보겠습니다.
# [1, 2]
# ==> [[1, 2],
#      [1, 2]]
# [3]
# [4]
# ==> [[3, 3],
#      [4, 4]]

# 3. 자주 사용되는 기능들

# 3-1. 행렬 곱셈과 곱셈의 차이(Matrix Multiplication Vs. Multiplication)
m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
print(m1.matmul(m2)) # 2 x 1

m1 = torch.FloatTensor([[1, 2], [3, 4]])
m2 = torch.FloatTensor([[1], [2]])
print('Shape of Matrix 1: ', m1.shape) # 2 x 2
print('Shape of Matrix 2: ', m2.shape) # 2 x 1
print(m1 * m2) # 2 x 2
print(m1.mul(m2))

# 3-2. 평균(Mean)
t = torch.FloatTensor([1, 2])
print(t.mean())

t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)

print(t.mean())

print(t.mean(dim=0))

print(t.mean(dim=1))

print(t.mean(dim=-1))

# 3-3. 덧셈(Sum)
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)

print(t.sum()) # 단순히 원소 전체의 덧셈을 수행
print(t.sum(dim=0)) # 행을 제거
print(t.sum(dim=1)) # 열을 제거
print(t.sum(dim=-1)) # 열을 제거

# 3-4. 최대(Max)와 아그맥스(ArgMax)
t = torch.FloatTensor([[1, 2], [3, 4]])
print(t)

print(t.max())

print(t.max(dim=0))

print('Max: ', t.max(dim=0)[0])
print('Argmax: ', t.max(dim=0)[1])

print(t.max(dim=1))
print(t.max(dim=-1))

# 3-5. 뷰(View) - 원소의 수를 유지하면서 텐서의 크기 변경. 매우 중요!
t = np.array([[[0, 1, 2],
               [3, 4, 5]],
              [[6, 7, 8],
               [9, 10, 11]]])
ft = torch.FloatTensor(t)

print(ft.shape) # 2 * 2 * 3

print(ft.view([-1, 3])) # ft라는 텐서를 (?, 3)의 크기로 변경
print(ft.view([-1, 3]).shape)

# (2, 2, 3) -> (2 * 2, 3) -> (4, 3)
# view는 기본적으로 변경 전과 변경 후의 텐서 안의 원소의 개수가 유지되어야 합니다.
# 파이토치의 view는 사이즈가 -1로 설정되면 다른 차원으로부터 해당 값을 유추합니다.

print(ft.view([-1, 1, 3]))
print(ft.view([-1, 1, 3]).shape)

# 3-6. 스퀴즈(Squeeze) - 1인 차원을 제거한다.
ft = torch.FloatTensor([[0], [1], [2]])
print(ft)
print(ft.shape)

print(ft.squeeze())
print(ft.squeeze().shape)

# 3-7. 언스퀴즈(Unsqueeze) - 특정 위치에 1인 차원을 추가한다.
ft = torch.Tensor([0, 1, 2])
print(ft.shape)

print(ft.unsqueeze(0)) # 인덱스가 0부터 시작하므로 0은 첫번째 차원을 의미한다.
print(ft.unsqueeze(0).shape)

print(ft.view(1, -1))
print(ft.view(1, -1).shape)

print(ft.unsqueeze(1))
print(ft.unsqueeze(1).shape)

print(ft.unsqueeze(-1))
print(ft.unsqueeze(-1).shape)

# 3-8. 타입 캐스팅(Type Casting)
lt = torch.LongTensor([1, 2, 3, 4])
print(lt)

print(lt.float())

bt = torch.ByteTensor([True, False, False, True])
print(bt)

print(bt.long())
print(bt.float())

# 3-9. 연결하기(concatenate)
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])

print(torch.cat([x, y], dim=0))

print(torch.cat([x, y], dim=1))

# 3-10. 스택킹(Stacking)
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])

print(torch.stack([x, y, z]))

print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))

print(torch.stack([x, y, z], dim=1))

# 3-11. ones_like와 zeros_like - 0으로 채워진 텐서와 1로 채워진 텐서
x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])
print(x)

print(torch.ones_like(x))

print(torch.zeros_like(x))

# 3-12. In-place Operation (덮어쓰기 연산)
x = torch.FloatTensor([[1, 2], [3, 4]])

print(x.mul(2.)) # 곱하기 2를 수행한 결과를 출력
print(x) # 기존의 값 출력

print(x.mul_(2.))  # 곱하기 2를 수행한 결과를 변수 x에 값을 저장하면서 결과를 출력
print(x) # 기존의 값 출력
