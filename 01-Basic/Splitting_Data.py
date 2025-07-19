import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 1. X와 y 분리하기

# 1-1. zip 함수를 이용하여 분리하기
X, y = zip(['a', 1], ['b', 2], ['c', 3])
print('x 데이터 :', X)
print('y 데이터: ', y)

# 리스트의 리스트 또는 행렬 또는 뒤에서 배울 개념인 2D 텐서.
sequence = [['a', 1], ['b', 2], ['c', 3]]
X, y = zip(*sequence)
print('x 데이터 :', X)
print('y 데이터: ', y)

# 1-2. 데이터프레임을 이용하여 분리하기
values = [['당신에게 드리는 마지막 혜택!', 1],
['내일 뵐 수 있을지 확인 부탁드...', 0],
['도연씨. 잘 지내시죠? 오랜만입...', 0],
['(광고) AI로 주가를 예측할 수 있다!', 1]]
columns = ['메일 본문', '스팸 메일 유무']

df = pd.DataFrame(values, columns=columns)
print(df)

X = df['메일 본문']
y = df['스팸 메일 유무']

print('x 데이터 :', X.to_list())
print('y 데이터 :', y.to_list())

# 1-3. Numpy를 이용하여 분리하기
np_array = np.arange(0, 16).reshape((4, 4))
print('전체 데이터 :')
print(np_array)

X = np_array[:, :3]
y = np_array[:,3]

print('x 데이터 :')
print(X)
print('y 데이터 :', y)

# 2. 테스트 데이터 분리하기

# 2-1. 사이킷 런을 이용하여 분리하기
# 임의로 X와 y 데이터를 생성
X, y = np.arange(10).reshape((5, 2)), range(5)

print('X 전체 데이터 :')
print(X)
print('y 전체 데이터 :')
print(list(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

print('x 훈련 데이터 :')
print(X_train)
print('X 테스트 데이터 :')
print(X_test)

print('y 훈련 데이터 :')
print(y_train)
print('y 테스트 데이터 :')
print(y_test)

# random_state의 값을 변경
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

print('y 훈련 데이터 :')
print(y_train)
print('y 테스트 데이터 :')
print(y_test)

# 2-2. 수동으로 분리하기
# 실습을 위해 임의로 X와 y가 이미 분리 된 데이터를 생성
X, y = np.arange(0, 24).reshape((12, 2)), range(12)

print('X 전체 데이터 :')
print(X)
print('y 전체 데이터 :')
print(list(y))

num_of_train = int(len(X) * 0.8)

# 테스트 데이터 = 전체 데이터 - 훈련 데이터
# 이유: 테스트 데이터를 int(전체 데이터 * 0.2)로 계산한다면, 전체가 4518이라고 가정했을 때 => 훈련: 3614, 테스트: 903
# 3614 + 903 = 4517 => 전체 4518에 비해 데이터 1개 누락
num_of_test = int(len(X) - num_of_train)
print('훈련 데이터의 크기 :', num_of_train)
print('테스트 데이터의 크기 :', num_of_test)

X_test = X[num_of_train:] # 전체 데이터 중에서 20%만큼 뒤의 데이터 저장
y_test = y[num_of_train:] # 전체 데이터 중에서 20%만큼 뒤의 데이터 저장
X_train = X[:num_of_train] # 전체 데이터 중에서 80%만큼 앞의 데이터 저장
y_train = y[:num_of_train] # 전체 데이터 중에서 80%만큼 앞의 데이터 저장

print('X 테스트 데이터 :')
print(X_test)
print('y 테스트 데이터 :')
print(list(y_test))