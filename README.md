# 2022_2_BA-Anomaly_detection_gaussian_and_Local_outlier_factor
Tutorial Homework 3(Business Analytics class in Industrial &amp; Management Engineering, Korea Univ.)


# Overview of Anomaly detection
Anomaly, Outlier, Novelty(이하 이상치)는 각각 다른 뉘앙스를 가지지만, 나머지 '일반적인' 데이터셋과는 다른 메커니즘에서 발생한 것으로 본다는 점에서 똑같은 의미를 가지고 있다. 비슷한 뉘앙스인 noise는 통제가 불가능하여 일정 부분 감수해야 하는 것과 대조적으로, 이상치는 메커니즘 자체가 다르다고 판단되기 때문에 판별하여 조치를 취할 수 있다. 여기서 말하는 조치란, 단순히 데이터 분석에서 이상치를 제거한다든가, 공정 관리에서 이상을 탐지하고 고치는 것만 포함되지 않는다. 경우에 따라서는 오히려 이상치에 해당하는 대상을 귀중히 대해야할 수도 있다.

![그림1](https://user-images.githubusercontent.com/106015570/202488361-b5294c38-9b1b-4b19-a655-d01fa616223c.png)

일반적으로 이상치라고 하면, 제조 공정에서의 불량품 등 부정적인 이미지가 강하다. 그러나 긍정적인 의미에서의 이상치 또한 언제든 존재한다. 예를 들어, 백화점에서 고가의 명품을 많이 구매하는 고객은 다른 고객들에 비해 이상할 정도로 사용량이 큰 소비 메커니즘을 가지는 이상치라고 볼 수 있다. 그러나 백화점 입장에서는 최고의 우수고객일 것이다. 이와 같은 경우에는 해당 고객, 즉 이상치의 패턴을 잘 인식하여, 비슷한 패턴을 가지는 고객을 끌어올 수 있도록 노력해야할 것이다. 즉, 이상치가 분석을 방해하는 요소로서 제거되는 것이 아닌, 특수 관리 대상으로서 오히려 면밀한 분석이 필요한 대상이 되는 것이다.

본 튜토리얼은 밀도 기반의 이상탐지를 다루어볼 것이다. 그 중에서도, 가우시안 분포를 가정한 이상탐지와, 분포에 대한 가정이 없는 이상탐지(LOF) 두 가지를 비교해 볼 것이다.

# Introduce Methods

분포 기반의 방법론이란, 데이터의 일정한 분포를 따라 밀집할 것이라는 가정 하에, 정상 분포를 벗어난다고 판단하는 인스턴스에 대하여 이상을 판단하는 것을 의미한다. 일반적으로 많이 사용되는 분포는 가우시안 분포이다. 가우시안 분포는 정규분포 등, 평균과 표준편차를 주요 파라미터로 하여 좌우가 대칭인 형태의 분포를 의미한다. 흔히들 "정규분포"라는 이름으로 알고있다.

![그림2](https://user-images.githubusercontent.com/106015570/202498865-2c74a630-e30b-4349-9151-69891b464b1b.png)


한편, 특정 분포를 가정하지 않고도, 각 인스턴스 간 거리를 이용하여 데이터 자체의 밀도와 그에 기반한 이상치를 탐지하는 방법이 있다. 그 중 본 튜토리얼에서 소개하고자 하는 내용은 바로 Local Outlier Factor(이하 LOF)이다. LOF의 가장 큰 특징은, 인스턴스별로 주변 이웃과의 거리가 절대적으로 같다고 하더라도, 주변의 밀도가 높을수록 이상치 스코어가 높아진다는 것이다. 즉, 원래 밀도가 낮은 곳에서 이웃과의 거리가 먼 경우는 이상으로 판단하지 않으나, 밀도가 높은 곳에서 이웃과의 거리가 먼 경우는 이상으로 판단한다는 것이다.

![그림3](https://user-images.githubusercontent.com/106015570/202498850-d08d379f-03d4-4db6-8249-bc16c3327f41.png)

# Tutorial of Anomaly Detection
## 코드 및 데이터 개요
본 tutorial에 사용된 패키지는 아래와 같이 import 되었다.

```

import numpy as np
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import Counter

from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

```

데이터는 sklearn의 make_blob 메소드를 이용하여 직접 제작되었다. 데이터의 hyperparameter는 전체 1000개 인스턴스 중, 5%(50개)를 outlier로 한다. 구체적으로는 아래의 표 및 소스 코드와 같이 데이터가 제작되었다.
 
|분포 형태|중점(x, y좌표)|표준편차|비고|
|------|---|---|---|
|단일 가우시안 분포|(5,10)| 1.2 |가우시안 1개|
|다중 가우시안 분포1|(5,10), (10,20)| 1.5, 1.5|표준편차가 같은 가우시안 2개|
|다중 가우시안 분포2|(5,10), (10,20)| 1.5, 2.8|표준편차가 각각 다른 가우시간 2개|

```

#### Setting toy dataset with some novelty
# setting dataset size and parameter

n_samples = 1000
outliers_fraction = 0.05
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers
blobs_params = dict(random_state=333, n_samples=n_inliers, n_features=2)
abnormal = 25.0 * (np.random.RandomState(333).rand(n_outliers, 2))

datasets = []

d1 = [x for x in make_blobs(centers=[[5, 10], [5, 10]], cluster_std=1.2, **blobs_params)[0]]
[d1.append(x) for x in abnormal]
d1=np.array(d1)

d2 = [x for x in make_blobs(centers=[[5, 10], [10, 20]], cluster_std=[1.5, 1.5], **blobs_params)[0]]
[d2.append(x) for x in abnormal]
d2=np.array(d2)

d3 = [x for x in make_blobs(centers=[[5, 10], [10, 20]], cluster_std=[1.5, 2.8], **blobs_params)[0]]
[d3.append(x) for x in abnormal]
d3=np.array(d3)

datasets.append(d1)
datasets.append(d2)
datasets.append(d3)

```

데이터셋 분포를 순서대로 시각화하면 아래의 그림들과 같다.

(단일 가우시안 분포 데이터셋)
![dataset_0](https://user-images.githubusercontent.com/106015570/202502161-9213761a-b394-473a-b11f-2b82df92fadb.png)

(다중 가우시안1 분포 데이터셋)
![dataset_1](https://user-images.githubusercontent.com/106015570/202502259-e2632137-b5ab-4d64-994f-f2900b3cfa36.png)

(다중 가우시안2 분포 데이터셋)
![dataset_2](https://user-images.githubusercontent.com/106015570/202502279-5dbb647b-992f-4efc-9955-1c6ecedece7f.png)

## 가우시안 분포 기반 방법론
1. 단일 가우시안 밀도 분석
단일 가우시안 밀도 분석은 하나의 가우시안 분포 및 파라미터에 의존하여 이상치를 판단한다. 소스코드 및 결과는 아래와 같다.

```
#### 1) Parametric Method : Using Gaussian distribution

# Estimate Gaussian parameters 

def gaussian_estimator(dataset):
    
    avg1 = np.mean(dataset.T[0])
    avg2 = np.mean(dataset.T[1])
    avg = np.array([avg1, avg2])
    stdev = np.std(dataset, axis=0)
    print(avg, stdev)
    
    return avg, stdev
    

# Estimate Gaussian distribution

def multi_gaussian_estimator(dataset, avg, stdev):
    
    k = len(avg)
    stdev=np.diag(stdev)
    dataset = dataset - avg.T
    p = 1/((2*np.pi)**(k/2)*(np.linalg.det(stdev)**0.5))* np.exp(-0.5* np.sum(dataset @ np.linalg.pinv(stdev) * dataset ,axis=1))
    
    return p

# plotting gaussian density(single distribution)

avg_list = []
std_list = []
dist_list = []

for idx, dataset in enumerate(datasets):
    avg, stdev = gaussian_estimator(dataset)
    avg_list.append(avg)
    std_list.append(stdev)
    p = multi_gaussian_estimator(dataset, avg, stdev)
    dist_list.append(p)
    
    plt.figure(figsize=(15,8))
    plt.scatter(dataset[:,0], dataset[:,1], marker = "o", c=p, cmap='jet');
    plt.colorbar();
    
    plt.show()

```

(단일 가우시안 분포 데이터셋)
![single_gaussian_0](https://user-images.githubusercontent.com/106015570/202503044-751af33d-b0e8-4a58-a41a-602f170fb0e5.png)

(다중 가우시안1 분포 데이터셋)
![single_gaussian_1](https://user-images.githubusercontent.com/106015570/202503064-91acd42e-aea2-419e-9f10-3f117440e370.png)

(다중 가우시안2 분포 데이터셋)
![single_gaussian_2](https://user-images.githubusercontent.com/106015570/202503094-9800c19d-1832-445e-a6ed-426a97666f13.png)

threshold 0.001에서 이상치를 판별한 결과, 단일 가우시안의 경우 50개 이상 인스턴스 중 90%에 해당하는 45개를 탐지해냈으나, 나머지 두 분포에서는 성능이 확연히 좋지 않음을 확인할 수 있었다.


|분포 형태|단일 가우시안 밀도 분석 결과|
|------|---|
|단일 가우시안 분포|이상 45개(50개 중 90%)|
|다중 가우시안 분포1|이상 501개(정상에 대해서도 이상 탐지)|
|다중 가우시안 분포2|이상 524개(정상에 대해서도 이상 탐지)|

(단일 가우시안 분포 데이터셋 결과)
![single_gaussian_0_outlier_plot](https://user-images.githubusercontent.com/106015570/202504196-b0f14c10-3db6-40b6-9550-29df9b9b400f.png)

(다중 가우시안1 분포 데이터셋 결과)
![single_gaussian_1_outlier_plot](https://user-images.githubusercontent.com/106015570/202504230-9601b927-3a33-4509-812e-4aa849062609.png)

(다중 가우시안2 분포 데이터셋 결과)
![single_gaussian_2_outlier_plot](https://user-images.githubusercontent.com/106015570/202504238-cbba0e1d-41d9-49b3-9d57-b05a0e1698dd.png)

2. 가우시안 혼합(Mixture of Gaussian)
가우시안 혼합은 여러 개의 가우시안 분포 및 파라미터에 의존하여 이상치를 판단한다. 단일 가우시안 밀도 분석의 지나치게 강한 가정(하나의 가우시안 분포만을 따름)을 완화하기 위한 방법으로 작용한다. 본 튜토리얼에서 가우시안 혼합은 결과를 상세히 분석하기 보다는 다중 가우시안 데이터셋 내 분포를 잘 잡아내는지를 확인하는 수준에 그친다. 소스코드 및 결과는 아래와 같다.

```

for p, dataset in zip(dist_list, datasets):
    mog = GaussianMixture(n_components = 2, covariance_type = 'full', random_state=333)
    mog.fit(dataset)
    
    plt.figure(figsize=(15,16))
    for i in range(2):
        plt.subplot(2, 1, i+1)
        plt.scatter(dataset[:,0],dataset[:,1],c=mog.predict_proba(dataset)[:,i],cmap='rainbow',marker='o')
        plt.colorbar();
    
    plt.show()

```

(단일 가우시안 분포 데이터셋 결과)
![MoG_0](https://user-images.githubusercontent.com/106015570/202506743-bcf0cc86-63a6-42c8-ac38-7163dc73adca.png)

(다중 가우시안1 분포 데이터셋 결과)
![MoG_1](https://user-images.githubusercontent.com/106015570/202506756-01b251a6-c253-4ffb-a35a-3b575dd2c442.png)

(다중 가우시안2 분포 데이터셋 결과)
![MoG_2](https://user-images.githubusercontent.com/106015570/202506762-81e81464-b0a0-4414-8496-d37a7e992ba6.png)

## Local Outlier Factor
LOF는 이웃과의 거리 및 밀도에 따라 이상치를 판단한다. LOF의 소스코드 및 이상치를 판단한 결과는 아래와 같다.

```
#### 2) Non-parametric method : Local Outlier Factor

lof = LocalOutlierFactor(n_neighbors=5)
result_lof = lof.fit_predict(dataset)

i = 0
for dataset in datasets:
    print(Counter(result_lof))
    df_lof = {'X1':dataset.T[0], 'X2':dataset.T[1], 'result_lof':result_lof, 'outlier_factor':lof.negative_outlier_factor_}
    df_lof = pd.DataFrame(df_lof)
    
    df_normal = df_lof[df_lof['result_lof'] == 1]
    df_abnormal = df_lof[df_lof['result_lof'] == -1]
    
    print("number of outlier : " + str (len(df_abnormal)))
    
    plt.figure(figsize=(15,8))
    plt.scatter(df_normal['X1'], df_normal['X2'], marker = "o", c='g')
    plt.scatter(df_abnormal['X1'],df_abnormal['X2'],marker="x", c='r') # plot novelties

    plt.savefig(f'C:/Users/kw764/Desktop/kw/00_lecture/2022-2/BA/튜토리얼이미지/LOF_{i}.png', edgecolor='black', format='png', dpi=200)
    i += 1

```

|분포 형태|단일 가우시안 밀도 분석 결과|
|------|---|
|단일 가우시안 분포|이상 45개(50개 중 90%)|
|다중 가우시안 분포1|이상 45개(50개 중 90%)|
|다중 가우시안 분포2|이상 45개(50개 중 90%)|

(단일 가우시안 분포 데이터셋 결과)
![LOF_0](https://user-images.githubusercontent.com/106015570/202507329-bc9cb504-0731-4aa6-989e-63e64a8a9ead.png)

(다중 가우시안1 분포 데이터셋 결과)
![LOF_1](https://user-images.githubusercontent.com/106015570/202507350-689f636d-8cb4-46a8-8be4-bd97054b9650.png)

(다중 가우시안2 분포 데이터셋 결과)
![LOF_2](https://user-images.githubusercontent.com/106015570/202507361-2fc3f4a2-d742-4ec7-b95f-e0f1c7bef2c0.png)

결과에서도 알 수 있듯이, 인근의 밀도가 높은 인스턴스의 경우 다른 인스턴스들보다 이웃 간 거리가 가까움에도 이상으로 판별되는 경우를 확인할 수 있다. 


# Conclusion

본 튜토리얼에서는 가상의 데이터셋을 만든 후 밀도 기반의 이상탐지 알고리즘을 시현해보았다. 본 튜토리얼은 실제 데이터셋을 이용하여 결과를 내보기 보다는, 각 알고리즘의 특성을 파악하고, 시각적으로 구현하는 것에 중점을 두었다. 이 때문에 y값, 즉 label을 설정하여 엄밀하게 이상을 탐지하는 것 보다는, label이 없는 상태에서 비지도 학습으로 지정된 이상 비율을 얼마나 만족시키는지를 확인하는 것에 그쳤다. 즉,  데이터를 좀 더 정밀하게 만들지 못하여, 결과의 신빙성이 낮아졌다는 것이다. 향후 가상의 데이터셋을 만들어 시현할 경우, y값에 대한 고려가 충분히 선행되어야 할 것이다. 
