# 암호화폐 시계열 딥 클러스터링

# 1. 시작하기

## 1.1 실험에 사용된 Python 라이브러리 버전

'''
Python : 3.9.7
numpy : 1.20.3
pandas: 1.4.2
n2d: 0.3.2
tensorflow: 2.9.1
fastdtw: 0.3.4
tslearn: 0.5.2
pyclustering: 0.10.1.2
cvxpy: 1.2.0
'''

# 2. 코드 설명

## 2.1. `Get_data.py`

데이터 수집을 실행하는 파일이다. 상위 200 여 개의 암호화폐들의 가격, 시가총액, 거래량을 CoinGecko API를 통해서 가져온다.

## 2.2. `extract_price_market_volume.ipynb`

위에서 수집된 데이터 중 데이터 포인트가 365개 미만인 암호화폐들은 제외하고 나머지 데이터들로 데이터 프레임을 구성한다.

## 2.3. `ols.ipynb`

이 코드는 시장 영향을 제거하기 위해 각 암호화폐의 로그 수익률과 시장 전체의 로그 수익률에 대한 선형회귀를 실행한다. 각 암호화폐 시계열 데이터의 시장 영향을 제거하여 노이즈 감소와 다양화 향상, 자산간 상관 관계 파악을 개선하여 군집 구조의 안정성을 향상시키고자 하는 코드이다.

## 2.4. `Clustering_n2d.ipynb`

이 파일은 n2d 알고리즘을 사용하여 암호화폐 데이터를 클러스터링한다. N2d 알고리즘은 1) 오토인코더를 사용하여 고차원 데이터를 압축하고, 2) UMAP을 사용하여 압축된 데이터를 저차원 매니폴드에 매핑한 후, 3) 계층적 클러스터링을 사용하여 클러스터를 생성하는 방법이다.


## 2.5. `best_deep_clustering_models.ipynb`

이전 파일에서 실행한 클러스터링 결과들 중 생성된 임베딩에 대해 클러스터링을 수행하고 클러스터의 성능을 평가한다. 평가에는 euclidean 거리를 사용한 실루엣 스코어, Dynamic Time Warping(DTW) 거리를 사용한 실루엣 스코어, 그리고 DB index 스코어가 사용된다. 


## 2.6. `deep_clustering_model_analysis.ipynb`

이 코드에서는 상위 5개의 딥 클러스터링 모델에 대해 통계 및 질적 분석이 수행된다. 분석 결과로는 클러스터 내 개별 암호화폐의 연환산 로그 수익률 분포, 평균, 표준편차, 최대 하락폭의 박스플롯, 그리고 클러스터의 대표 시계열이 포함된다. 이러한 분석은 딥 클러스터링 모델의 성능과 생성된 암호화폐 클러스터의 특성에 대한 통찰력을 제공한다.

## 2.7. `clustering_factor.ipynb`

이 코드에서는 작은 시가 총액 포트폴리오와 큰 시가 총액 포트폴리오 간의 수익 차이를 의미하는 CSMB과 Price-Volume (Finance에서 달러 가치의 거래량) 요인에 기반하여 암호화폐를 네 개의 클러스터로 분할하는 기준 모델인 팩터 클러스터링을 수행한다. Price-Volume 요인은 평균 일일 거래량 * 가격이고 섹션 구분은 2021년 12월 31일 (마지막 날)의 시가 총액을 기준으로 수행된다. 이를 통해 각 클러스터의 통계적 특성이 표시되고 각 클러스터의 대표 시계열을 얻어 효율적 투자선을 비교할 수 있다.


## 2.8. `Deep_clustering_model_analysis.ipynb`

이 코드에서는 상위 5개 딥 클러스터링 모델에 대해 통계 및 질적 분석이 수행된다. 이 분석의 결과로 개별 암호화폐의 연 평균 로그 수익률 분포, 평균, 표준 편차, 최대 하락폭에 대한 상자 그림 및 클러스터의 대표 시계열을 얻을 수 있다. 이 분석은 딥 클러스터링 모델의 성능과 생성된 암호화폐 클러스터의 특성에 대한 통찰력을 제공한다.


## 2.9. `clustering_k-means (eu), agglo (eu), agglo (dtw).ipynb`, `clustering_k-means(Pearson correlation).ipynb`

성능을 비교하기 위한 비교 모델들로 k means clustering에 유클리디안 거리 또는 피어슨 상관계수를 사용해서 클러스터링 한 모델과 agglomerative clustering에 유클리디안 거리 또는 dynamic time warping 알고리즘을 활용해서 클러스터링 하는 모델을 포함하고 있다. 마찬가지로 각 클러스터의 통계적 특성과 대표 시계열을 얻어 효율적 투자선을 비교할 수 있다. 


## 2.10. `efficient_frontier.ipynb`

해당 코드는 최적화된 자산 배분을 찾기 위해 Sharpe ratio와 같은 포트폴리오 최적화 지표를 사용한다. 이를 위해, 다양한 클러스터링 모델(딥 클러스터링, 팩터, k-means, DTW)을 통해 생성된 포트폴리오와 3개월 만기 미국 국채 금리(tb)를 대상으로 효율적 투자선을 시각화하여 제공한다. 또한, 각 클러스터링 모델의 연간 최대 샤프 지수, 포트폴리오 가중치 및 탄젠트 포트폴리오 마커와 함께 각 클러스터링 모델의 성능을 분석한다.

