# KNN_Mediapipe_Hangul-Sign-Language
## Python Mediapipe를 사용하여 손을 이용한 한글 지문자 표현입니다.
<img src="https://user-images.githubusercontent.com/78673090/135803938-022878e3-86bc-4134-9417-f1cc84094c67.gif" width="800" height="500"/>

## 요약
<img src="https://user-images.githubusercontent.com/78673090/135808283-6140c9ee-b35f-4191-9e2f-be90da308ba2.png" width="600" height="300"/>

![image](https://user-images.githubusercontent.com/78673090/135805341-5ae6503f-3e4c-45e0-b413-530d8bd35ca5.png)

- Mediapipe로 인식한 손의 각 부분 벡터의 사이 각도를 구함 (각 제스처의 각도를 csv파일로 저장)
- 각 제스처의 각도를 저장한 데이터셋을 KNN 최근접 알고리즘을 사용하여 알아냄

#### 1. [Make_Dataset](https://github.com/yoonth95/KNN_Mediapipe_Hangul-Sign-Language/blob/master/Make_Dataset.py)
- Mediapipe로 인식한 손에서 34가지의 제스처를 취했을 때 계산되는 각도를 csv파일로 저장
- 복합 자음자: ㄲ, ㄸ, ㅃ, ㅆ, ㅉ 등의 경우 한 손으로 하기엔 정확도가 너무 낮아져 일단은 빼고 진행 (추후에 두 손으로 할 생각)
- 자음, 모음 이외의 제스처 (출력, 지우기, 띄어쓰기 등)는 다른 제스처와 겹치지 않게 새로 만듬
- 0.000000 ~ 33.000000 라벨링 값을 넣어주며 저장 
- 손 모양을 정확하게 하고 다른 제스처와의 차이점이 명확하면 정확도가 훨씬 높아짐

#### 2. [KNN_Mediapipe_Hangul-Sign-Languge](https://github.com/yoonth95/KNN_Mediapipe_Hangul-Sign-Language/blob/master/KNN_Mediapipe_Hand-Gesture.ipynb)
- 저장된 Dataset를 사용하여 정확도 측정
- KNN을 사용했을 시 정확도는 k=1 ~ 10까지 알아봤을 때 k=1 일 경우 가장 높은 정확도를 보임 (최소 95이상)
- 화면상에서 글씨 출력 시 한글을 출력하려면 PIL을 사용
- 손 동작 했을 시 나오는 출력 단어는 PIL이 Mediapipe와 충돌이 생겨 오류가 발생 (영어로 표시)
- 예측된 글자는 PIL을 이용하여 한글로 출력
- 자음, 모음으로 된 글자를 문장으로 변환하여 최종 출력
