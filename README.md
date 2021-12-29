# jigsaw-toxic-competition
Our Solution for 4th jigsaw toxic competition

## Task

* 맥락 정보 없는 텍스트 입력에 대해 얼마나 toxic한지 판단
    * toxicity의 정도가 있음
* 14,000 커멘트에 대한 스코어를 계산하는 것이 과제
    * 커멘트 목록이 주어지고, 상대적인 toxicity 에 따라 점수를 준다.
    * toxicity 가 높으면 점수도 높고, 반대면 점수도 낮다.
* 데이터셋 텍스트에는 profane(신성 모독), vulgar(외설적인), offensive(무례한, 모욕적인) 한 내용이 있을 수 있다.
* validation_data.csv : 누가 더 toxic한지 알려주는 데이터
    - less_toxic : 덜 toxic한 문장
    - more_toxic : 더 toxic한 문장
* comments_to_score.csv : 전체 text의 toxicity 점수를 계산하여 score로 제출

## EDA

* EDA file
    - 학습 데이터로 사용한 데이터들에 대한 EDA

## Preprocessing the train files

* Preprocessing file
    - 학습 데이터로 사용한 데이터들을 합치기 위한 전처리 작업

## Data Augmentation

toxic 데이터 증강

1. 번역
2. rephrase
3. rephrase * 번역

# PLM

* BERT
* Electra
* ...

## 학습 방법

* 1
    * (기본)회귀
        * 데이터
            * 있는대로
            * 개수를 작은 것에 맞추기
            * 증강
* 2
    * TF-IDF
* 3
    * TAPT
        * 동일한 데이터로 사전학습 후 파인튜닝
* 4
    * 4-1 이진 분류
        * 레이블 중 중간 스코어는 날리고 학습해서 하나의 값을 예측하게 함
        * 이 경우 극단값 더 잘 예측
    * 4-2 이진분류 - PET
        * 사전학습에서 사용한 학습 방법 파인튜닝에서 사용