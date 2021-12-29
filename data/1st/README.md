# 첫번째 대회 데이터

* https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
* source : wikipedia
* labels : toxic, severe_toxic, obscene(외설적), threat(위협적), insult(모욕적), identity_hate
* 과제 : 입력 커멘트가 각 레이블에 속할 확률을 구함(multi-label classification)
* severe_toxic 은  toxic 에 포함되어 있음
    * severe_toxic 이면 toxic 에 포함
    * 하지만 toxic 이어도 severe_toxic 이 아닐 수 있음
* test_labels.csv 에서 0 은 오답, 1은 해당, 
    * -1은 안쓰인 데이터 지우기

## files

* train_1st.csv : 첫번째 대회 데이터 중 학습, 테스트 데이터 합쳐서 전처리한 파일
* train_pp.csv : train_1st.csv 에서 80% 데이터(학습용)
* dev_pp.csv : train_1st.csv 에서 20% 데이터(검증용)