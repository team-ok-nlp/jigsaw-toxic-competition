# 두번째 대회 데이터

* https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data
    * all_data.csv 사용
        * 구분 전의 전체 데이터
* source : 2017 Civil Comments
    * comment_text : 커멘트
    * toxicity 종류
        * 각각이 0~1 값
        * toxicity
        * severe_toxicity
        * obscene
        * threat
        * insult
        * identity_attack
        * sexual_explicit
    * 나머지 columns(identities) : 키워드가 등장 여부, 
        * 해당 키워드가 나온다고 무조건 toxic 한게 아님을 볼 수 있다.
    * identity_annotator_count : 해당 커멘트에 대해 identity 평가를 몇명이나 했는지
    * toxicity_annotator_count : 해당 커멘트에 대해 toxicity 평가를 몇명이나 했는지
        * 5명 이상인 커멘트에 대해 실시하면 신뢰도 높임
