# Ruddit

* paper: https://aclanthology.org/2021.acl-long.210.pdf
* data(github) : https://github.com/hadarishav/Ruddit
    * comment_id : 커멘트의 ID
        * comment_id 로 텍스트 가져와야함
    * post_id :  주어진 comment의 post
    * offensiveness_score: 해당 커멘트의 offensiveness score
        * -1 ~ 1
        * -1은 평범
        * 1은 엄청 공격적
    * 0~1로 스케일링해서 사용하기