# Data Info

## Raw Data
* We use jigsaq competition datasets (1st, 2nd and ruddit)
* For details on data, refer to the documents in each data directory

##
* We manage dataset in minio server.
* For load dataset, Use getData() in [exprements/utils.py](https://github.com/team-ok-nlp/jigsaw-toxic-competition/blob/9680d93458a2280b191faa62bce7d2b458e8c522/experiments/utils.py)
* getData() is get and save data from minio server when it is not in local dir