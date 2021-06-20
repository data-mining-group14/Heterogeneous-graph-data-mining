# Node Classification and Link Prediction on Heterogeneous Academic Network
This is the course project of EE226 Massive Data Mining.

 conda env create -f py38.yaml
 conda install -c stellargraph stellargraph

 python data_preprocess_by_author.py
 python merge_coauthor_reference.py
 python node2vec-link-prediction_multilevel.py
 python vote.py