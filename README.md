# Node Classification and Link Prediction on Heterogeneous Academic Network

This is the course project of SJTU EE226 Massive Data Mining.
We achieved 1st in the link prediction task and 4th in the node classification task.

Kaggle competition: 
Node classification:https://www.kaggle.com/c/EE226-2021spring-problem1. 
Link prediction:https://www.kaggle.com/c/EE226-2021spring-problem2.

## Requirements
You can install the environment needed with
```bash
conda env create -f py38.yaml
```

## Usage
To get the result for node classification:
```bash
cd Nodeclassification
python votefinal.py 
```
To get the result for link prediction: 
```bash
cd Linkprediction
sh linkpre.sh
```

## References
[1] [Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://github.com/tkipf/pygcn)