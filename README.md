# Impact of Baselines in Path attribution methods on BERT-based text classification task

## About The Project

This project intends to apply *integrated gradients*, an gradient-based path attribution methods algorithm, to the task of bert-based text classification, and to investigate the effect of different baselines on the results.



## Getting Started

### Prerequisites

1. Install the open-source-distribution [anaconda](https://www.anaconda.com/products/individual).
2. Create a new environment with python 3.9 and activate it.

```
conda create -n iML-project python=3.9
conda activate iML-project
```

3. Install requirements with `pip install -r requirements.txt`.



### Usage

Here are some examples of commands:

* Generate the KNN graph

```
python knn.py -nbrs 500 -distance l2
```
You can also download the prepared pickel file:
[500-nearest neighbors with L2 distance](https://drive.google.com/file/d/1GBVWZxIBK6HCGt6FcODPQO9hsKyZWeVv/view?usp=sharing)   / [500-nearest neighbors with L1 distance](https://drive.google.com/file/d/1a87Y62yFWM5Xy5AeFhr-sO35ht5ybEcg/view?usp=sharing)

* Visualize the Sum of Cumulative gradients and the interpolated inputs with different baseline

```
python main.py -baseline ['zero', 'constant', 'max', 'blurred', 'uniform'] -method ['IG','DIG'] -step 50 -seed 42
```
* Discretized Integrated Gradients with different baseline and Greedy strategy
```
python main.py -baseline ['zero', 'constant', 'max', 'blurred', 'uniform'] -method DIG -strategy greedy -step 50 -seed 42
```

* Evaluation different baseline

```
python main.py -baseline ['zero', 'constant', 'max', 'blurred', 'uniform'] -method ['IG','DIG']  -topk 30 -step 50 -seed 42
```

Each run will show how much the prediction confidence drops after ablation topk% of text when using the current method and baseline. 

Commands can be changed for other settings.

