# Audio Retrieval with-Text Queries

In this project we consider the task of retrieving audio streams based on a textual query. Our approach is based on several previous works adapting them to a combined framework for learning a joint embedding model for textual inputs and audio outputs, improving semantic knowledge and enabling the useof textual queries to search and retrieve audio. We experiment an improvement of the approach using two methods of fusion (CombSum, RRF), achieving promising results. Furthermore, we explore queryper formance  prediction in the scenario of multiple queries expressing a common information need (the audio segment) and propose an evaluation metric for this task. At last, we propose a method to integratequery performance weights to the fusion process.

This repository is forked from the [Audio Retrieval with Natural Language Queries](https://arxiv.org/pdf/2105.02192.pdf) repo with our addtion of fusion.py, QPP Implementation.ipynb and the data folder. The dataset used in our work is [CLOTHO](https://arxiv.org/pdf/1910.09387.pdf).

More information for retrival can be found at: https://github.com/oncescuandreea/audio-retrieval

### Requirements

The original paper used PyTorch 1.7.1., CUDA 10.1, and Python 3.7 to generate results and models. The required libraries for running this code (inculding our changes) can be found in `requirements/requirements.txt`.

```
pip install -r requirements/requirements.txt
```

To be able to run the original code, follow the instractions specified at the original repo. In order to run our additional code, all of the required data is in the Data folder. 

### Reproducing fusion results

To reproduce the fusion:
Run the `fusion.py` script run the following line:
```
python fusion.py --config "configs/clotho/train-vggish-vggsound.json" --device 0 --eval_from_training_config
```
### Fusion methods evaluation results on CLOTHO dataset

| Experts  | R@1 | R@10 |
| ----- | ---- | --- | 
| CE - VGGish + VGGSound  (base-line)  | 6.7 | 33.2 | 
| combSUM   | 10.0  | 46.5 | 
| RRF   | 9.7  | 45.9 | 
| Weighted combSUM  | 6.0  | 33.3 |

### Query performance prediction models evaluation
| Experts | Kendall's tau | Kendall's tau Inspired | 
| ----- | ---- | --- | 
| Linear Regression  | -0.33  | 43.28 | 
| MLP    | 1.43  | 42.25 | 
| XGBoost  | -0.67  | 43.49 | 

#### This code is based on https://github.com/oncescuandreea/audio-retrieval

