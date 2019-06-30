# Financial-DataMining

This repo contain codes to predict future stock price by machine learning algorithms.

## Models

A series of ML models are implemented.

* Random Forest
* AdaBoost
* XGBoost
* DNNs
* CNNs
* RNNs
* Transformer

## Environment

* Python 3.6
* Scikit-Learn 0.20
* XGBoost
* PyTorch1.0

## Dataset

Due to credential reason, raw dataset cannot be publicised for now. Processed data may be uploaded later.

Raw data is first processed via [Data_Preprocess.ipynb](Data_Preprocess.ipynb), which is fundamental step for whole project.

## How to RUN 

* For those ensemble models, simply run corresponding jupyter notebooks cell by cell

* For NN models, go to [./Regression_NN](./Regression_NN) run `Training.py`

  `Training.py` has a series of parameter you can adjust. For the details you may need to read the `Training.py` source code.

  Here is an example for training Transformer model:

  ```bash
  python Training.py --batch_size 1024 --prev_step 30  --suffix pre30_class_raw --num_classes 1 --epochs 30 --feature_size 256 --normalize --layers 1  --model Transformer
  ```

## Results to show (MSE metric)

| Model             | Random Forest | AdaBoost | XGBoost | CNN   | DNN   | GRU   | LSTM  | Transformer |
| ----------------- | ------------- | -------- | ------- | ----- | ----- | ----- | ----- | ----------- |
| Raw Data          | 0.555         | 0.601    | 0.546   | 0.551 | 0.554 | 0.548 | 0.549 | 0.545       |
| DWT               | 0.512         | 0.587    | 0.481   | 0.495 | 0.488 | 0.489 | 0.485 | 0.482       |
| DWT merge         | 0.490         | 0.582    | 0.464   | 0.486 | 0.473 | 0.486 | 0.484 | 0.471       |
| SAE               |               |          |         | 0.543 |       |       |       |             |
| Feature Synthesis | 0.554         | 0.593    | 0.545   |       |       |       |       |             |

## Trading Strategies:

Codes are in one of my partners' [repo](<https://github.com/gohsyi/trading_strategy>)
