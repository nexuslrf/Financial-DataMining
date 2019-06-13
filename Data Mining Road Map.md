# Data Mining Road Map

## Stage 1: Regression

* [ ] 虽然之前完成了一些，但还要改进，比如loss 的改进与约束；对label空间的再映射(**reweighting**, 咱们的场景对极端值很敏感，需要再考虑考虑)

## Stage 2: Feature Synthesis

Methods Mentioned:

* [ ] Simple methods such as addition, subtraction, multiplication, and polynomial combinations. Mathematical tools in signal processing such as [**WT (wavelet transform)**](https://pywavelets.readthedocs.io/).

* [ ] [Deep Feature Synthesis](https://www.featuretools.com/)

* [ ] [Stacked Auto Encoder](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0180944). A deep learning framework for financial time series using stacked auto-encoders and long-short term memory.

以上这几点估计都要做，具体做法嘛，就是把合成的数据去喂给第一问的各个model， 看看效果。

## Stage 3: Strategy

* [ ] RL 的financial strategies, 貌似可以不做了，不过[@ghy](https://github.com/gohsyi) 都有所了解了，能说点做点什么总归有用。
* [ ] 貌似有个必须完成的任务：the number of buying and spelling on testing set, the average price to buy and the average price to sell. Besides, attach action selection for each tick on testing set for submission.

## Stage 4: Others

* [ ] 整理代码，整理数据，作图，做论文。。。
