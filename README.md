
如果你想成功应对低资源或者小样本学习的场景，每个类只有1个样本或者几个样本，巧了那么这份代码正好能帮你实现好的准确率。
目前验证在自有中文分类上数据集准确率，对比bert-fintune效果

| exp | 5-way 1-shot | 5-way 5-shot | 10-way 1-shot |10-way 5-shot |
| :------| :------: | :------: |:------: |:------: |
|[basline]| 80.98% | 88.73%  | 71.20% | 80.71% |
|[ours]| 86.30% | 84.80%  | 72.10% | 92.20% |


1）完整支持 BERT 模型训练到部署, 包括:

- 支持 BERT GPU 单机、分布式预训练
- 支持 BERT GPU 多卡 Fine-tuning
- 提供 BERT 预测接口 demo, 方便多硬件设备生产环境的部署

2）支持 FP16/FP32 混合精度训练和 Fine-tuning，节省显存开销、加速训练过程；

3）提供转换成 Paddle Fluid 参数格式的 [BERT 开源预训练模型](https://github.com/google-research/bert) 供下载，以进行下游任务的 Fine-tuning, 包括如下模型:


| Model | Layers | Hidden size | Heads |Parameters |
| :------| :------: | :------: |:------: |:------: |
|[BERT-Large, Uncased (Whole Word Masking)](https://bert-models.bj.bcebos.com/wwm_uncased_L-24_H-1024_A-16.tar.gz)| 24 | 1024 | 16 | 340M |
|[BERT-Large, Cased (Whole Word Masking)](https://bert-models.bj.bcebos.com/wwm_cased_L-24_H-1024_A-16.tar.gz)| 24 | 1024 | 16 | 340M |
|[RoBERTa-Base, Chinese](https://bert-models.bj.bcebos.com/chinese_roberta_wwm_ext_L-12_H-768_A-12.tar.gz) | 12 | 768 |12 |110M |
|[RoBERTa-Large, Chinese](https://bert-models.bj.bcebos.com/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16.tar.gz) | 24 | 1024 |16 |340M |
|[BERT-Base, Uncased](https://bert-models.bj.bcebos.com/uncased_L-12_H-768_A-12.tar.gz) | 12 | 768 |12 |110M |
|[BERT-Large, Uncased](https://bert-models.bj.bcebos.com/uncased_L-24_H-1024_A-16.tar.gz) | 24 | 1024 |16 |340M |
|[BERT-Base, Cased](https://bert-models.bj.bcebos.com/cased_L-12_H-768_A-12.tar.gz)|12|768|12|110M|
|[BERT-Large, Cased](https://bert-models.bj.bcebos.com/cased_L-24_H-1024_A-16.tar.gz)|24|1024|16|340M|
|[BERT-Base, Multilingual Uncased](https://bert-models.bj.bcebos.com/multilingual_L-12_H-768_A-12.tar.gz)|12|768|12|110M|
|[BERT-Base, Multilingual Cased](https://bert-models.bj.bcebos.com/multi_cased_L-12_H-768_A-12.tar.gz)|12|768|12|110M|
|[BERT-Base, Chinese](https://bert-models.bj.bcebos.com/chinese_L-12_H-768_A-12.tar.gz)|12|768|12|110M|

每个压缩包都包含了模型配置文件 `bert_config.json`、参数文件夹 `params` 和词汇表 `vocab.txt`；

4）支持 BERT TensorFlow 模型到 Paddle Fluid 参数的转换。


## 目录结构
```text
.
├── data                     # 示例数据
├── inference                # 预测部署示例
├── model                    # 模型定义
├── reader                   # 数据读取
├── utils                    # 辅助文件
├── batching.py              # 构建 batch 脚本
├── convert_params.py        # 参数转换脚本
├── optimization.py          # 优化方法定义
├── predict_classifier.py    # 分类任务生成 inference model
|── run_classifier.py        # 分类任务的 fine tuning
|── run_squad.py             # 阅读理解任务 SQuAD 的 fine tuning
|── test_local_dist.sh       # 本地模拟分布式预训练
|── tokenization.py          # 原始文本的 token 化
|── train.py                 # 预训练过程的定义
|── train.sh                 # 预训练任务的启动脚本
```

## 安装
本项目依赖于 Paddle Fluid **1.7.1** 及以上版本，请参考[安装指南](http://www.paddlepaddle.org/#quick-start)进行安装。如果需要进行 TensorFlow 模型到 Paddle Fluid 参数的转换，则需要同时安装 TensorFlow 1.12。


5）
数据格式
./data/all/txt
每行两列
data   \t  label
自动划分训练和测试集

6)运行方法

sh .run_ce.sh
 
k-way n-shot

├── k                        # k分类
├── n                        # 每类n个样本
├── q                        # query_set 样本个数


