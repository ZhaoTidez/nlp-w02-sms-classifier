# 数据来源与用途

SMS Spam Collection，Tiago Almeida 与 José María Gómez Hidalgo，UCI Machine Learning Repository。
引用：Almeida, T. & Hidalgo, J. (2011). https://doi.org/10.24432/C5CC84
官方页面：https://archive.ics.uci.edu/dataset/228/sms+spam+collection
许可：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。课堂进行去重、划分及JSONL/标签转换，保留原文。2026-09-29下载核验原始5574条，原说明见原始数据说明.txt。

本课仅从W04训练分区3609条进一步划分，后续课程验证和测试分区保持原样。

| 文件 | 条数 | 字段 | 用途 |
|---|---:|---|---|
| train.jsonl | 2525 | id,text,label | 用正文与标签学习词表、IDF及分类参数 |
| dev.jsonl | 541 | id,text,label | 查看错误，比较与选择方法 |
| test_texts.jsonl | 543 | id,text | 生成全部测试短信的预测答案 |

JSONL每行一条记录，id是短信编号，text是英文原文，label为ham（正常）或spam（垃圾）。测试文件不含参考标签。学生提交test_predictions.csv中的label是算法预测，不是参考答案。

模型只用train学习，在dev上选择方案。不得利用测试集学习词表、调参、外查标签或硬编码答案。学生包不提供测试分数；教师使用保留的参考标签评测。

按类别固定哈希初分后，把按小写英文字母提词得到相同内容的12组跨分区短信收拢至训练集，以20次同标签唯一文本交换保持分区大小。编号、划分和哈希见manifest.json。三分区ID与规范化全文无交集，仍可能有其他近似模板。历史英文短信的实验表现不能外推到所有现代消息。
