"""可编辑的分类器起点。只用传入的训练正文和标签学习，不读取开发/检查标签。"""
from nltk.tokenize import RegexpTokenizer

class RuleClassifier:
    def __init__(self, keywords, min_hits=1):
        self.keywords=set(keywords)
        self.min_hits=min_hits
        self.tokenizer=RegexpTokenizer(r'[A-Za-z]+')
    def predict(self, texts):
        return ['spam' if len(set(self.tokenizer.tokenize(t.lower())) & self.keywords)>=self.min_hits else 'ham' for t in texts]

def build_classifier(train_texts, train_labels, method, config):
    """返回有 predict(texts) 方法的对象；该方法按输入顺序返回 ham/spam 列表。"""
    if method == 'rule':
        return RuleClassifier(config['keywords'], config.get('min_hits',1))
    if method in ['nb', 'lr']:
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.linear_model import LogisticRegression
        # 词表与 IDF 只在 train_texts 上学习。预测时只 transform，不重新 fit。
        vectorizer=TfidfVectorizer(lowercase=True, ngram_range=tuple(config.get('ngram_range',[1,1])))
        classifier=(MultinomialNB(alpha=config.get('alpha',1.0)) if method=='nb' else
                    LogisticRegression(C=config.get('C',1.0), max_iter=1000, solver='liblinear', random_state=42,
                                       class_weight=config.get('class_weight')))
        model=Pipeline([('tfidf',vectorizer),('classifier',classifier)])
        model.fit(train_texts, train_labels)
        return model
    if method == 'custom':
        # 在本文件中定义自己的类，在此创建/训练并返回它。
        # 例如 return MyClassifier(...).fit(train_texts, train_labels)
        raise NotImplementedError('请在 student_model.py 的 custom 分支接入自己的分类器。')
    raise ValueError('method 请填 rule、nb、lr 或 custom。')
