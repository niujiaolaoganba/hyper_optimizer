import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from task import TaskOptimizer



# 准备好数据
train = pd.read_csv('../data/input/train.csv')
test = pd.read_csv('../data/input/test.csv')
tag_vec = CountVectorizer(tokenizer=lambda x: x.split(','))
tag_vec.fit(train.tags)

X_train = pd.DataFrame(tag_vec.transform(train.tags).toarray(), columns = tag_vec.get_feature_names())
y_train = train.is_apply
X_test= pd.DataFrame(tag_vec.transform(test.tags).toarray(), columns = tag_vec.get_feature_names())
y_test = test.is_apply


#跑任务
optimizer = TaskOptimizer(X_train, y_train, X_test, y_test, cv = 3, max_evals = 9, verbose=True)
optimizer.run()


