from utils import logging_utils
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from task import TaskOptimizer
import datetime
import config

# 设置好日志
logname = "hyperopt_optimizer_%s.log"%(datetime.datetime.now().strftime("%H-%m-%d-%H-%M"))
logger = logging_utils._get_logger(config.LOG_DIR, logname)

# 准备好数据
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
tag_vec = CountVectorizer(tokenizer=lambda x: x.split(','))
tag_vec.fit(train.tags)

X_train = pd.DataFrame(tag_vec.transform(train.tags), columns = tag_vec.get_feature_names())
y_train = train.is_apply
X_test= pd.DataFrame(tag_vec.transform(test.tags), columns = tag_vec.get_feature_names())
y_test = test.is_apply


#跑任务
optimizer = TaskOptimizer(X_train, y_train, X_test, y_test, cv = 5, max_evals = 2, verbose=True)
optimizer.run()


