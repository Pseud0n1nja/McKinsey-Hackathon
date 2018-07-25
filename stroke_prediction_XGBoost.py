#importing libraries
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from category_encoders import OrdinalEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
import xgboost as xgb

#splitting traing and testing set
stroke_train = pd.read_csv("/Users/Arpit/Documents/Hackathon/train.csv", sep = ',')

stroke_test = pd.read_csv("/Users/Arpit/Documents/Hackathon/test.csv", sep = ',')
stroke_test.info()

# drop test id
test_id = stroke_test.id
stroke_test = stroke_test.drop("id", axis = 1)
stroke_test.info()

# eda
stroke_train.stroke.value_counts()*100/stroke_train.shape[0]

#Checking Nulls 
stroke_train.isnull().values.any()
stroke_train.isnull().sum()
stroke_train.isnull().sum()*100/stroke_train.shape[0]


#embedding pipleines
#Preprocessing the data 


class DataFrameSelector(BaseEstimator, TransformerMixin):
	def __init__(self, attribute_names):
		self.attribute_names = attribute_names
	def fit(self, X, y=None):
		return self
	def transform(self, X):
		return X[self.attribute_names]

# custom imputer 
class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy='mode', filler='NA'):
        self.strategy = strategy
        self.fill = filler
        
    def fit(self, X, y=None):
        if self.strategy in ['mean','median']:
            if not all(X.dtypes == np.number):
                raise ValueError('dtypes mismatch np.number dtype is \
                                 required for '+ self.strategy)
        if self.strategy == 'mean':
            self.fill = X.mean()
        elif self.strategy == 'median':
            self.fill = X.median()
        elif self.strategy == 'mode':
            self.fill = X.mode().iloc[0]
        elif self.strategy == 'fill':
            if type(self.fill) is list and type(X) is pd.DataFrame:
                self.fill = dict([(cname, v) for cname,v in zip(X.columns, self.fill)])
        return self
    
    def transform(self, X, y=None):
        return X.fillna(self.fill)
    

# divide dataframe into predictors and target
X = stroke_train.drop("stroke", axis = 1)
y = stroke_train.stroke

# preprocess X
num_attributes = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', ]
cat_attributes = ['gender', 'ever_married',  'work_type', 'Residence_type', 'smoking_status' ]

num_pipeline = Pipeline([
				('selector', DataFrameSelector(num_attributes)),
				('imputer', Imputer()),
				('scaler', StandardScaler())
			])

cat_pipeline = Pipeline([
				('selector', DataFrameSelector(cat_attributes)),
				('imputer', CustomImputer(strategy='mode')),
				('label_encoder', OrdinalEncoder()),
				('one_hot_encoder', OneHotEncoder())  # avoid this step if too much categories in a column
			])

full_pipeline = FeatureUnion(transformer_list=[
					("num_pipeline", num_pipeline),
					("cat_pipeline", cat_pipeline)
			])

full_pipeline.fit(X)
X = full_pipeline.transform(X)

# mixed sampling
ros = RandomOverSampler(ratio='minority', random_state=4)
X, y = ros.fit_sample(X, y)

# divide into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 4, stratify = y)
			

# xg boost

# =============================================================================
# # =============================================================================
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
model.score(X_test, y_test)
# # =============================================================================
# =============================================================================

# =============================================================================
#from xgboost import XGBClassifier
#import xgboost as xgb
#
#dtrain = xgb.DMatrix(X_train, label = y_train)
#dtest = xgb.DMatrix(X_test, label = y_test)
#parameters = dict(booster = "gbtree", silent = 0, eta = 0.1, gamma = 0, max_depth = 6, min_child_weight = 1, subsample = 1, colsample_bytree = 1, colsample_bylevel = 1, alpha = 0, objective = "binary:logistic", eval_metric = ["error", "logloss"], num_classes = 2, seed = 4)
# 
#eval_list  = [(dtest,'eval'), (dtrain,'train')]
#model = xgb.train(parameters, dtrain, num_boost_round = 500, evals = eval_list, early_stopping_rounds = 10)
#y_pred = model.predict(dtest, ntree_limit = model.best_ntree_limit) 
#xgb.plot_importance(model)                      
#xgb.plot_tree(model, num_trees=2)     
# =============================================================================

# make predictions on test set
# preprocess test
stroke_test = full_pipeline.transform(stroke_test)
y_pred_test = model.predict(stroke_test)


# create df
df = pd.DataFrame(data = {'id': test_id, 'stroke': y_pred_test}, columns = ['id', 'stroke'])
df.head(100)

# submit predictions
df.to_csv("xgboost_default_submission_final.csv", sep = ',', index = False)























