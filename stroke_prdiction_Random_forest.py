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

# divide train data into train_0 and train_1
train_0 = stroke_train.loc[stroke_train.stroke == 0,:]
train_1 = stroke_train.loc[stroke_train.stroke == 1,:]

train_0.stroke.value_counts()*100/train_0.shape[0]
train_1.stroke.value_counts()*100/train_1.shape[0]

# subsample train_0 - use frac =  5%
train_0 = train_0.sample(frac=0.1, replace=False)

# concatenate train_0 and train_1
stroke_train = pd.concat([train_0, train_1], axis=0)
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

# divide into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 4, stratify = y)
			

# random forest
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 1000, max_features = 4, oob_score = True, random_state = 4)
model.fit(X_train, y_train)
model.score(X_test, y_test)
model.oob_score_
model.feature_importances_


# make predictions on test set
# preprocess test
stroke_test = full_pipeline.transform(stroke_test)
stroke_test.shape

# predict on test
y_pred_test = model.predict(stroke_test)

# create df
df = pd.DataFrame(data = {'id': test_id, 'stroke': y_pred_test}, columns = ['id', 'stroke'])
df.head(100)

# submit predictions
df.to_csv("forest_submission.csv", sep = ',', index = False)


























