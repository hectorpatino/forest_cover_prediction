import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import seaborn as sns

from feature_engine.discretisation import EqualWidthDiscretiser, EqualFrequencyDiscretiser
from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures, DropCorrelatedFeatures
from feature_engine.encoding import RareLabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


# %% md

# Custom Transformers

# %%

class DistanceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['EuclideanDistanceHidroloy'] = np.around(
            np.sqrt(X['Horizontal_Distance_To_Hydrology'] ** 2 +
                    X['Vertical_Distance_To_Hydrology'] ** 2),
            4)
        X.drop(['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology'], axis=1, inplace=True)
        return X


# %%

class DropIdentifierFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X.drop('Id', axis=1, inplace=True)
        return X


# %%
from sklearn.utils.validation import check_is_fitted
class FromDummiesToCategories(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_operate):
        self.cols_to_operate = set(cols_to_operate)

    def fit(self, X, y=None):
        self.columns_to_keep_ = set(X.columns.tolist()) - self.cols_to_operate
        return self

    def transform(self, X):
        check_is_fitted(self, 'columns_to_keep_')
        X = X.copy()
        X = X[self.columns_to_keep_]
        return X


# %% md

# Load Data

# %%

data = pd.read_csv(r'../../../data/train.csv')
data.head()

# %% md

X_train, X_test, y_train, y_test = train_test_split(data.drop('Cover_Type', axis=1),
                                                    data['Cover_Type'],
                                                    test_size=.2,
                                                    random_state=42)
# %%

soil_columns = [x for x in data.columns if x.startswith('Soil_Type')]

# %%

pipeline_list_1 = [
    ('dropuniquefeatures', DropIdentifierFeatures()),
    ('du', FromDummiesToCategories(cols_to_operate=soil_columns)),
    # ('dp', DropConstantFeatures(tol=0.99)),
    # ('dd', DropDuplicateFeatures()),
    # ('dt', DistanceTransformer()),
    # ('dteq', EqualWidthDiscretiser(bins=15, variables=['EuclideanDistanceHidroloy'])),
    # ('ed', EqualWidthDiscretiser(bins=36, variables=['Aspect'])),
    # ('edh', EqualWidthDiscretiser(bins=26, variables=['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm'])),
    # ('dcf', DropCorrelatedFeatures(variables=None, method='pearson', threshold=0.80)),
]

# %%

pipe_2 = Pipeline(pipeline_list_1)
X_train_pipe_2 = pipe_2.fit_transform(X_train)
X_train_pipe_2.head()

# %%

pipe_list_1_rf = [('rf', RandomForestClassifier())]
pipe_2_rf = Pipeline(pipeline_list_1 + pipe_list_1_rf)
rfpg1 = {
    'rf__n_estimators': [150,],
    'rf__max_depth': [20,],
    'rf__min_samples_split': [4],
    'du__cols_to_operate': soil_columns,
}
grid1_pipe_2 = GridSearchCV(pipe_2_rf, param_grid=rfpg1, cv=5, n_jobs=-1, verbose=3, scoring='precision')
grid1_pipe_2.fit(X_train, y_train)