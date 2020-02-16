import pandas as pd
from math import sqrt
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import AdaBoostRegressor as ABR
from sklearn.ensemble import BaggingRegressor as BR
from sklearn.ensemble import ExtraTreesRegressor as ETR
from sklearn.neural_network import MLPRegressor as MLPR
from sklearn.linear_model import LinearRegression as LR
from xgboost import XGBRegressor


# Import data
train = pd.read_csv('train.csv', header=0, sep=",", encoding="ISO-8859-1")
test = pd.read_csv('test.csv', header=0, sep=",", encoding="ISO-8859-1")

# DataFrame
target = train['relevance']
piv_train = train.shape[0]
df = pd.read_csv('dataframe.csv', header=0, sep=",", encoding="ISO-8859-1")

df_model = df

# for col in df_model.columns:
#     print(col)

# Replace NA
for col in df_model.columns:
    df_model[col].fillna(value=df_model[col].mean(), inplace=True)

# Scale data
scaler = MinMaxScaler(feature_range=(1, 3))
df_model = df_model.drop("product_uid", axis=1)
identite = df_model.id
df_model = pd.DataFrame(data = scaler.fit_transform(df_model), columns=df_model.columns)
df_model.id = identite


models = [KNR(weights='distance', p=1, n_neighbors=13, leaf_size=10, algorithm='auto'), 
          # SVR(),
          LR(normalize=True,fit_intercept=True),
          BR(n_estimators=500, max_samples=0.8, max_features=0.6),
          DTR(max_features='sqrt', max_depth=7),
          RFR(n_estimators=100, min_samples_split=2, max_features=0.4, max_depth=7, bootstrap=False),
          MLPR(),
          ABR(n_estimators=50),
          ETR(max_features=None, max_depth=20),
          XGBRegressor(subsample=1, n_estimators=100, min_child_weight=2, max_depth=7, gamma=0.1, colsample_bytree=1.0, objective='reg:squarederror')] 


X = df_model.iloc[0:piv_train]
y = target.values
n_splits = 10
kf = KFold(n_splits=n_splits, random_state=2019)
new_features = pd.DataFrame({"id": df.iloc[0:piv_train].id})

print("Getting features for train dataset")
for idx, mod in enumerate(models):
    print(idx, " : ", mod)
    rmse = []
    y_pred = []
    y_id = []
    n=1
    for train_index, cv_index in kf.split(X, y):
        X_train, X_cv = X.iloc[train_index], X.iloc[cv_index]
        y_train, y_cv = y[train_index], y[cv_index]
        
        y_id += list(X_cv.id.values)
        
        X_train = X_train.drop("id", axis=1)
        X_cv = X_cv.drop("id", axis=1)
        
        mod.fit(X_train, y_train)       
        y_prediction = mod.predict(X_cv)
        y_pred += list(y_prediction)
        mse = mean_squared_error(y_cv, y_prediction)
        print("iteration ", n, " :: mse=", mse, " RMSE=", sqrt(mse))
        n+=1
      
    df_f = pd.DataFrame({"id" : y_id, "feature_model_" + str(idx) : y_pred})
    new_features = pd.merge(new_features, df_f, on="id")
    print("Modele terminé")

# Saving new features (train)
dataset_enrichi = pd.merge(train["id"], new_features, on='id')
dataset_enrichi.to_csv('new_features_regressor_train.csv', index=False)

# Repeat for test dataset
X = df_model.iloc[0:piv_train]
X_test = df_model.iloc[piv_train:df.shape[0]]
X_train = X.drop("id", axis=1)
X_test = X_test.drop("id", axis=1)
y = target.values
y_id = df.iloc[piv_train:df.shape[0]].id
new_features_test = pd.DataFrame({"id": y_id})

print("Getting features for test dataset")
for idx, mod in enumerate(models):
    print(idx, " : ", mod)
    mod.fit(X_train, y)
    y_pred = list(mod.predict(X_test))
    df_f = pd.DataFrame({"id" : y_id, "feature_model_" + str(idx) : y_pred})
    new_features_test = pd.merge(new_features_test, df_f, on="id")
    print("Modele terminé")

# Saving new features (train)
dataset_enrichi_test = pd.merge(df.iloc[piv_train:df.shape[0]].id, new_features_test, on='id')
dataset_enrichi_test.to_csv('new_features_regressor_test.csv', index=False)