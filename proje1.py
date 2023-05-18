
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

warnings.simplefilter(action='ignore', category=Warning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

##   EXPLORE DATA   ##

train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

train_df["SalePrice"]
#saleprice test df te yok
train_df.info()
test_df.info()

df_ = pd.concat([train_df, test_df])
df = df_.copy()
df.info()

# hafta 8 case ini incele miuul çözüm olan
# uçtan uca ml ek ders ve aykırı gözlem ek ders izle
# feature engineering kısmına kadar olan bölümlerde farklı görselleştirmeler kullan


# 1. Genel Resim
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
# 5. Korelasyon Analizi (Analysis of Correlation)

def check_df(dataframe):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)

def grab_col_names(dataframe, cat_th=10, car_th=20):


    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')


    return cat_cols, cat_but_car, num_cols

cat_cols, cat_but_car, num_cols = grab_col_names(df)
#
# def cat_summary(dataframe, col_name, plot=False):
#     print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
#                         "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
#
#     if plot:
#         sns.countplot(x=dataframe[col_name], data=dataframe)
#         plt.show(block=True)
#
#
# for col in cat_cols:
#     cat_summary(df, col, True)
#
# def num_summary(dataframe, numerical_col, plot=False):
#     quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
#     print(dataframe[numerical_col].describe(quantiles).T)
#
#     if plot:
#         dataframe[numerical_col].hist(bins=50)
#         plt.xlabel(numerical_col)
#         plt.title(numerical_col)
#         plt.show(block=True)
#
#     print("#####################################")
#
# for col in num_cols:
#     num_summary(df, col, False)
#
#
# def target_summary_with_cat(dataframe, target, categorical_col):
#     print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")
#
#
# for col in cat_cols:
#     target_summary_with_cat(df, "SalePrice", col)
#     #görselleştir
#
#
# df["SalePrice"].hist(bins=100)
# plt.show(block=True)
#
# np.log1p(df['SalePrice']).hist(bins=100)
# plt.show(block=True)
#
# corr = df[num_cols].corr()
#
# sns.set(rc={'figure.figsize': (12, 12)})
# sns.heatmap(corr, cmap="RdBu")
# plt.show(block=True)


#
# def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
#     corr = dataframe.corr()
#     cor_matrix = corr.abs()
#     upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
#     drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
#     if plot:
#         import seaborn as sns
#         import matplotlib.pyplot as plt
#         sns.set(rc={'figure.figsize': (15, 15)})
#         sns.heatmap(corr, cmap="RdBu")
#         plt.show(block=True)
#     return drop_list
#
# high_correlated_cols(df, plot=True)

## preprocess and feature engineering

def outlier_thresholds(dataframe, col_name, q1 = 0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquentile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquentile_range
    low_limit = quartile1 - 1.5 * interquentile_range
    return low_limit, up_limit

# def check_outlier(dataframe, col_name):
#     low_limit, up_limit = outlier_thresholds(dataframe, col_name)
#     if dataframe[(dataframe[col_name]< low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
#         return True
#
#     else:
#         return False
#
#
# for col in num_cols:
#     print(col, check_outlier(df, col))

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(df,col)
#
# for col in num_cols:
#     print(col, check_outlier(df, col))

# df.isnull().sum().sort_values(ascending=False)

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns, n_miss

missing_values_table(df)


no_cols = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu",
           "GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]

for col in no_cols:
    df[col].fillna("NO",inplace=True)

missing_values_table(df)


def quick_missing_imp(data, num_method="median", cat_length=20, target="SalePrice"):
    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir

    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    # değişken object ve sınıf sayısı cat_lengthe eşit veya altındaysa boş değerleri mode ile doldur
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

    # num_method mean ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # num_method median ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data

df = quick_missing_imp(df, num_method="median", cat_length=17)

# def rare_analyser(dataframe, target, cat_cols):
#     for col in cat_cols:
#         print(col, ":", len(dataframe[col].value_counts()))
#         print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
#                             "RATIO": dataframe[col].value_counts() / len(dataframe),
#                             "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
#
# for col in df.columns:
#     rare_analyser(df, "SalePrice", cat_cols)


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


df_with_rare = rare_encoder(df,0.02)
df_with_rare.head()

# for col in df.columns:
#     rare_analyser(df_with_rare, "SalePrice", cat_cols)

df = df_with_rare.copy()


ngb =df.groupby("Neighborhood").SalePrice.mean().reset_index()
ngb["CLUSTER_NEIGHBORHOOD"] = pd.cut(df.groupby("Neighborhood").SalePrice.mean().values, 4, labels = range(1,5))
ngb.groupby("CLUSTER_NEIGHBORHOOD").SalePrice.agg({"count", "mean", "max"})
df = pd.merge(df, ngb.drop(["SalePrice"], axis = 1), how = "left", on = "Neighborhood")
df.groupby("CLUSTER_NEIGHBORHOOD").SalePrice.agg({"count", "median", "mean", "max", "std"})
del ngb


df["NEW_STREET_BASE"] = df["LotFrontage"] * df["TotalBsmtSF"]
df["NEW_STREET_2ND_FLOOR"] = df["LotFrontage"] * df["2ndFlrSF"]
df["NEW_STREET_VENEER"] = df["LotFrontage"] * df["MasVnrArea"]
df["NEW_VENEER_2ND_FLOOR"] = df["MasVnrArea"] * df["2ndFlrSF"]
df["NEW_STREET_BASE"] = df["MSSubClass"] * df["TotalBsmtSF"]
df["NEW_TYPE_2ND_FLOOR"] = df["MSSubClass"] * df["2ndFlrSF"]
df["NEW_TYPE_2ND_GARAGE"] = df["MSSubClass"] * df["GarageArea"]
df["NEW_GARAGE_2ND_FLOOR"] = df["GarageArea"] * df["2ndFlrSF"]
df["NEW_LIVING_AREA_GARAGE"] = df["GrLivArea"] * df["GarageArea"]
df["NEW_LIVING_AREA_BASE"] = df["GrLivArea"] * df["TotalBsmtSF"]
df["NEW_LIVING_AREA_2ND_FLOOR"] = df["GrLivArea"] * df["2ndFlrSF"]
df["NEW_LIVING_AREA_VENEER"] = df["GrLivArea"] * df["MasVnrArea"]
df["NEW_PORCH_LIVING_AREA"] = df["OpenPorchSF"] * df["GrLivArea"]
df["NEW_PORCH_GARAGE"] = df["OpenPorchSF"] * df["GarageArea"]
df["NEW_PORCH_TYPE"] = df["OpenPorchSF"] * df["MSSubClass"]
df["NEW_TotalFlrSF"] = df["1stFlrSF"] + df["2ndFlrSF"]
df["NEW_PORCH_NEW_STREET"] = df["OpenPorchSF"] * df["LotFrontage"]
df["NEW_PORCH_BASE"] = df["OpenPorchSF"] * df["LotFrontage"]
df["NEW_LotRatio"] = df.GrLivArea / df.LotArea
df["NEW_TotalHouseArea"] = df.NEW_TotalFlrSF + df.TotalBsmtSF
df["NEW_RatioArea"] = df.NEW_TotalHouseArea / df.LotArea
df["NEW_GarageLotRatio"] = df.GarageArea / df.LotArea
df["NEW_PorchArea"] = df.OpenPorchSF + df.EnclosedPorch + df.ScreenPorch + df["3SsnPorch"] + df.WoodDeckSF
df["NEW_DifArea"] = (df.LotArea - df["1stFlrSF"] - df.GarageArea - df.NEW_PorchArea - df.WoodDeckSF)
df["NEW_OverallGrade"] = df["OverallQual"] * df["OverallCond"]
df["NEW_Restoration"] = df.YearRemodAdd - df.YearBuilt
df["NEW_HouseAge"] = df.YrSold - df.YearBuilt
df["NEW_RestorationAge"] = df.YrSold - df.YearRemodAdd
df["NEW_GarageAge"] = df.GarageYrBlt - df.YearBuilt
df["NEW_GarageRestorationAge"] = np.abs(df.GarageYrBlt - df.YearRemodAdd)
df["NEW_GarageSold"] = df.YrSold - df.GarageYrBlt
df["NEW_MasVnrRatio"] = df.MasVnrArea / df.NEW_TotalHouseArea
df["TOTALBATH"] = df.BsmtFullBath + df.BsmtHalfBath*0.5 + df.FullBath + df.HalfBath*0.5
df["TOTALFULLBATH"] = df.BsmtFullBath + df.FullBath
df["TOTALHALFBATH"] = df.BsmtHalfBath + df.HalfBath
df["ROOMABVGR"] = df.BedroomAbvGr + df.KitchenAbvGr
df["RATIO_ROOMABVGR"] = df["ROOMABVGR"] / df.TotRmsAbvGrd  # bedroom ve kitchen'ın evin içindeki oranı
df["REMODELED"] = np.where(df.YearBuilt == df.YearRemodAdd, 0 ,1)   # ev restore edildimi
df["ISNEWHOUSE"] = np.where(df.YearBuilt == df.YrSold, 1 ,0)   # ev yeni mi
df['TOTAL_FLRSF'] = df['1stFlrSF'] + df['2ndFlrSF'] # evin 1 ve ikici katı toplam metrekaresi
df['FLOOR'] = np.where((df['2ndFlrSF'] < 1), 1,2)  # evin ikinci katı var mı ?
df['TOTAL_HOUSE_AREA'] = df.TOTAL_FLRSF + df.TotalBsmtSF
df["NEW_TotalBsmtFin"] = df.BsmtFinSF1 + df.BsmtFinSF2
df["NEW_TotalSqFeet"] = df.GrLivArea + df.TotalBsmtSF
df.drop(["MoSold", "YrSold"],axis = 1, inplace = True)  # bu değişkenlere gerek kalmadığı için siliyorum.

drop_list = ["Street", "Alley", "LandContour", "Utilities", "LandSlope","Heating", "PoolQC", "MiscFeature","Neighborhood"]
df.drop(drop_list, axis=1, inplace=True)

cat_cols, cat_but_car, num_cols = grab_col_names(df)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and len(df[col].unique()) == 2]

for col in binary_cols:
    label_encoder(df, col)


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

def min_max_scaler(dataframe, num_col):
    scaler = MinMaxScaler()

    dataframe[num_col] = scaler.fit_transform(dataframe[[num_col]])
    return dataframe

for col in num_cols:
    min_max_scaler(df, col)

len(df.columns)



train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()]

y = train_df['SalePrice']
X = train_df.drop(["Id", "SalePrice"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

# models = [('LR', LinearRegression()),
#           #("Ridge", Ridge()),
#           #("Lasso", Lasso()),
#           #("ElasticNet", ElasticNet()),
#           ('KNN', KNeighborsRegressor()),
#           ('CART', DecisionTreeRegressor()),
#           ('RF', RandomForestRegressor()),
#           #('SVR', SVR()),
#           ('GBM', GradientBoostingRegressor()),
#           ("XGBoost", XGBRegressor(objective='reg:squarederror')),
#           ("LightGBM", LGBMRegressor())]
#           # ("CatBoost", CatBoostRegressor(verbose=False))]
#
# for name, regressor in models:
#     rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
#     print(f"RMSE: {round(rmse, 4)} ({name}) ")

# RMSE: 63372.725 (LR)
# RMSE: 42740.5492 (KNN)
# RMSE: 41869.2228 (CART)
# RMSE: 28594.8214 (RF)
# RMSE: 25994.5967 (GBM)
# RMSE: 27985.414 (XGBoost)
# RMSE: 28031.6646 (LightGBM)


# hiperparametre optimizasyonlarını gerçekleştiriniz.

lgbm_model = LGBMRegressor(random_state=46)

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [100, 300, 500, 1000],
               "colsample_bytree": [0.5, 0.7, 1]}


lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)

flgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)

test_pred = np.expm1(flgbm_final.predict(test_df.drop(["Id","SalePrice"], axis = 1), num_iteration=flgbm_final.best_iteration_))

test_df["SalePrice"] = test_pred
test_df.to_csv("results12.csv", columns=["Id", "SalePrice"], index=False)


#TODO: kaggle dan house price predictions ile ilgili kodlara bak

# feature importance
def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig("importances.png")

model = LGBMRegressor()
model.fit(X, y)

plot_importance(model, X, 50)


# test

predictions = flgbm_final.predict(test_df.drop(["Id","SalePrice"], axis=1))

dictionary = {"Id":test_df.index, "SalePrice":predictions}
dfSubmission = pd.DataFrame(dictionary)
dfSubmission.to_csv("housePricePredictions.csv", index=False)