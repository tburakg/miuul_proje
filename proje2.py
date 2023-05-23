
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from lightgbm import LGBMRegressor
from sklearn.exceptions import ConvergenceWarning

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn import metrics

warnings.simplefilter(action='ignore', category=Warning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


train_df = pd.read_csv("dataset/train.csv")
test_df = pd.read_csv("dataset/test.csv")

train_df["SalePrice"]
train_df.info()
test_df.info()

df_ = pd.concat([train_df, test_df])
df = df_.copy()
df.info()


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



def outlier_thresholds(dataframe, col_name, q1 = 0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquentile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquentile_range
    low_limit = quartile1 - 1.5 * interquentile_range
    return low_limit, up_limit

df.loc[(df["Fireplaces"]  > 1), "Fireplaces"].sort_values()
def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(df,col)


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns, n_miss

missing_values_table(df)

Isnull = df.isnull().sum() / len(df) * 100
Isnull = Isnull[Isnull > 0]
Isnull.sort_values(inplace = True, ascending = False)
Isnull
Isnull = Isnull.to_frame()
Isnull.columns = ['count']
Isnull.index.names = ['İsimler']
Isnull['Name'] = Isnull.index

plt.figure(figsize = (15, 9))
sns.set(style = 'whitegrid')
sns.barplot(x = 'Name', y = 'count', data = Isnull)
plt.xticks(rotation = 90)
plt.show(block = True)


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

    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data

df = quick_missing_imp(df, num_method="median", cat_length=17)

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

model_lgb = LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.1, n_estimators=2000,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

model_lgb.fit(X_train,y_train)

lgb_pred = model_lgb.predict(X_test)


print('MAE:', metrics.mean_absolute_error(y_test, lgb_pred))
print('MSE:', metrics.mean_squared_error(y_test, lgb_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, lgb_pred)))

plt.figure(figsize=(15,8))
plt.scatter(y_test,lgb_pred, c='orange')
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
plt.show(block = True)


pd.DataFrame({"act": y_test, "pred": lgb_pred})

dictionary = {"id": y_test.index,"act":y_test*100000, "SalePrice":lgb_pred*100000}
dfSubmission = pd.DataFrame(dictionary)
dfSubmission.to_csv("housePricePredictions2.csv", index=False)

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


plot_importance(model_lgb, X, 40)


