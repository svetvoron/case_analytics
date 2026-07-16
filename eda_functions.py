import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def check_data_integrity(df):
    # Проверка на отсутствующие значения
    missing_values = df.isnull().sum()
    
    # Число уникальных значений по столбцам
    unique_values = df.nunique()
    
    # Проверка типов данных
    data_types = df.dtypes
    
    # Проверка уникальности строк
    # duplicated_rows = df.duplicated().sum()
    
    # Проверка диапазона значений для числовых полей
    numeric_ranges = df.describe()
    
    # Проверка диапазона значений для категориальных полей
    object_ranges = df.describe(include=object)
    
    return {
        'missing_values': missing_values,
        'unique_values': unique_values,
        'data_types': data_types,
        # 'duplicated_rows': duplicated_rows,
        'numeric_ranges': numeric_ranges,
        'object_ranges': object_ranges
    }


def handle_outliers(df, columns, method):
    df_clean = df.copy()
    
    for column in columns:
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[column] = df_clean[column].clip(lower_bound, upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[column]))
            df_clean[column] = df_clean[column].mask(z_scores > 3, df_clean[column].median())
    if df_clean.equals(df):
        flag = 0
    else: flag = 1
    return df_clean, flag


def handle_missing_values_numeric(df, method):
    # Удаление строк с пропущенными значениями
    if method == 'dropna':
        df_imputed = df.dropna()
    
    # Заполнение пропусков средним значением
    elif method == 'mean':
        imputer = SimpleImputer(strategy='mean')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index.to_list())
    
    # Заполнение пропусков медианой
    elif method == 'median':
        imputer = SimpleImputer(strategy='median')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index.to_list())
    
    # Заполнение пропусков самым частым значением
    elif method == 'most_frequent':
        imputer = SimpleImputer(strategy='most_frequent')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index.to_list())
    
    # Заполнение пропусков методом k-ближайших соседей
    elif method == 'knn':
        imputer = KNNImputer(n_neighbors=5)
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index.to_list())
    
    return df_imputed


def handle_missing_values_object(df, method):
    # Удаление строк с пропущенными значениями
    if method == 'dropna':
        df_imputed = df.dropna()
    
    # Заполнение пропусков самым частым значением
    elif method == 'most_frequent':
        imputer = SimpleImputer(strategy='most_frequent')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index.to_list())
    
    # Заполнение пропусков const
    elif method == 'const':
        imputer = SimpleImputer(strategy='constant', fill_value='NA')
        df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index.to_list())
    
    return df_imputed


def encode_categorical_variables(df, columns, method):
    df_encoded = df.copy()
    if method == 'label':
        le = LabelEncoder()
        for col in columns:
            df_encoded[col] = le.fit_transform(df[col])

    elif method == 'onehot':
        ohe = OneHotEncoder(sparse_output=False)
        encoded_cols = ohe.fit_transform(df[columns])
        encoded_col_names = ohe.get_feature_names_out(columns)
        encoded_df = pd.DataFrame(encoded_cols, columns=encoded_col_names, index=df.index)
        df_encoded = pd.concat([df_encoded.drop(columns, axis=1), encoded_df], axis=1)

    return df_encoded


def scale(df, method):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    # Нормализация (масштабирование в диапазон [0, 1])
    if method == 'normalization':
        scaler = MinMaxScaler()
        df_scaled = df.copy()
        df_scaled[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    # Стандартизация (приведение к нулевому среднему и единичной дисперсии)
    elif method == 'standardization':
        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    return df_scaled
