import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from eda_functions import *

method_outlier_list = ['iqr', 'zscore']
method_num_fill_list = ['dropna', 'mean', 'median', 'most_frequent', 'knn']
method_obj_fill_list = ['dropna', 'most_frequent', 'const']
method_encode_list = ['label', 'onehot']
method_scale_list = ['standardization', 'normalization']

st.set_page_config(layout="wide")

col1 = st.columns(1)[0]
col2, col3, col4 = st.columns([0.33, 0.33, 0.33])
col5, col6 = st.columns([0.5, 0.5])
col7 = st.columns([1])[0]
col5, col6 = st.columns([0.5, 0.5])
col7 = st.columns([1])[0]
col8, col9 = st.columns([0.5, 0.5])
col23, col24 = st.columns([0.5, 0.5])
col10 = st.columns([1])[0]
col11, col12 = st.columns([0.5, 0.5])
col13 = st.columns([1])[0]
col14, col15 = st.columns([0.5, 0.5])
col16 = st.columns([1])[0]
col17, col18 = st.columns([0.5, 0.5])
col19 = st.columns([1])[0]
col20, col21 = st.columns([0.5, 0.5])
col22 = st.columns([1])[0]

# df = pd.read_csv('titanic.csv')
with col1:
    uploaded_file = st.file_uploader('Загрузить файл')
    if uploaded_file:
        df = pd.read_csv(uploaded_file, encoding = 'cp1251', encoding_errors='replace')
    else: 
        st.error('Upload file')
        st.stop()
df = df.set_index('ClientId')
df = df.drop(columns='Target').join(df[['Target']])

numeric_columns = df.select_dtypes(include=[np.number]).columns
null_columns = df.columns[(df.isna().any())]
numeric_null_columns = list(set(numeric_columns) & set(null_columns))
object_columns = df.select_dtypes(include=[object]).columns
object_null_columns = list(set(object_columns) & set(null_columns))

with col1:
    st.header("Исходная таблица")
    st.dataframe(df)
    st.header("Краткий EDA")

integrity_report = check_data_integrity(df)
# for key, value in integrity_report.items():
#     st.subheader(key)
#     st.dataframe(value)
with col2:
    st.subheader('Число пропущенных значений')
    st.dataframe(integrity_report['missing_values'])
with col3:
    st.subheader('Число уникальных значений')
    st.dataframe(integrity_report['unique_values'])
with col4:
    st.subheader('Типы данных')
    st.dataframe(integrity_report['data_types'])
with col5:
    st.subheader('Статистики числовых столбцов')
    st.dataframe(integrity_report['numeric_ranges'])
with col6:
    st.subheader('Статистики текстовых столбцов')
    st.dataframe(integrity_report['object_ranges'])
    
with col7:
    st.header("Поиск выбросов")
    col_outlier = st.selectbox('Колонка для поиска', options=numeric_columns, key=1, index=2)
    method_outlier = st.selectbox('Метод поиска', options=method_outlier_list, key=2)
    df_cleaned, flag_change = handle_outliers(df, [col_outlier], method=method_outlier)

with col8:
    st.text("Статистики исходные:")
    st.dataframe(df[col_outlier].describe().round(2))
with col9:
    if flag_change == 1:
        st.text(f"\nСтатистики после очистки выбросов (метод {method_outlier}):")
        st.dataframe(df_cleaned[col_outlier].describe().round(2))
    else:
        st.text(f"\nВыбросы не найдены (метод {method_outlier})")

with col23:
    st.text('histplot')
    plt.figure(figsize=(8,6))
    # plt.hist(df[[col_outlier]])
    plt.hist(x=col_outlier, data=df)
    plt.xlabel(col_outlier)
    st.pyplot(plt)

with col24:
    st.text('boxplot')
    plt.figure(figsize=(8,6))
    sns.boxplot(x=col_outlier, data=df)
    st.pyplot(plt)

with col10:
    if flag_change:        
        outlier_delete = st.checkbox("Заменить выбросные значения", key=9)
        if method_outlier == 'iqr':
            st.info("Для метода iqr замена на (Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)")
        elif method_outlier == 'zscore':
            st.info("Для метода zscore замена на median")
        if outlier_delete:
            df = df_cleaned[[col_outlier]].join(df.drop(columns=col_outlier))

    st.header("Заполнение пропусков")
    st.subheader("Числовые столбцы")
    numeric_col_fill = st.selectbox('Колонка для заполнения', options=numeric_null_columns, key=3)
    method_num_fill = st.selectbox('Метод заполнения', options=method_num_fill_list, key=4)
df_imp_num = handle_missing_values_numeric(df[[numeric_col_fill]], method = method_num_fill)

with col11:
    st.text("Статистики исходные:")
    st.dataframe(df[[numeric_col_fill]].describe().round(2))
    st.text(f"\nРазмер таблицы: {df[[numeric_col_fill]].shape}")
with col12:
    st.text(f"\nСтатистики после заполнения пропусков (методом {method_num_fill}):")
    st.dataframe(df_imp_num.describe().round(2))
    st.text(f"\nРазмер таблицы: {df_imp_num.shape}")
    # st.text(f"Missing values: {df_imp_num.isnull().sum().sum()}")

df_imp_num = df_imp_num.join(df.drop(columns=numeric_col_fill))
df = df_imp_num.copy()

with col13:
    st.subheader("Текстовые столбцы")
    object_col_fill = st.selectbox('Колонка для заполнения', options=object_null_columns, key=7)
    method_obj_fill = st.selectbox('Метод заполнения', options=method_obj_fill_list)
df_imputed_obj = handle_missing_values_object(df[[object_col_fill]], method=method_obj_fill)

with col14:
    st.text("Статистики исходные:")
    st.dataframe(df[[object_col_fill]].describe().round(2))
    st.text(f"\nРазмер таблицы: {df[[object_col_fill]].shape}")
with col15:
    st.text(f"\nСтатистики после заполнения пропусков (методом {method_obj_fill}):")
    st.dataframe(df_imputed_obj.describe().round(2))
    st.text(f"\nРазмер таблицы: {df_imputed_obj.shape}")
    # st.text(f"Missing values: {df_imputed_obj.isnull().sum().sum()}")

df_imp_obj = df_imputed_obj.join(df.drop(columns=object_col_fill))
df = df_imp_obj.copy()

with col16:
    st.header("Кодирование категориальных столбцов")
    encode_col = st.multiselect('Колонка для кодирования', options=object_columns, key=8, default=['Org','Size'])
    method_encode = st.selectbox('Метод кодирования', options=method_encode_list, key=6)
df_encoded = encode_categorical_variables(df, encode_col, method=method_encode)

with col17:
    st.text("Исходные данные (топ 5):")
    st.dataframe(df[encode_col].head())
with col18:
    if method_encode == 'label':
        st.text("\nДанные (топ 5) после кодирования (метод label):")
        st.dataframe(df_encoded[encode_col].head())
    elif method_encode == 'onehot':
        st.text("\nДанные (топ 5) после кодирования (метод onehot):")
        st.dataframe(df_encoded.filter(regex='^(Org|Size)').head()) # TODO: parametrize
df = df_encoded.copy()

with col19:
    st.header("Шкалирование данных")
    st.info("Только для числовых столбцов")
    method_scale = st.selectbox('Метод шкалирования', options=method_scale_list, key=5)
df_scaled = scale(df.drop(columns='Target'), method_scale)
df_scaled = df_scaled.join(df[['Target']])
with col20:
    st.text("Статистики исходные:")
    st.dataframe(df.describe().round(2))
with col21:
    st.text(f"\nСтатистики после шкалирования (метод {method_scale}):")
    st.dataframe(df_scaled.describe().round(2))

object_columns_left = df.select_dtypes(include=[object]).columns.to_list()
with col22:
    object_delete = st.checkbox(f"Удалить оставшиеся столбцы {object_columns_left}")
if object_delete:
    df_total = df_scaled.drop(columns=object_columns_left)
else:
    df_total = df_scaled.copy()

with col22:
    st.header("Итоговая таблица после всех преобразований")
    st.dataframe(df_total)
    df_total_csv = df_total.reset_index().to_csv(index=False).encode('cp1251')
    st.download_button(label="Выгрузить таблицу", data=df_total_csv, file_name='data_preprocessed.csv')
