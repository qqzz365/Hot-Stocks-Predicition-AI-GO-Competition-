import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import skew
from sklearn.preprocessing import RobustScaler
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
def load_data(path):
    # read the data
    print('load data...')
    df = pd.read_csv(path)
    # df = df.drop(columns='ID')
    print(f'初始數據筆數:{len(df)}')
    print(f'初始數features數:{len(df.columns)}')
    return df


def fill_na_median(df, fill_value=None, testset=False):
    if not testset:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        zero_values = {col: 0 for col in numeric_cols}
    
        df_filled = df.copy()
        for col in numeric_cols:
            df_filled[col] = df_filled[col].fillna(0)
        
        return df_filled, zero_values
    else:
        df_filled = df.copy()
        numeric_cols = [col for col in fill_value.keys() if col in df.columns]
        for col in numeric_cols:
            df_filled[col] = df_filled[col].fillna(fill_value[col])
        
        return df_filled


def one_hot_encode(df):
    categorical_features = ['季IFRS財報_DPZ等級', '季IFRS財報_Z等級']
    
    for feature in categorical_features:
        if feature in df.columns:
            # 将NA值填充为"Missing"
            df[feature].fillna("Missing", inplace=True)
            
            # 将特征转换为category类型
            df[feature] = df[feature].astype('category')
            
            # 输出类别信息
            print(f"{feature} 的类别: {df[feature].cat.categories.tolist()}")
            print(f"{feature} 共有 {len(df[feature].cat.categories)} 个类别")
            
            # 执行one-hot编码
            dummies = pd.get_dummies(df[feature], prefix=feature)
            
            # 删除原始列，将one-hot编码结果合并到DataFrame中
            df = pd.concat([df.drop(feature, axis=1), dummies], axis=1)
    
    return df


def drop_na(df):
    print('drop columns with too much na...')
    threshold = 40  # 如果na比例超過20%就全丟
    to_drop_col = [col for col in df.columns if df[col].isna().sum()/len(df)*100 > threshold] 
    print(f'dropped columns: {to_drop_col}')  # 修正：添加f-string
    df = df.drop(columns=to_drop_col)
    print(f'number of drop columns:{len(to_drop_col)}')
    print(f'ratio of drop columns:{len(to_drop_col)/len(df.columns):.2f}')
    return df, to_drop_col



def normalization(df, scaler=None, testset=False):
    print('使用穩健標準化方法...')
    # 找出需要标准化的数值列，排除标签列"飆股"和类别特征
    num_col = [col for col in df.columns if not (col.startswith('季IFRS財報_DPZ等級') or 
                                              col.startswith('季IFRS財報_Z等級') or 
                                              col == '飆股' or
                                              col == '_is_train')]  # 確保不標準化標記列
    print('len of num columns:', len(num_col))
    
    if testset == False:
        # 训练集模式
        scaler = RobustScaler(quantile_range=(25.0, 75.0))  # 使用IQR (四分位距)
        normalized_values = scaler.fit_transform(df[num_col])
        # 更新原始数据框中的值
        df_copy = df.copy()
        df_copy[num_col] = normalized_values
        return df_copy, scaler
    else:
        # 测试集模式
        normalized_values = scaler.transform(df[num_col])
        # 更新原始数据框中的值
        df_copy = df.copy()
        df_copy[num_col] = normalized_values
        return df_copy
    


def concate_data(X_train, y_train, X_test, y_test):
    # concate
    print('concate data...')
    trainingset = pd.concat([X_train, y_train], axis=1)
    validationset = pd.concat([X_test, y_test], axis=1)
    return trainingset, validationset


def feature_transform(df):
    """
    對數值特徵進行轉換以減少偏態，針對偏態較大的特徵先移動到正域再進行對數轉換
    此函數處理已合併的訓練集和測試集數據
    """
    print('對合併後的完整數據集進行特徵轉換...')
    
    # 僅處理數值型欄位，排除目標變數和數據集標識符
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if col != '飆股' and col != '_is_train']
    
    transformed_cols = []
    
    for col in numeric_cols:
        # 排除全NaN的列
        if df[col].notna().any():
            data = df[col].dropna()
            skew_value = skew(data)
            
            # 對於高偏態(>3.0)的特徵進行對數轉換
            if abs(skew_value) > 3.0:
                # 保存非缺失值的索引
                non_na_idx = df[col].notna()
                
                # 計算需要的平移量，確保所有值都是正的
                min_value = data.min()
                shift_value = 0 if min_value > 0 else abs(min_value) + 1
                
                # 應用平移和對數轉換
                df.loc[non_na_idx, col] = np.log1p(df.loc[non_na_idx, col] + shift_value)
                
                transformed_cols.append(col)
                print(f"對 {col} 應用平移對數轉換 (偏態: {skew_value:.2f}, 平移量: {shift_value})")
                
                # 記錄轉換後的偏態變化
                new_skew = skew(df.loc[non_na_idx, col].dropna())
                print(f"  轉換後偏態: {new_skew:.2f}")
    
    print(f"共轉換了 {len(transformed_cols)} 個高偏態特徵")
    return df

def drop_9000(df):
    df_front = df.loc[:,'外資券商_分點進出':'季IFRS財報_財務信評']
    df_back = df.loc[:,'個股收盤價':]
    df = pd.concat([df_front,df_back], axis = 1)
    return df

def drop_col(df, droped_col):
    df = df.drop(columns=droped_col)
    print(f'number of drop columns:{len(droped_col)}')
    print(f'ratio of drop columns:{len(droped_col)/len(df.columns):.2f}')
    return df



df = pd.read_csv('df_1000.csv')  #　放原始資料
df_process = drop_9000(df)
df_process, droped_col = drop_na(df_process)
df_process = one_hot_encode(df_process)
df_process = feature_transform(df_process)
df_process, scaler = normalization(df_process)
df_process, fill_value = fill_na_median(df_process)
df_process.loc[:, '季IFRS財報_DPZ等級_-0.3845':'季IFRS財報_Z等級_Missing'] = \
    df_process.loc[:, '季IFRS財報_DPZ等級_-0.3845':'季IFRS財報_Z等級_Missing'].astype(int)
col = df_process.pop('飆股')
df_process['飆股'] = col
df_process.to_csv('df_process.csv', index=False)
print('final features len:',len(df_process.columns))

df = pd.read_csv('Public_Test/public_x_100.csv')  #　放原始資料
df_process = drop_9000(df)
df_process = drop_col(df_process, droped_col)
df_process = one_hot_encode(df_process)
df_process = feature_transform(df_process)
df_process, scaler = normalization(df_process)
df_process, fill_value = fill_na_median(df_process)
df_process.loc[:, '季IFRS財報_DPZ等級_-0.3845':'季IFRS財報_Z等級_Missing'] = \
    df_process.loc[:, '季IFRS財報_DPZ等級_-0.3845':'季IFRS財報_Z等級_Missing'].astype(int)
df_process.to_csv('Public_Test/public_x_process_1.csv', index=False)
print('final features len:',len(df_process.columns))