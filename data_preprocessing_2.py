import pandas as pd
import numpy as np

# 計算指標的平均值
def calculate_d0tod20group_average(input_df, output_df, var_count, group_list):
    col_names = []
    for i in range(1, 4):
        for col in group_list[0]:
            col_names.append(f'{col}_group{i}_avg')
    for group_idx, group in enumerate(group_list):
        if group_idx > 0:
            for var_idx in range(var_count):
                output_df[(group_idx-1)*var_count+var_idx] = input_df.iloc[:,(7*var_count)*(group_idx-1)+var_idx:(7*var_count)*group_idx:var_count].mean(axis = 1)
    output_df.columns = col_names
    return output_df

# 計算指標的差異 (6天 - 當天)
def calculate_d0tod20group_difference(input_df, output_df, var_count, group_list):
    epsilon = 1e-8
    col_names = []
    for i in range(1, 4):
        for col in group_list[0]:
            col_names.append(f'{col}_group{i}_diff')
    for group_idx, group in enumerate(group_list):
        if group_idx > 0:
            for var_idx in range(var_count):
                output_df[(group_idx-1)*var_count+var_idx] = np.log(input_df.iloc[:, (7*var_count)*(group_idx-1)+var_idx+6*var_count]+epsilon) - np.log(input_df.iloc[:, (7*var_count)*(group_idx-1)+var_idx]+epsilon)
    output_df.columns = col_names
    return output_df

# 計算obv
def calculate_obv(input_df, output_df, d0):
    fea = d0[0][:-3]
    time_cols = input_df.columns
    price_cols = [col for col in time_cols if '收盤價' in col]
    vol_cols = [col for col in time_cols if '成交量' in col]
    price_df = input_df[price_cols]
    vol_df = input_df[vol_cols]
    def compute_obv(prev_obv, prev_price, curr_price, curr_vol):
        if prev_price >= curr_price:
            curr_obv = prev_obv + curr_vol
        else:
            curr_obv = prev_obv - curr_vol
        return curr_obv
    for i in range(0, 21):
        output_df[f'{fea}obv{i}'] = pd.Series([0] * len(input_df))
    for i in range(len(price_df)):
        for j in range(20, 0, -1):
            output_df.iloc[i, j-1] = compute_obv(output_df.iloc[i, j], price_df.iloc[i, j], price_df.iloc[i, j-1], vol_df.iloc[i, j-1])
    
    return output_df

def calculate_pmv(input_df, output_df, d0):
    fea = d0[0][:-3]
    time_cols = input_df.columns
    price_cols = [col for col in time_cols if '收盤價' in col]
    vol_cols = [col for col in time_cols if '成交量' in col]
    price_df = input_df[price_cols]
    vol_df = input_df[vol_cols]
    def compute_pmv(prev_price, curr_price, curr_vol):
        return (curr_price - prev_price) * curr_vol
    for i in range(0, 21):
        output_df[f'{fea}pmv{i}'] = pd.Series([0] * len(input_df))
    for i in range(len(price_df)):
        for j in range(20, 0, -1):
            output_df.iloc[i, j-1] = compute_pmv(price_df.iloc[i, j], price_df.iloc[i, j-1], vol_df.iloc[i, j-1])
    return output_df

# 個股
def STK_pipeline(X_data, original=True, avg=True, diff=True, obv=True, pmv = True):
    # 提取個股相關的欄位
    time_cols = [
        col for col in X_data.columns 
        if col.startswith('個股') 
        and not any(col.startswith(prefix) for prefix in ['個股券商分點籌碼分析', '個股券商分點區域分析', '個股主力'])
        and not ('報酬率' in col or '波動度' in col or '乖離率' in col)
    ]
    non_time_cols = [
        col for col in X_data.columns 
        if col.startswith('個股') 
        and not any(col.startswith(prefix) for prefix in ['個股券商分點籌碼分析', '個股券商分點區域分析', '個股主力'])
        and ('報酬率' in col or '波動度' in col or '乖離率' in col)
    ]
    all_cols = time_cols + non_time_cols
    d0_col = ['個股收盤價', '個股成交量']

    # 每個組別的列：根據不同天數分組
    group_1_col = d0_col + [col for col in time_cols if any(f'前{day}天' in col for day in range(1, 7))]
    group_2_col = [col for col in time_cols if any(f'前{day}天' in col for day in range(7, 14))]
    group_3_col = [col for col in time_cols if any(f'前{day}天' in col for day in range(14, 21))]

    # 初始化Dataframe
    STK_df = X_data[time_cols]
    NTSTK_df = X_data[non_time_cols]
    avg_df = pd.DataFrame()
    diff_df = pd.DataFrame()
    obv_df = pd.DataFrame()
    pmv_df = pd.DataFrame()

    # 計算平均值 差值 obv pmv
    if avg:
        calculate_d0tod20group_average(STK_df, avg_df, 2, [d0_col, group_1_col, group_2_col, group_3_col])
    if diff:
        calculate_d0tod20group_difference(STK_df, diff_df, 2, [d0_col, group_1_col, group_2_col, group_3_col])
    if obv:
        calculate_obv(STK_df, obv_df, d0_col)
    if pmv:
        calculate_pmv(STK_df, pmv_df, d0_col)

    final_df = pd.DataFrame()
  
    if avg or diff or obv or pmv:
        final_df = pd.concat([final_df, NTSTK_df], axis=1)
        if avg:
            final_df = pd.concat([final_df, avg_df], axis=1)
        if diff:
            final_df = pd.concat([final_df, diff_df], axis=1)
        if obv:
            final_df = pd.concat([final_df, obv_df], axis=1)
        if pmv:
            final_df = pd.concat([final_df, pmv_df], axis=1)
        if original:
            final_df = pd.concat([final_df, STK_df], axis=1)
    else:
        return X_data[all_cols]

    return final_df

# 上市加權指數
def TAIEX_pipeline(X_data, original=True, avg=True, diff=True, obv=True, pmv = True):
    # 提取所有上市加權指數相關的欄位
    time_cols = [
        col for col in X_data.columns if col.startswith('上市加權指數')
        and not ('報酬率' in col or '波動度' in col or '乖離率' in col)
    ]
    non_time_cols = [
        col for col in X_data.columns if col.startswith('上市加權指數')
        and ('報酬率' in col or '波動度' in col or '乖離率' in col)
    ]
    all_cols = time_cols + non_time_cols
    d0_col = ['上市加權指數收盤價', '上市加權指數成交量']
    
    # 每個組別的列：根據不同天數分組
    group_1_col = d0_col + [col for col in time_cols if any(f'前{day}天' in col for day in range(1, 7))]
    group_2_col = [col for col in time_cols if any(f'前{day}天' in col for day in range(7, 14))]
    group_3_col = [col for col in time_cols if any(f'前{day}天' in col for day in range(14, 21))]

    # 初始化Dataframe
    TAIEX_df = X_data[time_cols]
    NTTAIEX_df = X_data[non_time_cols]
    avg_df = pd.DataFrame()
    diff_df = pd.DataFrame()
    obv_df = pd.DataFrame()
    pmv_df = pd.DataFrame()

    # 計算平均值 差值 obv
    if avg:
        calculate_d0tod20group_average(TAIEX_df, avg_df, 2, [d0_col, group_1_col, group_2_col, group_3_col])
    if diff:
        calculate_d0tod20group_difference(TAIEX_df, diff_df, 2, [d0_col, group_1_col, group_2_col, group_3_col])
    if obv:
        calculate_obv(TAIEX_df, obv_df, d0_col)
    if pmv:
        calculate_pmv(TAIEX_df, pmv_df, d0_col)

    final_df = pd.DataFrame()
  
    if avg or diff or obv:
        final_df = pd.concat([final_df, NTTAIEX_df], axis=1)
        if avg:
            final_df = pd.concat([final_df, avg_df], axis=1)
        if diff:
            final_df = pd.concat([final_df, diff_df], axis=1)
        if obv:
            final_df = pd.concat([final_df, obv_df], axis=1)
        if pmv:
            final_df = pd.concat([final_df, pmv_df], axis=1)
        if original:
            final_df = pd.concat([final_df, TAIEX_df], axis=1)
    else:
        return X_data[all_cols]

    return final_df

def mypipeline(X_data, y_data=None):
    # 若存在 'ID' 欄位，將其刪除
    if 'ID' in X_data.columns:
        X_data = X_data.drop(columns=['ID'])
    
    # 若傳入了 y_data，則使用 y_data，否則從 X_data 中提取 '飆股' 欄位
    if y_data is not None:
        y = y_data
    else:
        if '飆股' in X_data.columns:
            y = X_data['飆股']
            X_data = X_data.drop(columns=['飆股'])  # 從 X_data 中刪除 '飆股' 欄位
        else:
            y = None  # 若無 '飆股' 欄位，則 y 設為 None

    taiex_df = TAIEX_pipeline(X_data, original=False, avg=False, diff=False, obv=True, pmv=True)
    STK_df = STK_pipeline(X_data, original=False, avg=False, diff=False, obv=True, pmv=True)
    X_data = pd.concat([X_data, taiex_df, STK_df], axis = 1)

    return X_data, y_data

# ==================使用pipeline==================
df = pd.read_csv('df_process.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
preprocessed_X, preprocessed_y = mypipeline(X, y)

df_process_2 = pd.concat([preprocessed_X, preprocessed_y], axis=1)

df_process_2.to_csv('df_process_2.csv', index=False)


df = pd.read_csv('Public_Test/public_x_process_1.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
preprocessed_X, preprocessed_y = mypipeline(X, y)

df_process_2 = pd.concat([preprocessed_X, preprocessed_y], axis=1)

df_process_2.to_csv('Public_Test/public_x_process_2.csv', index=False)