import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
import platform
import os
import json
import re
import pickle
from datetime import datetime
from collections import defaultdict
from matplotlib.font_manager import FontProperties
warnings.filterwarnings('ignore')

# 載入資料 (請修改路徑)
print("開始載入資料")
df_train = pd.read_csv("training.csv")
df_test = pd.read_csv("public_x.csv")
df_submission = pd.read_csv("submission_template_public.csv")
print("載入完成")

# 設置中文字體支援 (保留原有函數)
def set_chinese_font():
    """設置中文字體，並返回字體屬性"""
    import matplotlib

    # 檢查系統中的中文字體
    chinese_fonts = []
    for font in plt.matplotlib.font_manager.fontManager.ttflist:
        if any(char in font.name for char in ['黑', '明', '宋', '楷', '微', '華', '文']):
            chinese_fonts.append(font.name)
    
    # 檢查指定路徑的字體
    font_paths = [
        # Windows 字體
        'C:/Windows/Fonts/msjh.ttc',    # 微軟正黑體
        'C:/Windows/Fonts/mingliu.ttc', # 細明體
        'C:/Windows/Fonts/kaiu.ttf',    # 標楷體
        # macOS 字體
        '/System/Library/Fonts/PingFang.ttc',  # 蘋方
        '/Library/Fonts/Arial Unicode.ttf',    # Arial Unicode MS
        # Linux 字體
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',  # 文泉驛微米黑
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',  # Noto Sans CJK
    ]
    
    # 找到第一個存在的字體路徑
    font_path = None
    for path in font_paths:
        if os.path.exists(path):
            font_path = path
            break
    
    # 設置字體
    if font_path:
        font_prop = FontProperties(fname=font_path)
    elif chinese_fonts:
        font_prop = FontProperties(family=chinese_fonts[0])
    else:
        # 使用預設字體
        font_prop = FontProperties()
    
    # 設置 matplotlib 全局字體
    matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'WenQuanYi Zen Hei'] 
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
    
    return font_prop

# 呼叫函數設置字體
font_prop = set_chinese_font()

# 創建保存模型的資料夾
models_dir = "saved_models"
os.makedirs(models_dir, exist_ok=True)

def main():
    # 提取標籤和特徵
    y = df_train['飆股'].values  
    X = df_train.drop(['ID', '飆股'], axis=1, errors='ignore')
    
    # 準備測試資料
    test_ids = df_test['ID'].values
    X_test = df_test.drop(['ID'], axis=1, errors='ignore')
    
    # 確保訓練集和測試集有相同的特徵
    common_features = list(set(X.columns) & set(X_test.columns))
    X = X[common_features]
    X_test = X_test[common_features]
    
    print(f"訓練資料特徵數量: {X.shape[1]}")
    print(f"飆股比例: {100 * np.mean(y):.2f}%")
    
    # 設定交叉驗證參數
    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # 計算正樣本權重
    pos_samples = np.sum(y)
    neg_samples = len(y) - pos_samples
    scale_pos_weight = neg_samples / pos_samples if pos_samples > 0 else 1.0
    print(f"正負樣本權重比例 scale_pos_weight: {scale_pos_weight:.2f}")
    
    # 設定LightGBM參數
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42
    }
    
    # 儲存每個fold的資訊
    models = []
    model_paths = []
    thresholds = []
    feature_importance_folds = defaultdict(list)
    
    print(f"\n===== 開始{n_folds}折交叉驗證 =====")
    
    # 測試集預測結果
    test_predictions = np.zeros(len(X_test))
    
    # 交叉驗證流程
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n----- Fold {fold+1}/{n_folds} -----")
        
        # 分割資料
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # 確認標籤分佈
        print(f"Fold {fold+1} 訓練集飆股比例: {100*np.mean(y_tr):.2f}%, "
              f"驗證集飆股比例: {100*np.mean(y_val):.2f}%")
        
        # 建立資料集
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # 訓練模型
        evaluation_results = {}
        lgb_model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'validation'],
            callbacks=[
                lgb.early_stopping(50), 
                lgb.log_evaluation(100),
                lgb.record_evaluation(evaluation_results)
            ],  
        )
        
        # 儲存模型
        models.append(lgb_model)
        model_path = os.path.join(models_dir, f"model_fold_{fold+1}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(lgb_model, f)
        model_paths.append(model_path)
        
        # 評估結果
        print(f"最佳迭代次數: {lgb_model.best_iteration}")
        
        # 驗證集預測
        y_val_prob = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
        
        # 找尋最佳閾值
        best_f1 = 0
        best_threshold = 0.5
        for thresh in np.arange(0.1, 1.0, 0.05):
            y_val_pred = (y_val_prob >= thresh).astype(int)
            f1 = f1_score(y_val, y_val_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        
        thresholds.append(best_threshold)
        print(f"Fold {fold+1} 最佳閾值: {best_threshold:.3f}, 最佳F1: {best_f1:.4f}")
        
        # 特徵重要性分析
        feature_importance = lgb_model.feature_importance(importance_type='gain')
        feature_names = X.columns
        
        # 儲存每個fold的特徵重要性
        for i, name in enumerate(feature_names):
            feature_importance_folds[name].append(feature_importance[i])
        
        # 計算驗證集的效能
        y_val_pred = (y_val_prob >= best_threshold).astype(int)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        
        print(f"驗證集 - 準確率: {val_acc:.4f}, F1分數: {val_f1:.4f}")
        
        # 使用當前模型進行測試集預測併累加
        fold_test_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
        test_predictions += fold_test_pred
        
        # 繪製學習曲線 (最後一個fold)
        if fold == n_folds - 1:
            plt.figure(figsize=(10, 6))
            plt.plot(evaluation_results['train']['binary_logloss'], label='訓練損失')
            plt.plot(evaluation_results['validation']['binary_logloss'], label='驗證損失')
            plt.axvline(x=lgb_model.best_iteration, color='r', linestyle='--', 
                      label=f'最佳迭代次數: {lgb_model.best_iteration}')
            
            plt.xlabel('迭代次數', fontproperties=font_prop)
            plt.ylabel('二元對數損失', fontproperties=font_prop)
            plt.title('訓練及驗證損失曲線', fontproperties=font_prop)
            plt.legend(prop=font_prop)
            
            plt.grid(True)
            plt.savefig('learning_curves.png')
            print("已保存學習曲線圖到 learning_curves.png")
    
    # ===== 交叉驗證結果分析 =====
    print("\n===== 交叉驗證結果分析 =====")
    
    # 閾值分析
    mean_threshold = np.mean(thresholds)
    std_threshold = np.std(thresholds)
    print(f"閾值平均值: {mean_threshold:.3f} ± {std_threshold:.3f}")
    
    # 特徵重要性穩定性分析
    feature_importance_df = pd.DataFrame(feature_importance_folds)
    
    # 計算每個特徵的平均重要性
    importance_mean = feature_importance_df.mean()
    
    # 特徵重要性總結
    feature_summary = pd.DataFrame({
        'feature': importance_mean.index,
        'mean_importance': importance_mean.values
    })
    
    # 按平均重要性排序
    top_features = feature_summary.sort_values('mean_importance', ascending=False).head(20)
    
    # 繪製前20個特徵的重要性
    plt.figure(figsize=(12, 10))
    
    # 繪製平均特徵重要性
    ax = plt.subplot(1, 1, 1)
    
    # 獲取原始數據
    plot_data = top_features.copy()
    
    # 繪製條形圖
    bars = sns.barplot(x='mean_importance', y='feature', data=plot_data, ax=ax)
    
    # 修改y軸標籤字體
    plt.yticks(fontproperties=font_prop)
    
    plt.title('前20個重要特徵分析', fontproperties=font_prop, fontsize=14)
    plt.xlabel('平均重要性', fontproperties=font_prop, fontsize=12)
    plt.ylabel('特徵', fontproperties=font_prop, fontsize=12)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    print("已保存特徵重要性圖到 feature_importance.png")
    
    # ===== 對測試集進行最終預測 =====
    print("\n===== 測試集預測 =====")
    
    # 取平均
    test_predictions /= n_folds
    
    # 使用平均閾值進行分類
    y_test_pred = (test_predictions >= mean_threshold).astype(int)
    
    # 保存預測結果
    result_df = pd.DataFrame({
        'ID': test_ids,
        '飆股': y_test_pred
    })
    
    # 將預測結果合併到提交範例中
    # 確保ID的排序與提交範例一致
    df_submission['飆股'] = result_df.set_index('ID').loc[df_submission['ID']]['飆股'].values
    
    # 保存結果
    submission_path = "submission.csv"
    df_submission.to_csv(submission_path, index=False)
    print(f"預測結果已保存至 {submission_path}")
    
    # 保存預測概率
    proba_path = "prediction_probabilities.csv"
    proba_df = pd.DataFrame({
        'ID': test_ids,
        'probability': test_predictions
    })
    proba_df.to_csv(proba_path, index=False)
    print(f"預測概率已保存至 {proba_path}")
    
    # 保存模型路徑和閾值
    model_info = {
        'model_paths': model_paths,
        'mean_threshold': float(mean_threshold),
        'thresholds': [float(t) for t in thresholds],
        'feature_importance': top_features.to_dict('records')
    }
    
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    
    print("模型資訊已保存至 model_info.json")
    
    return {
        'models': models,
        'model_paths': model_paths,
        'mean_threshold': mean_threshold,
        'feature_importance': top_features,
        'submission_path': submission_path
    }

if __name__ == "__main__":
    main()
