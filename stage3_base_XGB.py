import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
import optuna
import xgboost as xgb
import os
from typing import Any, Dict, Tuple, List, Optional
from joblib import Parallel, delayed
import gc

# 創建結果保存目錄
os.makedirs('result_XGB', exist_ok=True)

def xgb_prauc(predt, dtrain):
    """自定義 AUPRC 評估函數，用於 XGBoost 的早停機制"""
    y_true = dtrain.get_label()
    auprc = average_precision_score(y_true, predt)
    return 'prauc', -auprc

def load_data(data_path: str, is_test: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """載入資料並分離特徵與目標變數，確保資料為 float32 格式以節省記憶體"""
    if not os.path.exists(data_path.replace('.csv', '.parquet')):
        df = pd.read_csv(data_path)
        df = df.astype({col: 'float32' for col in df.select_dtypes('float64').columns})
        df.to_parquet(data_path.replace('.csv', '.parquet'))
    df = pd.read_parquet(data_path.replace('.csv', '.parquet'))
    if is_test:
        X = df
        y = None
    else:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    return X, y

def split_features_by_categories(X: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """依據特徵類別定義分割特徵"""
    import json
    with open('feature_categories.json', 'r', encoding='utf-8') as f:
        feature_categories = json.load(f)
    
    feature_dfs = {}
    for category_name, category_info in feature_categories.items():
        start_col = category_info['start']
        end_col = category_info['end']
        start_idx = list(X.columns).index(start_col) if start_col in X.columns else None
        end_idx = list(X.columns).index(end_col) if end_col in X.columns else None
        if start_idx is not None and end_idx is not None:
            category_cols = list(X.columns)[start_idx:end_idx+1]
            feature_dfs[category_name] = X[category_cols]
            print(f"已擷取 {category_name} 特徵: {len(category_cols)} 個")
        else:
            print(f"警告: 無法找到 {category_name} 的起始或結束欄位")
    return feature_dfs

def get_feature_combinations() -> List[Tuple[str, List[str]]]:
    """取得特徵組合列表，產生所有可能的特徵組合（共7種）"""
    # 定義我們的三個特徵類別及其英文簡寫
    feature_categories = ["上市加權", "技術指標", "其他"]
    feature_abbr = ["WI", "TI", "OT"]  # Weighted Index, Technical Indicators, Others
    feature_map = dict(zip(feature_categories, feature_abbr))
    
    combinations = []
    
    # 使用二進制位元組合生成所有可能組合（除了空集合）
    for i in range(1, 2**len(feature_categories)):
        selected_categories = []
        combo_name_parts = []
        
        # 檢查每個位元是否為1，決定是否將該特徵類別加入組合
        for j in range(len(feature_categories)):
            if (i >> j) & 1:
                selected_categories.append(feature_categories[j])
                combo_name_parts.append(feature_map[feature_categories[j]])
        
        # 創建組合名稱和對應的特徵列表
        combo_name = "model_" + "_".join(combo_name_parts)
        combinations.append((combo_name, selected_categories))
    
    print(f"生成了 {len(combinations)} 種特徵組合")
    for combo_name, selected_cats in combinations:
        cat_names = [f'"{cat}"' for cat in selected_cats]
        print(f"  - {combo_name}: {', '.join(cat_names)}")
    
    return combinations

def train_fold(params, fold, train_idx, val_idx, X, y, model_name):
    """訓練單一折模型"""
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        evals=[(dval, 'val')],
        custom_metric=xgb_prauc,
        num_boost_round=2000,
        early_stopping_rounds=50,
        evals_result=evals_result,
        verbose_eval=100
    )
    y_pred = model.predict(dval, iteration_range=(0, model.best_iteration))
    auprc = average_precision_score(y_val, y_pred)
    return fold, model, y_pred, val_idx, model.best_iteration, auprc, evals_result

def xgb_objective(trial, X, y, cv=5, model_name="model"):
    """Optuna 超參數搜索的目標函數，針對 XGBoost 模型"""
    pos_ratio = np.mean(y)
    print(f"[{model_name}] 試驗 {trial.number} - 正樣本比例: {pos_ratio:.6f}")
    
    params = {
        # 基本目標
        'objective': 'binary:logistic',  # 二分類問題
        'eval_metric': 'auc',  # 使用AUC評估，適合不平衡數據
        'verbosity': 0,
        
        # 主要超參數
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),  # 控制較小以防過擬合
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),  # 較大範圍以應對不平衡
        
        # 處理不平衡數據的特定參數
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 50, 100),  # 約為負/正樣本比例
        
        # 正則化參數
        'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
        'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 0, 10),
    }
    
    cv_scores = []
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(X))
    fold_models = []
    best_iterations = []
    
    print(f"[{model_name}] 試驗 {trial.number} 開始交叉驗證（{cv}折）...")
    # 保留折層級並行
    results = Parallel(n_jobs=cv)(
        delayed(train_fold)(params, fold, train_idx, val_idx, X, y, model_name)
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y))
    )
    
    for fold, model, y_pred, val_idx, best_iter, auprc, evals_result in sorted(results, key=lambda x: x[0]):
        cv_scores.append(auprc)
        oof_predictions[val_idx] = y_pred
        fold_models.append(model)
        best_iterations.append(best_iter)
        print(f"[{model_name}] 試驗 {trial.number} - Fold {fold+1} - AUC: {evals_result['val']['auc'][-1]:.4f}, AUPRC: {auprc:.4f}")
        trial.report(auprc, fold)
        if trial.should_prune():
            print(f"[{model_name}] 試驗 {trial.number} 被剪枝")
            raise optuna.exceptions.TrialPruned()
    
    trial.set_user_attr("fold_models", fold_models)
    trial.set_user_attr("best_iterations", best_iterations)
    trial.set_user_attr("oof_predictions", oof_predictions)
    trial.set_user_attr("fold_scores", cv_scores)
    trial.set_user_attr("best_params", params)
    
    average_auprc = np.mean(cv_scores)
    print(f"[{model_name}] 試驗 {trial.number} 完成 - 平均 AUPRC: {average_auprc:.4f}")
    return average_auprc

def tune_and_get_oof_predictions_xgb(X: pd.DataFrame, y: pd.Series, model_name: str, num_trials: int = 20, cv: int = 5) -> Tuple[np.ndarray, Dict, List[float], List[Any], List[int]]:
    """使用 Optuna 調整 XGBoost 超參數並返回 out-of-fold 預測結果和訓練好的模型"""
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    print(f"[{model_name}] 開始超參數搜索（{num_trials}次試驗）...")
    
    for trial_number in range(num_trials):
        trial = study.ask()
        trial._trial_id = trial_number
        try:
            value = xgb_objective(trial, X, y, cv, model_name)
            study.tell(trial, value)
        except optuna.exceptions.TrialPruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
    
    best_trial = study.best_trial
    best_oof_predictions = best_trial.user_attrs["oof_predictions"]
    best_params = best_trial.user_attrs["best_params"]
    fold_scores = best_trial.user_attrs["fold_scores"]
    fold_models = best_trial.user_attrs["fold_models"]
    best_iterations = best_trial.user_attrs["best_iterations"]
    
    print(f"[{model_name}] 超參數搜索完成 - 最佳AUPRC (PR-AUC)分數: {study.best_value:.4f}")
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"[{model_name}] 研究統計:")
    print(f"  剪枝Trials數: {len(pruned_trials)}")
    print(f"  完成Trials數: {len(complete_trials)}")
    print(f"  剪枝比例: {len(pruned_trials) / len(study.trials):.2f}")
    
    return best_oof_predictions, best_params, fold_scores, fold_models, best_iterations

def predict_test_data_xgb(X_test: pd.DataFrame, fold_models, best_iterations, fold_scores) -> np.ndarray:
    """使用模型預測測試資料"""
    weights = np.array(fold_scores) / np.sum(fold_scores)
    test_predictions = np.zeros(len(X_test))
    dtest = xgb.DMatrix(X_test)
    
    for fold, (model, best_iter) in enumerate(zip(fold_models, best_iterations)):
        fold_predictions = model.predict(dtest, iteration_range=(0, best_iter)) * weights[fold]
        test_predictions += fold_predictions
        
    return test_predictions

def train_combination_xgb(combo_name, categories, feature_dfs, y, num_trials, cv):
    
    """訓練單個特徵組合的XGBoost模型"""
    combined_features = pd.concat([feature_dfs[category] for category in categories], axis=1)
    results = {}
    
    model_name_xgb = f"{combo_name}_xgb"
    print(f"\n訓練 XGBoost 模型: {model_name_xgb} (使用特徵類別: {', '.join(categories)})")
    oof_preds_xgb, model_params_xgb, fold_scores_xgb, fold_models_xgb, model_iterations_xgb = tune_and_get_oof_predictions_xgb(
        combined_features, y, model_name_xgb, num_trials, cv)
    results[model_name_xgb] = {
        'oof_preds': oof_preds_xgb,
        'params': model_params_xgb,
        'fold_scores': fold_scores_xgb,
        'fold_models': fold_models_xgb,
        'iterations': model_iterations_xgb
    }
    
    return results

def save_results(all_oof_predictions, y, model_weights, test_preds, feature_combinations, all_model_data):
    """保存所有結果檔案"""
    # 保存OOF預測結果
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_indices = np.zeros(len(y), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(kf.split(range(len(y)), y)):
        fold_indices[val_idx] = fold_idx + 1
    
    oof_data = {'fold': fold_indices, 'target': y.values}
    for model_name, oof_preds in all_oof_predictions.items():
        oof_data[f'{model_name}_pred_proba'] = oof_preds
    
    oof_df = pd.DataFrame(oof_data)
    oof_df['weighted_pred_proba'] = sum(
        oof_df[f'{model_name}_pred_proba'] * weight 
        for model_name, weight in model_weights.items()
    )
    oof_df.to_csv('result_XGB/xgb_models_oof_predictions.csv', index=False)
    
    # 保存加權預測結果
    weighted_test_predictions = np.zeros(len(test_preds[f'{list(model_weights.keys())[0]}_pred_proba']))
    for model_name, weight in model_weights.items():
        if f'{model_name}_pred_proba' in test_preds:
            weighted_test_predictions += test_preds[f'{model_name}_pred_proba'] * weight
    
    pd.DataFrame({'prediction': weighted_test_predictions}).to_csv(
        'result_XGB/final_predictions_xgb_weighted.csv', index=False)
    
    # 保存元模型資料
    meta_train_columns = {model_name: all_oof_predictions[model_name] for model_name in all_oof_predictions.keys()}
    meta_train_df = pd.DataFrame(meta_train_columns)
    meta_train_df['target'] = y.values
    
    meta_test_columns = {model_name: test_preds[f'{model_name}_pred_proba'] for model_name in all_oof_predictions.keys()}
    meta_test_df = pd.DataFrame(meta_test_columns)
    
    meta_train_df.to_csv('result_XGB/meta_train_xgb.csv', index=False)
    meta_test_df.to_csv('result_XGB/meta_test_xgb.csv', index=False)
    
    print("\n===== XGBoost模型性能比較 =====")
    for combo_name, _ in feature_combinations:
        xgb_model = f"{combo_name}_xgb"
        if xgb_model in all_oof_predictions:
            xgb_score = average_precision_score(y, all_oof_predictions[xgb_model])
            print(f"組合 {combo_name}: XGBoost AUPRC: {xgb_score:.4f}")
    
    weighted_auprc = average_precision_score(y, oof_df['weighted_pred_proba'])
    print(f"加權集成 AUPRC: {weighted_auprc:.4f}")

def main():
    """主函數：執行資料載入、特徵分割、XGBoost模型訓練與評估、測試資料預測"""
    np.random.seed(42)
    
    # 載入訓練資料
    train_path = "df_1000.csv"
    X, y = load_data(train_path)
    
    # 分割特徵
    feature_dfs = split_features_by_categories(X)
    for category_name, category_df in feature_dfs.items():
        print(f"{category_name} 特徵數量: {category_df.shape[1]}")
    other_dfs = [df for name, df in feature_dfs.items() if name not in ['上市加權', '技術指標']]
    if other_dfs:
        feature_dfs['其他'] = pd.concat(other_dfs, axis=1)
    feature_dfs = {k: v for k, v in feature_dfs.items() if k in ['上市加權', '技術指標', '其他']}

    # 獲取特徵組合
    feature_combinations = get_feature_combinations()
    print(f"總特徵組合數量: {len(feature_combinations)}")
    
    # 設定超參數搜索參數
    num_trials = 20
    cv = 5
    
    # 訓練模型
    print(f"\n開始訓練所有特徵組合（總共 {len(feature_combinations)} 個組合）...")
    n_jobs = 7
    print(f"使用 {n_jobs} 個並行任務進行特徵組合訓練")
    # 並行訓練不同特徵組合
    results = Parallel(n_jobs=n_jobs)(
        delayed(train_combination_xgb)(combo_name, categories, feature_dfs, y, num_trials, cv)
        for combo_name, categories in feature_combinations
    )
    
    # 合併結果
    all_oof_predictions = {}
    all_model_params = {}
    all_fold_scores = {}
    all_model_iterations = {}
    all_fold_models = {}
    all_model_data = {}
    
    for result in results:
        for model_name, data in result.items():
            all_oof_predictions[model_name] = data['oof_preds']
            all_model_params[model_name] = data['params']
            all_fold_scores[model_name] = data['fold_scores']
            all_model_iterations[model_name] = data['iterations']
            all_fold_models[model_name] = data['fold_models']
            all_model_data[model_name] = data
    
    # 計算模型權重
    model_auprc_scores = {}
    for model_name in all_oof_predictions.keys():
        model_auprc = average_precision_score(y, all_oof_predictions[model_name])
        model_auprc_scores[model_name] = model_auprc
        print(f"{model_name}: {model_auprc:.4f}")
    
    weights_sum = sum(model_auprc_scores.values())
    model_weights = {model_name: score / weights_sum for model_name, score in model_auprc_scores.items()}
    
    print(f"\n模型權重分配:")
    for model_name, weight in model_weights.items():
        print(f"  {model_name} 權重: {weight:.4f}")
    
    # 預測測試資料
    test_path = "Public_Test/public_x_100.csv"
    X_test, _ = load_data(test_path, is_test=True)
    test_feature_dfs = split_features_by_categories(X_test)
    other_dfs = [df for name, df in test_feature_dfs.items() if name not in ['上市加權', '技術指標']]
    if other_dfs:
        test_feature_dfs['其他'] = pd.concat(other_dfs, axis=1)
    test_feature_dfs = {k: v for k, v in test_feature_dfs.items() if k in ['上市加權', '技術指標', '其他']}

    
    print("\n使用各XGBoost模型預測測試資料...")
    test_preds = {}
    for combo_name, categories in feature_combinations:
        combined_test_features = pd.concat([test_feature_dfs[category] for category in categories], axis=1)
        model_name_xgb = f"{combo_name}_xgb"
        # 直接使用內存中的模型和參數進行預測
        test_pred_xgb = predict_test_data_xgb(
            combined_test_features, 
            all_fold_models[model_name_xgb], 
            all_model_iterations[model_name_xgb],
            all_fold_scores[model_name_xgb]
        )
        test_preds[f'{model_name_xgb}_pred_proba'] = test_pred_xgb
    
    # 保存結果
    save_results(all_oof_predictions, y, model_weights, test_preds, feature_combinations, all_model_data)
    print("\n所有預測和模型結果已保存完成")

if __name__ == "__main__":
    main()