import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import optuna
from optuna.pruners import MedianPruner
import lightgbm as lgb
import os
import joblib
from typing import Any, Dict, Tuple, List, Optional

# 創建模型保存目錄
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

def lgb_prauc(preds, train_data):
    """自定義 AUPRC 評估函數，用於 LightGBM 的早停機制
    
    參數:
        preds: 模型預測值
        train_data: 包含標籤的資料集
    
    返回:
        評估指標名稱、AUPRC值、是否越高越好
    """
    labels = train_data.get_label()
    auprc = average_precision_score(labels, preds)
    return 'prauc', auprc, True

def load_data(data_path: str, is_test: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """載入資料並分離特徵與目標變數
    
    參數:
        data_path: CSV 資料檔路徑
        is_test: 是否為測試資料(沒有目標變數)
        
    返回:
        X: 特徵資料
        y: 目標變數（測試資料時為None）
    """
    df = pd.read_csv(data_path)
    if is_test:
        X = df  # 所有列都是特徵
        y = None
    else:
        X = df.iloc[:, :-1]  # 所有列除了最後一列
        y = df.iloc[:, -1]   # 最後一列為目標變數
    
    return X, y

def split_features(X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """依據欄位名稱分割特徵為技術指標特徵和其他特徵
    
    參數:
        X: 特徵資料
        
    返回:
        X_tech: 技術指標特徵
        X_other: 其他特徵
    """
    # 找出包含「技術指標」字樣的欄位名稱
    tech_indicator_cols = [col for col in X.columns if '技術指標' in col]
    other_cols = [col for col in X.columns if '技術指標' not in col]
    
    X_tech = X[tech_indicator_cols]
    X_other = X[other_cols]
    
    return X_tech, X_other

def objective(trial, X, y, cv=5, model_name="model"):
    """Optuna 超參數搜索的目標函數，使用 Optuna 的 MedianPruner 進行剪枝
    
    參數:
        trial: Optuna trial 物件
        X: 特徵資料
        y: 目標變數
        cv: 交叉驗證折數
        model_name: 模型名稱，用於保存模型
    
    返回:
        平均 AUPRC 分數
    """
    # 定義超參數搜索空間
    params = {
        "objective": "binary",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "bagging_freq": 1,
        "metric": "",  # 清空原生metric，使用自定義評估函數
        "num_threads": 4,
        
        # 模型結構參數
        "max_depth": trial.suggest_int("max_depth", 3, 11),
        "num_leaves": trial.suggest_int("num_leaves", 2, 2**10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        
        # 學習與正則化參數
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-9, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-9, 10.0, log=True),
        
        # 採樣參數
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
    }
    
    # 交叉驗證評估
    cv_scores = []
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # 創建空的預測陣列
    oof_predictions = np.zeros(len(X))
    
    # 保存每個fold的模型和最佳迭代次數
    fold_models = []
    best_iterations = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 準備資料集
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # 訓練模型（使用足夠的迭代次數）
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            feval=lgb_prauc,
            num_boost_round=2000,  # 設定足夠大的迭代次數
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(100)
            ],
        )
        
        # 保存最佳迭代次數
        best_iterations.append(model.best_iteration)
        
        # 使用最佳迭代次數進行預測以評估模型
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        
        # 計算AUPRC
        auprc = average_precision_score(y_val, y_pred)
        cv_scores.append(auprc)
        
        # 儲存預測結果
        oof_predictions[val_idx] = y_pred
        
        # 保存fold模型
        fold_models.append(model)
        
        # 報告當前折的結果給Optuna，用於剪枝決策
        trial.report(auprc, fold)
        
        # 檢查是否應該剪枝
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    # 保存這個trial所有fold的模型
    trial.set_user_attr("fold_models", fold_models)
    # 保存這個trial的best_iterations
    trial.set_user_attr("best_iterations", best_iterations)
    # 儲存這個trial的out-of-fold預測
    trial.set_user_attr("oof_predictions", oof_predictions)
    # 儲存每個fold的分數，用於之後的加權
    trial.set_user_attr("fold_scores", cv_scores)
    # 儲存最佳模型參數
    trial.set_user_attr("best_params", params)
    
    return np.mean(cv_scores)

def tune_and_get_oof_predictions(X: pd.DataFrame, y: pd.Series, model_name: str, num_trials: int = 50, cv: int = 5) -> Tuple[np.ndarray, Dict, List[float], List[Any]]:
    """使用 Optuna 調整超參數並儲存 out-of-fold 預測結果和訓練好的模型
    
    參數:
        X: 特徵資料
        y: 目標變數
        model_name: 模型名稱（用於輸出結果）
        num_trials: Optuna 嘗試次數
        cv: 交叉驗證折數
        
    返回:
        oof_preds: out-of-fold 預測結果
        best_params: 最佳參數
        fold_scores: 每個fold的分數
        fold_models: 每個fold訓練好的模型
    """
    # 檢查是否已有保存的模型和結果
    model_path = f"models/{model_name}"
    if os.path.exists(f"{model_path}_study.pkl"):
        print(f"載入已有的{model_name}模型和結果")
        study = joblib.load(f"{model_path}_study.pkl")
        
        best_trial = study.best_trial
        best_oof_predictions = best_trial.user_attrs["oof_predictions"]
        best_params = best_trial.user_attrs["best_params"]
        fold_scores = best_trial.user_attrs["fold_scores"]
        fold_models = best_trial.user_attrs["fold_models"]
        best_iterations = best_trial.user_attrs["best_iterations"]
        
        print(f"{model_name} 最佳AUPRC (PR-AUC)分數: {study.best_value:.4f}")
        
        return best_oof_predictions, best_params, fold_scores, fold_models, best_iterations
    
    # 建立 Optuna 研究，使用中位數剪枝
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    
    # 執行超參數搜索
    study.optimize(
        lambda trial: objective(trial, X, y, cv, model_name), 
        n_trials=num_trials
    )
    
    # 獲取最佳 trial 的資訊
    best_trial = study.best_trial
    best_oof_predictions = best_trial.user_attrs["oof_predictions"]
    best_params = best_trial.user_attrs["best_params"]
    fold_scores = best_trial.user_attrs["fold_scores"]
    fold_models = best_trial.user_attrs["fold_models"]
    best_iterations = best_trial.user_attrs["best_iterations"]
    
    # 顯示結果
    print(f"{model_name} 最佳AUPRC (PR-AUC)分數: {study.best_value:.4f}")
    
    # 顯示剪枝統計
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"研究統計:")
    print(f"  剪枝Trials數: {len(pruned_trials)}")
    print(f"  完成Trials數: {len(complete_trials)}")
    print(f"  剪枝比例: {len(pruned_trials) / len(study.trials):.2f}")
    
    # 保存study物件以便後續使用
    joblib.dump(study, f"{model_path}_study.pkl")
    
    # 保存每個fold的模型
    for i, model in enumerate(fold_models):
        model.save_model(f"{model_path}_fold{i}.txt")
    
    # 保存最佳參數
    joblib.dump(best_params, f"{model_path}_best_params.pkl")
    
    # 保存fold分數
    joblib.dump(fold_scores, f"{model_path}_fold_scores.pkl")
    
    # 保存最佳迭代次數
    joblib.dump(best_iterations, f"{model_path}_best_iterations.pkl")
    
    # 返回預測結果、最佳參數、fold分數和模型
    return best_oof_predictions, best_params, fold_scores, fold_models, best_iterations

def predict_test_data(X_test: pd.DataFrame, model_name: str) -> np.ndarray:
    """使用保存的模型預測測試資料
    
    參數:
        X_test: 測試特徵資料
        model_name: 模型名稱，用於載入模型
        
    返回:
        test_predictions: 測試資料的預測結果
    """
    model_path = f"models/{model_name}"
    
    # 載入fold分數作為權重
    fold_scores = joblib.load(f"{model_path}_fold_scores.pkl")
    
    # 規格化fold分數作為權重
    weights = np.array(fold_scores) / np.sum(fold_scores)
    
    # 使用每個fold的模型進行預測
    test_predictions = np.zeros(len(X_test))
    
    # 獲取fold數量
    num_folds = len(fold_scores)
    
    for fold in range(num_folds):
        # 載入此fold的模型
        model = lgb.Booster(model_file=f"{model_path}_fold{fold}.txt")
        
        # 預測並加權
        fold_predictions = model.predict(X_test) * weights[fold]
        test_predictions += fold_predictions
    
    return test_predictions

def train_stacking_lr_model(oof_df: pd.DataFrame, cv: int = 5, num_trials: int = 30) -> Tuple[Dict[int, float], StratifiedKFold, pd.DataFrame, pd.Series, List[LogisticRegression], Dict]:
    """使用交叉驗證訓練Stacking模型(Logistic Regression)，並使用Optuna優化參數
    
    參數:
        oof_df: 包含各個模型OOF預測和目標變數的DataFrame
        cv: 交叉驗證折數
        num_trials: Optuna嘗試次數
        
    返回:
        fold_prauc_scores: 每個fold的PRAUC分數，用於加權平均
        kf: 交叉驗證分割器，用於後續預測
        X_stack: 用於訓練的特徵資料
        y_stack: 用於訓練的目標變數
        fold_models: 每個fold訓練好的模型
        best_params: 最佳參數
    """
    # 準備stacking訓練資料
    X_stack = oof_df.drop(['target', 'fold', 'weighted_pred_proba'] if 'weighted_pred_proba' in oof_df.columns else ['target', 'fold'], axis=1)
    y_stack = oof_df['target']
    
    # 檢查是否已有保存的stacking模型和結果
    if (os.path.exists("models/stacking_lr_fold_scores.pkl") and 
        os.path.exists("models/stacking_lr_fold_models.pkl") and 
        os.path.exists("models/stacking_lr_best_params.pkl")):
        
        print("載入已有的stacking Logistic Regression模型和結果")
        fold_prauc_scores = joblib.load("models/stacking_lr_fold_scores.pkl")
        fold_models = joblib.load("models/stacking_lr_fold_models.pkl")
        best_params = joblib.load("models/stacking_lr_best_params.pkl")
        
        # 直接使用各fold原始評分的平均值作為整體評分
        fold_scores_values = list(fold_prauc_scores.values())
        fold_avg_prauc = sum(fold_scores_values) / len(fold_scores_values)
        print(f"Stacking LR模型整體PRAUC (fold平均值): {fold_avg_prauc:.4f}")
        
        # 創建交叉驗證分割器 (僅用於後續操作)
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            
        return fold_prauc_scores, kf, X_stack, y_stack, fold_models, best_params
    
    # 如果沒有保存的模型，則訓練新模型
    print("沒有找到保存的stacking模型，開始訓練新模型")
    
    # 建立Optuna研究，使用MedianPruner
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    
    # 初始化交叉驗證
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    fold_prauc_scores = {}  # 儲存每個fold的PRAUC分數
    oof_predictions = np.zeros(len(oof_df))  # 存儲stacking模型的OOF預測
    fold_models = []  # 保存每個fold的模型
    best_params = {}  # 保存每個fold的最佳參數
    
    # 記錄每個fold的分割索引，用於後續分析
    all_fold_indices = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_stack, y_stack)):
        all_fold_indices.append((train_idx, val_idx))
    
    # 對每個fold進行優化和訓練
    for fold_idx, (train_idx, val_idx) in enumerate(all_fold_indices):
        fold_num = fold_idx + 1
        
        # 分割訓練集和驗證集
        X_fold_train, X_fold_val = X_stack.iloc[train_idx], X_stack.iloc[val_idx]
        y_fold_train, y_fold_val = y_stack.iloc[train_idx], y_stack.iloc[val_idx]
        
        # 打印當前fold的正樣本比例
        train_pos_ratio = y_fold_train.mean()
        val_pos_ratio = y_fold_val.mean()
        print(f"Fold {fold_num} - 訓練集正樣本比例: {train_pos_ratio:.4f}, 驗證集正樣本比例: {val_pos_ratio:.4f}")
        
        # 定義當前fold的objective函數
        def fold_objective(trial):
            # 定義超參數搜索空間 (針對LR模型)
            C = trial.suggest_float("C", 1e-5, 100.0, log=True)
            solver = trial.suggest_categorical("solver", ["liblinear", "saga"])
            penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
            max_iter = trial.suggest_int("max_iter", 100, 1000)
            class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
            
            # 建立LR模型
            lr = LogisticRegression(
                C=C,
                solver=solver,
                penalty=penalty,
                max_iter=max_iter,
                class_weight=class_weight,
                random_state=42
            )
            
            # 訓練模型
            lr.fit(X_fold_train, y_fold_train)
            
            # 預測驗證集（獲取機率值而非類別）
            val_preds = lr.predict_proba(X_fold_val)[:, 1]
            
            # 計算PRAUC
            prauc = average_precision_score(y_fold_val, val_preds)
            
            return prauc
        
        # 建立Optuna研究
        study = optuna.create_study(direction="maximize", pruner=pruner)
        
        # 執行超參數搜索
        study.optimize(fold_objective, n_trials=num_trials)
        
        # 獲取最佳參數
        best_trial = study.best_trial
        fold_best_params = best_trial.params
        best_params[fold_num] = fold_best_params
        
        # 用最佳參數重新訓練模型
        lr_model = LogisticRegression(
            C=fold_best_params["C"],
            solver=fold_best_params["solver"],
            penalty=fold_best_params["penalty"],
            max_iter=fold_best_params["max_iter"],
            class_weight=fold_best_params["class_weight"],
            random_state=42
        )
        
        # 訓練模型
        lr_model.fit(X_fold_train, y_fold_train)
        
        # 保存模型
        fold_models.append(lr_model)
        
        # 預測驗證集
        val_preds = lr_model.predict_proba(X_fold_val)[:, 1]
        
        # 計算此fold的PRAUC
        fold_prauc = average_precision_score(y_fold_val, val_preds)
        fold_prauc_scores[fold_num] = fold_prauc
        
        # 儲存OOF預測
        oof_predictions[val_idx] = val_preds
        
        print(f"Stacking LR模型 Fold {fold_num} 最佳PRAUC: {fold_prauc:.4f}")
    
    # 使用fold平均值作為模型性能指標
    fold_avg_prauc = sum(fold_prauc_scores.values()) / len(fold_prauc_scores)
    print(f"Stacking LR模型整體PRAUC (fold平均值): {fold_avg_prauc:.4f}")
    
    # 保存stacking模型結果
    joblib.dump(fold_prauc_scores, "models/stacking_lr_fold_scores.pkl")
    joblib.dump(fold_models, "models/stacking_lr_fold_models.pkl")
    joblib.dump(best_params, "models/stacking_lr_best_params.pkl")
    joblib.dump(oof_predictions, "models/stacking_lr_oof_predictions.pkl")
    
    # 輸出正樣本分布
    print("\n各fold中正樣本比例:")
    for fold in range(1, max(oof_df['fold'])+1):
        fold_mask = oof_df['fold'] == fold
        fold_total = sum(fold_mask)
        fold_positive = sum(y_stack[fold_mask])
        if fold_total > 0:
            fold_ratio = fold_positive / fold_total
            print(f"Fold {fold}: {fold_ratio:.4f} ({fold_positive}/{fold_total})")
    
    # 返回最終模型、每個fold的PRAUC分數、交叉驗證分割器以及特徵和目標變數
    return fold_prauc_scores, kf, X_stack, y_stack, fold_models, best_params

def predict_with_stacking_lr(test_preds_df: pd.DataFrame) -> np.ndarray:
    """使用保存的stacking LR模型預測測試資料
    
    參數:
        test_preds_df: 包含基礎模型預測結果的DataFrame
        
    返回:
        final_predictions: 最終加權預測結果
    """
    # 載入stacking模型和權重
    fold_models = joblib.load("models/stacking_lr_fold_models.pkl")
    fold_scores = joblib.load("models/stacking_lr_fold_scores.pkl")
    
    # 計算權重總和以便標準化
    weights_sum = sum(fold_scores.values())
    
    # 初始化最終預測陣列
    final_predictions = np.zeros(len(test_preds_df))
    
    # 使用每個fold的模型進行預測並加權
    for fold_num, fold_model in enumerate(fold_models, 1):
        # 預測測試集（獲取機率值而非類別）
        fold_test_pred = fold_model.predict_proba(test_preds_df)[:, 1]
        
        # 使用此fold的PRAUC作為權重進行加權
        weight = fold_scores[fold_num] / weights_sum
        final_predictions += fold_test_pred * weight
        
        print(f"Fold {fold_num} 權重: {weight:.4f}, PRAUC: {fold_scores[fold_num]:.4f}")
    
    return final_predictions

def predict_with_stacking_lr(test_preds_df: pd.DataFrame) -> np.ndarray:
    """使用保存的stacking LR模型預測測試資料
    
    參數:
        test_preds_df: 包含基礎模型預測結果的DataFrame
        
    返回:
        final_predictions: 最終加權預測結果
    """
    # 載入stacking模型和權重
    fold_models = joblib.load("models/stacking_lr_fold_models.pkl")
    fold_scores = joblib.load("models/stacking_lr_fold_scores.pkl")
    
    # 計算權重總和以便標準化
    weights_sum = sum(fold_scores.values())
    
    # 初始化最終預測陣列
    final_predictions = np.zeros(len(test_preds_df))
    
    # 使用每個fold的模型進行預測並加權
    for fold_num, fold_model in enumerate(fold_models, 1):
        # 預測測試集（獲取機率值而非類別）
        fold_test_pred = fold_model.predict_proba(test_preds_df)[:, 1]
        
        # 使用此fold的PRAUC作為權重進行加權
        weight = fold_scores[fold_num] / weights_sum
        final_predictions += fold_test_pred * weight
        
        print(f"Fold {fold_num} 權重: {weight:.4f}, PRAUC: {fold_scores[fold_num]:.4f}")
    
    return final_predictions

def main():

    """主函數：執行資料載入、特徵分割、三個模型訓練與評估、測試資料預測"""
    # 設定隨機種子
    np.random.seed(42)
    
    # 載入訓練資料
    train_path = "df_process_2.csv"
    X, y = load_data(train_path)
    if 'ID' in X.columns:
        X = X.drop(columns=['ID'])
    
    # 依據欄位名稱是否包含「技術指標」來分割特徵
    X_tech, X_other = split_features(X)

    
    print(f"技術指標特徵數量: {X_tech.shape[1]}")
    print(f"其他特徵數量: {X_other.shape[1]}")
    print(f"總特徵數量: {X.shape[1]}")
    
    # 設定Optuna和交叉驗證參數
    num_trials = 20  # 實際使用中可以調整
    cv = 5  # 建議使用5折或10折交叉驗證
    
    # 訓練三個模型並獲取OOF預測與模型
    
    # 模型1: 僅使用技術指標特徵
    print("\n訓練模型1: 僅使用技術指標特徵")
    oof_tech, tech_params, tech_fold_scores, tech_models, tech_iterations = tune_and_get_oof_predictions(
        X_tech, y, "model1_tech", num_trials, cv)
    
    # 模型2: 僅使用其他特徵
    print("\n訓練模型2: 僅使用其他特徵")
    oof_other, other_params, other_fold_scores, other_models, other_iterations = tune_and_get_oof_predictions(
        X_other, y, "model2_other", num_trials, cv)
    
    # 模型3: 使用所有特徵
    print("\n訓練模型3: 使用所有特徵")
    oof_all, all_params, all_fold_scores, all_models, all_iterations = tune_and_get_oof_predictions(
        X, y, "model3_all", num_trials, cv)
    
    # 先獲取每個樣本的fold編號
    kf_for_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    fold_indices = np.zeros(len(X), dtype=int)

    for fold_idx, (_, val_idx) in enumerate(kf_for_fold.split(X, y)):
        fold_indices[val_idx] = fold_idx + 1

    # 建立包含三個模型OOF預測和fold資訊的DataFrame
    oof_df = pd.DataFrame({
        'fold': fold_indices,
        'oof_tech_pred_proba': oof_tech,
        'oof_other_pred_proba': oof_other,
        'oof_all_pred_proba': oof_all,
        'target': y.values
    })

    # 計算三個模型的AUPRC分數
    tech_auprc = average_precision_score(oof_df['target'], oof_df['oof_tech_pred_proba'])
    other_auprc = average_precision_score(oof_df['target'], oof_df['oof_other_pred_proba'])
    all_auprc = average_precision_score(oof_df['target'], oof_df['oof_all_pred_proba'])
    
    # 計算權重
    weights_sum = tech_auprc + other_auprc + all_auprc
    tech_weight = tech_auprc / weights_sum
    other_weight = other_auprc / weights_sum
    all_weight = all_auprc / weights_sum
    
    # 添加加權預測
    oof_df['weighted_pred_proba'] = (
        oof_df['oof_tech_pred_proba'] * tech_weight +
        oof_df['oof_other_pred_proba'] * other_weight +
        oof_df['oof_all_pred_proba'] * all_weight
    )
    
    # 重新排列欄位順序
    oof_df = oof_df[['fold', 'oof_tech_pred_proba', 'oof_other_pred_proba', 'oof_all_pred_proba', 'weighted_pred_proba', 'target']]

    print("\n三個模型的OOF預測AUPRC分數:")
    print(f"模型1(技術指標): {tech_auprc:.4f}")
    print(f"模型2(其他特徵): {other_auprc:.4f}")
    print(f"模型3(所有特徵): {all_auprc:.4f}")
    
    # 輸出權重情況
    print(f"\n模型權重分配:")
    print(f"  技術指標模型權重: {tech_weight:.4f}")
    print(f"  其他特徵模型權重: {other_weight:.4f}")
    print(f"  全部特徵模型權重: {all_weight:.4f}")
    
    # 計算加權預測的PRAUC
    weighted_auprc = average_precision_score(oof_df['target'], oof_df['weighted_pred_proba'])
    print(f"加權預測AUPRC: {weighted_auprc:.4f}")

    # 儲存OOF預測結果
    oof_df.to_csv('results/three_models_oof_predictions.csv', index=False)
    
    # 訓練Stacking Logistic Regression模型（使用交叉驗證和Optuna優化）
    print("\n訓練Stacking Logistic Regression模型（使用交叉驗證和Optuna優化）")
    # 這裡改用train_stacking_lr_model而非原本的train_stacking_lgbm_model
    stacking_fold_scores, kf, X_stack, y_stack, stacking_models, best_params = train_stacking_lr_model(oof_df, cv=cv, num_trials=20)
    
    # 載入測試資料
    test_path = "Public_Test/public_x_process_2.csv"
    X_test, _ = load_data(test_path, is_test=True)
    
    # 依據欄位名稱分割測試資料特徵
    X_test_tech, X_test_other = split_features(X_test)

    # 使用三個模型預測測試資料（直接使用保存的模型）
    print("\n使用各模型預測測試資料")
    test_tech_pred = predict_test_data(X_test_tech, "model1_tech")
    test_other_pred = predict_test_data(X_test_other, "model2_other")
    test_all_pred = predict_test_data(X_test, "model3_all")

    # 整合三個模型的預測結果
    test_preds_df = pd.DataFrame({
        'oof_tech_pred_proba': test_tech_pred,
        'oof_other_pred_proba': test_other_pred,
        'oof_all_pred_proba': test_all_pred
    })

    # 使用stacking LR模型進行最終預測
    print("\n使用Stacking Logistic Regression模型進行最終預測")
    # 這裡改用predict_with_stacking_lr而非原本的predict_with_stacking_lgbm
    final_predictions = predict_with_stacking_lr(test_preds_df)

    # 保存最終預測結果
    results_df = pd.DataFrame({
        'prediction': final_predictions
    })
    results_df.to_csv('results/final_predictions_lr.csv', index=False)

    print("\n已將三個模型的out-of-fold預測儲存到 results/three_models_oof_predictions.csv")
    print("Stacking LR模型最終預測結果已儲存到 results/final_predictions_lr.csv")

    # 建立meta_model訓練資料
    meta_train_df = pd.DataFrame({
        'oof_tech_pred_proba': oof_tech,
        'oof_other_pred_proba': oof_other,
        'oof_all_pred_proba': oof_all,
        'target': y.values
    })

    # 建立meta_model測試資料
    meta_test_df = pd.DataFrame({
        'oof_tech_pred_proba': test_tech_pred,
        'oof_other_pred_proba': test_other_pred,
        'oof_all_pred_proba': test_all_pred
    })
    meta_train_df.to_csv('results/meta_train.csv', index=False)
    meta_test_df.to_csv('results/meta_test.csv', index=False)
    print("\n已將元模型訓練資料儲存到 results/meta_train.csv")
    print("已將元模型測試資料儲存到 results/meta_test.csv")


if __name__ == "__main__":
    main()