import argparse
import os
import os.path as osp
import time
import pandas as pd
from typing import Any, Optional, Dict, List, Tuple
import json
import numpy as np
import optuna
import torch
from torch.nn import BCEWithLogitsLoss, Module
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torcheval.metrics.functional import binary_auprc
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import gc
import pickle

import torch_frame as tf
from torch_frame import stype
from torch_frame.data import DataLoader, Dataset
from torch_frame.nn.encoder import EmbeddingEncoder, LinearBucketEncoder
from torch_frame.nn.models import (
    MLP, ExcelFormer, FTTransformer, ResNet,
    TabNet, TabTransformer, Trompt,
)
from torch_frame.gbdt import LightGBM
from torch_frame.typing import TaskType

# 定義全局變數
OUTPUT_DIR = 'train_outputs'  # 統一輸出目錄

# 創建必要的目錄
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 文件路徑
COMBINED_VAL_FILE = os.path.join(OUTPUT_DIR, 'combined_validation.csv')  # 合併驗證結果
FINAL_TEST_FILE = os.path.join(OUTPUT_DIR, 'final_test_predictions.csv')  # 最終測試預測
COMBINED_METRICS_FILE = os.path.join(OUTPUT_DIR, 'combined_metrics.json')  # 整合模型指標

TRAIN_CONFIG_KEYS = ["batch_size", "gamma_rate", "base_lr"]
NUM_CV_FOLDS = 5  # 5 折交叉驗證

# 深度學習模型類型
DL_MODEL_TYPES = [
    'TabNet', 'MLP', 'ResNet'
    ]

# 清理GPU記憶體的函數
def clean_gpu_memory():
    """清理GPU記憶體以避免記憶體不足問題"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 載入測試集並前處理的函數
def load_test_dataset(col_to_stype):
    """載入測試集並應用與訓練集相同的預處理"""
    print("Loading test dataset...")
    test_df = pd.read_csv('final_meta_test.csv')
    
    # 保存原始ID以便最終輸出
    test_ids = test_df['id'].copy() if 'id' in test_df.columns else test_df.index
    
    # 如果ID存在則刪除
    if 'id' in test_df.columns:
        test_df = test_df.drop('id', axis=1)
    
    # 確保測試集有target欄位，-1表示未知
    if 'target' not in test_df.columns:
        test_df['target'] = -1
    
    # 建立並實體化測試集
    test_dataset = Dataset(df=test_df, col_to_stype=col_to_stype, target_col='target')
    test_dataset.materialize()
    
    return test_dataset, test_ids

# 针對嚴重不平衡情況優化的Focal Loss類
class FocalLoss(Module):
    """
    針對極度不平衡(正樣本<10%)的二元分類優化的Focal Loss
    
    FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
    """
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            alpha_weight = torch.where(targets == 1, 
                                      torch.ones_like(targets) * self.alpha,
                                      torch.ones_like(targets) * (1 - self.alpha))
            focal_weight = alpha_weight * focal_weight
        
        focal_loss = focal_weight * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss
        
def get_model_config(model_type: str, full_dataset, full_tensor_frame):
    """根據模型類型獲取模型配置和訓練配置搜索空間"""
    model_cls = None
    col_stats = full_dataset.col_stats
    tensor_frame = full_tensor_frame
    
    # 處理深度學習模型
    if model_type == 'TabNet':
        model_search_space = {
            'split_attn_channels': [64, 128, 256],
            'split_feat_channels': [64, 128, 256],
            'gamma': [1., 1.2, 1.5],
            'num_layers': [4, 6, 8],
        }
        train_search_space = {
            'batch_size': [2048, 4096],
            'base_lr': [0.001, 0.01],
            'gamma_rate': [0.9, 0.95, 1.],
        }
        model_cls = TabNet
    elif model_type == 'FTTransformer':
        model_search_space = {
            'channels': [64, 128, 256],
            'num_layers': [4, 6, 8],
        }
        train_search_space = {
            'batch_size': [256, 512],
            'base_lr': [0.0001, 0.001],
            'gamma_rate': [0.9, 0.95, 1.],
        }
        model_cls = FTTransformer
    elif model_type == 'FTTransformerBucket':
        model_search_space = {
            'channels': [64, 128, 256],
            'num_layers': [4, 6, 8],
        }
        train_search_space = {
            'batch_size': [256, 512],
            'base_lr': [0.0001, 0.001],
            'gamma_rate': [0.9, 0.95, 1.],
        }
        model_cls = FTTransformer
    elif model_type == 'ResNet':
        model_search_space = {
            'channels': [64, 128, 256],
            'num_layers': [4, 6, 8],
        }
        train_search_space = {
            'batch_size': [256, 512],
            'base_lr': [0.0001, 0.001],
            'gamma_rate': [0.9, 0.95, 1.],
        }
        model_cls = ResNet
    elif model_type == 'MLP':
        model_search_space = {
            'channels': [64, 128, 256],
            'num_layers': [1, 2, 4],
        }
        train_search_space = {
            'batch_size': [256, 512],
            'base_lr': [0.0001, 0.001],
            'gamma_rate': [0.9, 0.95, 1.],
        }
        model_cls = MLP
    elif model_type == 'TabTransformer':
        model_search_space = {
            'channels': [16, 32, 64, 128],
            'num_layers': [4, 6, 8],
            'num_heads': [4, 8],
            'encoder_pad_size': [2, 4],
            'attn_dropout': [0, 0.2],
            'ffn_dropout': [0, 0.2],
        }
        train_search_space = {
            'batch_size': [128, 256],
            'base_lr': [0.0001, 0.001],
            'gamma_rate': [0.9, 0.95, 1.],
        }
        model_cls = TabTransformer
    elif model_type == 'Trompt':
        model_search_space = {
            'channels': [64, 128, 192],
            'num_layers': [4, 6, 8],
            'num_prompts': [64, 128, 192],
        }
        train_search_space = {
            'batch_size': [128, 256],
            'base_lr': [0.01, 0.001],
            'gamma_rate': [0.9, 0.95, 1.],
        }
        if tensor_frame.num_cols > 20:
            model_search_space['channels'] = [64, 128]
            model_search_space['num_prompts'] = [64, 128]
        elif tensor_frame.num_cols > 50:
            model_search_space['channels'] = [64]
            model_search_space['num_prompts'] = [64]
        model_cls = Trompt
    elif model_type == 'ExcelFormer':
        from torch_frame.transforms import (
            CatToNumTransform,
            MutualInformationSort,
        )

        local_col_stats = col_stats
        
        categorical_transform = CatToNumTransform()
        categorical_transform.fit(full_dataset.tensor_frame, local_col_stats)
        transformed_tensor_frame = categorical_transform(full_tensor_frame)
        local_col_stats = categorical_transform.transformed_stats

        mutual_info_sort = MutualInformationSort(task_type=full_dataset.task_type)
        mutual_info_sort.fit(transformed_tensor_frame, local_col_stats)
        transformed_tensor_frame = mutual_info_sort(transformed_tensor_frame)

        tensor_frame = transformed_tensor_frame
        col_stats = local_col_stats

        model_search_space = {
            'in_channels': [128, 256],
            'num_heads': [8, 16, 32],
            'num_layers': [4, 6, 8],
            'diam_dropout': [0, 0.2],
            'residual_dropout': [0, 0.2],
            'aium_dropout': [0, 0.2],
            'mixup': [None, 'feature', 'hidden'],
            'beta': [0.5],
            'num_cols': [tensor_frame.num_cols],
        }
        train_search_space = {
            'batch_size': [256, 512],
            'base_lr': [0.001],
            'gamma_rate': [0.9, 0.95, 1.],
        }
        model_cls = ExcelFormer

    assert model_cls is not None
    assert col_stats is not None
    
    assert set(train_search_space.keys()) == set(TRAIN_CONFIG_KEYS)
    
    # 返回ExcelFormer的變換對象，用於測試集變換
    transform_objects = {}
    if model_type == 'ExcelFormer':
        transform_objects = {
            'categorical_transform': categorical_transform,
            'mutual_info_sort': mutual_info_sort
        }
    
    return {
        'model_cls': model_cls,
        'col_stats': col_stats,
        'tensor_frame': tensor_frame,
        'model_search_space': model_search_space,
        'train_search_space': train_search_space,
        'transform_objects': transform_objects
    }

def train(
    model: Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device,
    loss_function: Module = None,
    out_channels: int = 1,
    verbose: bool = False
) -> float:
    model.train()
    loss_accum = total_count = 0
    
    if loss_function is None:
        loss_function = FocalLoss(alpha=0.5, gamma=2.0)

    # 使用tqdm來顯示進度條，根據verbose參數決定是否顯示
    progress_bar = tqdm(loader, desc=f'Epoch: {epoch}', disable=not verbose)
    for tf in progress_bar:
        tf = tf.to(device)
        y = tf.y
        if isinstance(model, ExcelFormer):
            pred, y = model(tf, mixup_encoded=True)
        elif isinstance(model, Trompt):
            pred = model(tf)
            num_layers = pred.size(1)
            pred = pred.view(-1, out_channels)
            y = tf.y.repeat_interleave(num_layers)
        else:
            pred = model(tf)

        if pred.size(1) == 1:
            pred = pred.view(-1, )
        
        y = y.to(torch.float)
        loss = loss_function(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        loss_accum += float(loss) * len(tf.y)
        total_count += len(tf.y)
        optimizer.step()
        
        # 更新進度條顯示當前的損失
        if verbose:
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
    return loss_accum / total_count


@torch.no_grad()
def evaluate(
    model: Module,
    loader: DataLoader,
    device,
    out_channels: int = 1
) -> float:
    """評估模型在驗證集上的PRAUC表現"""
    model.eval()
    
    # 收集所有的預測和標籤
    all_probs = []
    all_targets = []
    
    for tf in loader:
        tf = tf.to(device)
        pred = model(tf)
        if isinstance(model, Trompt):
            pred = pred.mean(dim=1)
        if pred.size(-1) == 1:
            pred = pred.view(-1)
        
        # sigmoid轉換為概率
        probs = torch.sigmoid(pred)
        
        # 收集批次的預測和標籤
        all_probs.append(probs.cpu())
        all_targets.append(tf.y.cpu())
    
    # 合併所有批次的結果
    all_probs = torch.cat(all_probs)
    all_targets = torch.cat(all_targets)
    
    # 使用binary_auprc計算PRAUC
    prauc_val = binary_auprc(all_probs, all_targets).item()
    return prauc_val

@torch.no_grad()
def generate_predictions(
    model: Module,
    loader: DataLoader,
    indices: np.ndarray,
    device,
    verbose: bool = False
) -> pd.DataFrame:
    model.eval()
    all_probs = []
    all_indices = []
    idx_counter = 0
    
    # 使用tqdm顯示進度
    progress_bar = tqdm(loader, desc="Generating predictions", disable=not verbose)
    
    for tf in progress_bar:
        tf = tf.to(device)
        pred = model(tf)
        if isinstance(model, Trompt):
            pred = pred.mean(dim=1)
        
        if pred.size(-1) == 1:
            pred = pred.view(-1)
            
        probs = torch.sigmoid(pred).cpu().numpy()
        
        batch_size = len(probs)
        batch_indices = indices[idx_counter:idx_counter+batch_size].tolist()
        idx_counter += batch_size
        
        all_probs.extend(probs.tolist())
        all_indices.extend(batch_indices)
    
    return pd.DataFrame({
        'index': all_indices,
        'prob': all_probs
    })

def train_gbdt_model(
    gbdt_model_type: str,  # 新增參數，指定使用的GBDT模型類型 ('LightGBM')
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    test_dataset,
    full_dataset,
    full_tensor_frame,
    num_trials: int,
    device,
    seed: int,
    verbose: bool = True
):
    """訓練GBDT模型(LightGBM)，完成所有的交叉驗證折"""
    print(f"\n{'='*50}")
    print(f"開始訓練 {gbdt_model_type} 模型")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    # 為該模型類型創建專屬目錄
    model_output_dir = os.path.join(OUTPUT_DIR, gbdt_model_type)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    
    # 初始化存儲每個折的結果
    all_test_predictions = []
    fold_metrics = {}
    val_fold_predictions = {}  # 格式: {index: (fold_num, prediction)}
    fold_aucs = {}  # 存儲每個fold的AUC，用於加權
    
    # 對每個fold進行訓練和評估
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        fold_num = fold_idx + 1
        print(f"\n----- 開始處理 {gbdt_model_type} 的第 {fold_num}/{len(cv_splits)} 折 -----")
        
        try:
            # 準備本折的訓練和驗證資料
            train_tensor_frame = full_tensor_frame[train_idx]
            val_tensor_frame = full_tensor_frame[val_idx]
            
            print(f"開始 {gbdt_model_type} 的超參數優化 (fold {fold_num})...")
            
            # 根據模型類型創建模型實例
            if gbdt_model_type == 'LightGBM':
                model = LightGBM(task_type=TaskType.BINARY_CLASSIFICATION, num_classes=2)
            else:
                raise ValueError(f"不支援的GBDT模型類型: {gbdt_model_type}")
            
            # 設置模型保存路徑
            model_save_path = f"{model_output_dir}/{gbdt_model_type}_fold_{fold_num}.pkl"
            
            # 使用Optuna進行超參數調整
            model.tune(
                tf_train=train_tensor_frame,
                tf_val=val_tensor_frame, 
                num_trials=num_trials
            )
            
            # 保存調整好的模型
            with open(model_save_path, 'wb') as f:
                pickle.dump(model, f)
            
            # 評估驗證集表現
            val_pred = model.predict(tf_test=val_tensor_frame)
            
            # 計算AUPRC
            val_probs = torch.tensor(val_pred)
            val_targets = val_tensor_frame.y
            val_auc = binary_auprc(val_probs, val_targets).item()
            print(f"Fold {fold_num} 驗證集 AUPRC: {val_auc:.4f}")
            
            # 保存這個折的指標和AUC，用於加權
            fold_metrics[f"fold_{fold_num}"] = {
                "val_auc": val_auc
            }
            fold_aucs[fold_num] = val_auc
            
            # 將驗證集預測和對應的fold存入字典
            for i, idx in enumerate(val_idx):
                val_fold_predictions[int(idx)] = (fold_num, float(val_pred[i]))
            
            # 生成測試集預測
            print(f"生成測試集預測...")
            test_pred = model.predict(tf_test=test_dataset.tensor_frame)
            
            # 創建與深度學習模型相同格式的預測DataFrame
            test_indices = np.arange(len(test_dataset.tensor_frame))
            test_predictions = pd.DataFrame({
                'index': test_indices,
                f"{gbdt_model_type}_fold_{fold_num}": test_pred,
                f"auc_{fold_num}": val_auc
            })
            all_test_predictions.append(test_predictions)
            
            # 保存模型詳細資訊
            model_info = {
                'model_type': gbdt_model_type,
                'fold_number': fold_num,
                'best_auc': val_auc,
                'model_path': model_save_path,
                'training_completed_timestamp': time.time()
            }
            
            model_info_path = f"{model_output_dir}/model_info_fold_{fold_num}.json"
            with open(model_info_path, 'w') as f:
                json.dump(model_info, f, indent=2)
                
            # 清理記憶體
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error training {gbdt_model_type} for fold {fold_num}: {str(e)}")
            # 將錯誤資訊添加到指標中
            fold_metrics[f"fold_{fold_num}"] = {
                "error": str(e)
            }
            
            # 為出錯的fold填充默認預測
            for idx in val_idx:
                val_fold_predictions[int(idx)] = (fold_num, 0.5)
            
            # 測試集預測保持不變，但設置AUC為0（這樣加權時會被忽略）
            test_indices = np.arange(len(test_dataset.tensor_frame)).tolist()
            test_predictions = pd.DataFrame({
                'index': test_indices,
                f"{gbdt_model_type}_fold_{fold_num}": [0.5] * len(test_indices),
                f"auc_{fold_num}": [0.0] * len(test_indices)
            })
            all_test_predictions.append(test_predictions)
    
    # 合併所有折的驗證集預測
    if val_fold_predictions:
        # 創建按索引排序的DataFrame
        sorted_indices = sorted(val_fold_predictions.keys())
        merged_val_df = pd.DataFrame({
            'index': sorted_indices,
            'fold': [val_fold_predictions[idx][0] for idx in sorted_indices],
            'prediction': [val_fold_predictions[idx][1] for idx in sorted_indices]
        })
    
    # 合併所有折的測試集預測並使用AUC加權
    if all_test_predictions:
        merged_test_df = all_test_predictions[0]
        for df in all_test_predictions[1:]:
            merged_test_df = pd.merge(merged_test_df, df, on='index', how='outer')
        
        # 計算AUC加權平均，作為最終預測
        pred_cols = [col for col in merged_test_df.columns if col.startswith(f"{gbdt_model_type}_fold_")]
        auc_cols = [col for col in merged_test_df.columns if col.startswith("auc_")]
        
        # 創建加權矩陣
        weights = np.array([merged_test_df[auc_col].values for auc_col in auc_cols]).T
        predictions = np.array([merged_test_df[pred_col].values for pred_col in pred_cols]).T
        
        # 處理可能的0權重（避免除以0）
        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除以0
        
        # 計算加權平均
        weighted_preds = (predictions * weights).sum(axis=1) / row_sums.squeeze()
        merged_test_df[gbdt_model_type] = weighted_preds
        
        # 計算平均AUC，用於最終ensemble
        model_mean_auc = np.mean([fold_aucs.get(fold, 0) for fold in range(1, len(cv_splits) + 1)])
        merged_test_df[f"{gbdt_model_type}_mean_auc"] = model_mean_auc
    
    # 返回合併後的測試集預測和驗證集預測字典以及平均AUC
    return merged_test_df, val_fold_predictions, model_mean_auc

def train_deep_learning_model(
    model_type: str,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    test_dataset,
    full_dataset,
    full_tensor_frame,
    dl_epochs: int,
    dl_num_trials: int,
    device,
    seed: int,
    verbose: bool = True
):
    """訓練指定類型的深度學習模型，完成所有的交叉驗證折"""
    print(f"\n{'='*50}")
    print(f"開始訓練 {model_type} 模型")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    # 為該模型類型創建專屬目錄
    model_output_dir = os.path.join(OUTPUT_DIR, model_type)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    
    # 獲取模型配置
    config = get_model_config(model_type, full_dataset, full_tensor_frame)
    model_cls = config['model_cls']
    col_stats = config['col_stats']
    tensor_frame = config['tensor_frame']
    model_search_space = config['model_search_space']
    train_search_space = config['train_search_space']
    transform_objects = config.get('transform_objects', {})
    
    # 初始化存儲每個折的結果
    all_test_predictions = []
    fold_metrics = {}
    val_fold_predictions = {}  # 格式: {index: (fold_num, prediction)}
    fold_aucs = {}  # 存儲每個fold的AUC，用於加權
    
    # 為測試集準備張量框架
    test_tensor_frame = test_dataset.tensor_frame
    
    # 特殊處理ExcelFormer測試集轉換
    if model_type == 'ExcelFormer' and transform_objects:
        categorical_transform = transform_objects.get('categorical_transform')
        mutual_info_sort = transform_objects.get('mutual_info_sort')
        
        if categorical_transform and mutual_info_sort:
            print(f"應用ExcelFormer的轉換到測試集...")
            transformed_test_frame = categorical_transform(test_tensor_frame)
            transformed_test_frame = mutual_info_sort(transformed_test_frame)
            test_tensor_frame = transformed_test_frame
    
    # 對每個fold進行訓練和評估
    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        fold_num = fold_idx + 1
        print(f"\n----- 開始處理 {model_type} 的第 {fold_num}/{len(cv_splits)} 折 -----")
        
        try:
            # 準備本折的訓練和驗證資料
            train_tensor_frame = tensor_frame[train_idx]
            val_tensor_frame = tensor_frame[val_idx]
            
            # 二元分類設置
            out_channels = 1
            
            print(f"開始 {model_type} 的超參數優化 (fold {fold_num})...")
            
            # 創建Optuna研究，添加模型類型和折疊編號到研究名稱
            study_name = f"{model_type}_fold_{fold_num}_study"
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=seed),
                pruner=optuna.pruners.MedianPruner(),
                study_name=study_name
            )
            
            def objective(trial):
                # 選擇模型配置
                model_cfg = {}
                for name, search_list in model_search_space.items():
                    model_cfg[name] = trial.suggest_categorical(name, search_list)
                
                # 選擇訓練配置
                train_cfg = {}
                for name, search_list in train_search_space.items():
                    train_cfg[name] = trial.suggest_categorical(name, search_list)
                
                # 為嚴重不平衡優化Focal Loss參數
                focal_alpha = trial.suggest_float('focal_alpha', 0.7, 0.9)
                focal_gamma = trial.suggest_float('focal_gamma', 1.0, 5.0)
                
                # 創建自定義的Focal Loss
                custom_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
                
                # 特殊處理FTTransformerBucket
                if model_type == 'FTTransformerBucket':
                    stype_encoder_dict = {
                        stype.categorical: EmbeddingEncoder(),
                        stype.numerical: LinearBucketEncoder(),
                    }
                    model_cfg['stype_encoder_dict'] = stype_encoder_dict
                
                col_names_dict = tensor_frame.col_names_dict
                model = model_cls(
                    **model_cfg,
                    out_channels=out_channels,
                    col_stats=col_stats,
                    col_names_dict=col_names_dict,
                ).to(device)
                model.reset_parameters()
                
                # 訓練模型
                optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg['base_lr'])
                lr_scheduler = ExponentialLR(optimizer, gamma=train_cfg['gamma_rate'])
                
                batch_size = train_cfg['batch_size']
                
                train_loader = DataLoader(train_tensor_frame,
                                        batch_size=batch_size, shuffle=True,
                                        drop_last=True)
                val_loader = DataLoader(val_tensor_frame,
                                        batch_size=batch_size)
                
                best_auc = 0
                patience = 10  # 設置早停耐心值
                counter = 0    # 初始化計數器
                model_save_path = f"{model_output_dir}/{model_type}_trial_{trial.number}_fold_{fold_num}.pt"
                
                for epoch in range(1, dl_epochs + 1):
                    # 訓練一個epoch
                    train_loss = train(model, train_loader, optimizer, epoch, device,
                        loss_function=custom_loss_fn, out_channels=out_channels, 
                        verbose=False)  # 關閉詳細輸出
                    
                    # 計算驗證集AUC
                    auc_val = evaluate(model, val_loader, device, out_channels)
                    
                    # 只在超參數搜索時輸出簡單摘要
                    if verbose and (epoch % (dl_epochs // min(5, dl_epochs)) == 0 or epoch == dl_epochs):
                        print(f"  Trial {trial.number} Epoch {epoch}/{dl_epochs}: Loss={train_loss:.4f}, AUC={auc_val:.4f}")
                    
                    # 檢查是否有改善
                    if auc_val > best_auc:
                        best_auc = auc_val
                        counter = 0  # 重置計數器
                        
                        # 保存當前最佳模型，增加模型類型識別
                        torch.save({
                            'model_type': model_type,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'trial_number': trial.number,
                            'fold_number': fold_num,
                            'epoch': epoch,
                            'model_cfg': model_cfg,
                            'train_cfg': train_cfg,
                            'focal_alpha': focal_alpha,
                            'focal_gamma': focal_gamma,
                            'auc_val': auc_val,
                            'timestamp': time.time()
                        }, model_save_path)
                        
                        trial.set_user_attr('best_model_path', model_save_path)
                    else:
                        counter += 1  # 增加計數器
                    
                    # 早期停止檢查
                    if counter >= patience:
                        if verbose:
                            print(f"  Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs)")
                        break
                    
                    lr_scheduler.step()
                    trial.report(auc_val, epoch)
                    
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                
                # 清理GPU記憶體
                del model
                clean_gpu_memory()
                
                # 保存找到的最佳Focal Loss參數和模型路徑
                trial.set_user_attr('best_focal_alpha', focal_alpha)
                trial.set_user_attr('best_focal_gamma', focal_gamma)
                
                return best_auc
            
            # 運行Optuna優化
            study.optimize(objective, n_trials=dl_num_trials, n_jobs=1)
            
            # 獲取最佳trial
            best_trial = study.best_trial
            best_auc = best_trial.value
            
            print(f"Fold {fold_num} 的最佳驗證AUC: {best_auc:.4f}")
            print(f"最佳參數: {best_trial.params}")
            
            # 獲取最佳模型路徑
            best_model_path = best_trial.user_attrs.get('best_model_path')
            
            if best_model_path and os.path.exists(best_model_path):
                print(f"載入最佳模型 (Trial {best_trial.number})...")
                
                # 載入模型
                checkpoint = torch.load(best_model_path)
                
                # 檢查模型類型
                if checkpoint.get('model_type') != model_type:
                    print(f"警告: 檔案中的模型類型 ({checkpoint.get('model_type')}) 與當前模型類型 ({model_type}) 不匹配!")
                
                # 獲取模型配置和參數
                best_model_cfg = checkpoint['model_cfg']
                best_train_cfg = checkpoint['train_cfg']
                best_focal_alpha = checkpoint['focal_alpha']
                best_focal_gamma = checkpoint['focal_gamma']
                final_val_auc = checkpoint['auc_val']
                
                # 特殊處理FTTransformerBucket
                if model_type == 'FTTransformerBucket':
                    stype_encoder_dict = {
                        stype.categorical: EmbeddingEncoder(),
                        stype.numerical: LinearBucketEncoder(),
                    }
                    best_model_cfg['stype_encoder_dict'] = stype_encoder_dict
                
                # 創建模型
                col_names_dict = tensor_frame.col_names_dict
                model = model_cls(
                    **best_model_cfg,
                    out_channels=out_channels,
                    col_stats=col_stats,
                    col_names_dict=col_names_dict,
                ).to(device)
                
                # 載入保存的權重
                model.load_state_dict(checkpoint['model_state_dict'])
                
                print(f"成功載入最佳模型，驗證AUC: {final_val_auc:.4f}")
                
                # 保存這個折的指標和AUC，用於加權
                fold_metrics[f"fold_{fold_num}"] = {
                    "val_auc": final_val_auc
                }
                fold_aucs[fold_num] = final_val_auc
                
                # 清理其他trial的模型檔案
                trial_models = [f for f in os.listdir(model_output_dir) 
                              if f.startswith(f"{model_type}_trial_") and f.endswith(f"_fold_{fold_num}.pt")]
                for model_file in trial_models:
                    file_path = os.path.join(model_output_dir, model_file)
                    if file_path != best_model_path and os.path.exists(file_path):
                        os.remove(file_path)
                print(f"已清理 {model_type} 模型其他trial的檔案")
                
                # 保存模型詳細資訊
                model_info = {
                    'model_type': model_type,
                    'fold_number': fold_num,
                    'best_trial_number': best_trial.number,
                    'best_params': best_trial.params,
                    'best_auc': final_val_auc,
                    'train_config': best_train_cfg,
                    'model_config': best_model_cfg,
                    'focal_loss_params': {
                        'alpha': best_focal_alpha,
                        'gamma': best_focal_gamma
                    },
                    'model_path': best_model_path,
                    'training_completed_timestamp': time.time()
                }
                
                model_info_path = f"{model_output_dir}/model_info_fold_{fold_num}.json"
                with open(model_info_path, 'w') as f:
                    json.dump(model_info, f, indent=2)
                
                # 生成驗證集預測
                print(f"生成驗證集預測...")
                val_loader = DataLoader(val_tensor_frame, batch_size=best_train_cfg['batch_size'])
                val_predictions = generate_predictions(model, val_loader, val_idx, device, verbose)
                # 將預測結果和對應的fold存入字典
                for _, row in val_predictions.iterrows():
                    val_fold_predictions[int(row['index'])] = (fold_num, float(row['prob']))
                
                # 生成測試集預測
                print(f"生成測試集預測...")
                test_loader = DataLoader(test_tensor_frame, batch_size=best_train_cfg['batch_size'])
                test_indices = np.arange(len(test_tensor_frame))
                test_predictions = generate_predictions(model, test_loader, test_indices, device, verbose)
                # 添加fold的AUC作為列
                test_predictions[f"auc_{fold_num}"] = final_val_auc
                test_predictions.rename(columns={'prob': f"{model_type}_fold_{fold_num}"}, inplace=True)
                all_test_predictions.append(test_predictions)
                
                # 清理記憶體
                del model
                clean_gpu_memory()
            else:
                print(f"警告: 找不到保存的最佳模型，將使用預設預測...")
                # 為出錯的fold填充默認預測
                for idx in val_idx:
                    val_fold_predictions[int(idx)] = (fold_num, 0.5)
                
                # 測試集預測保持不變，但設置AUC為0（這樣加權時會被忽略）
                test_indices = np.arange(len(test_tensor_frame)).tolist()
                test_predictions = pd.DataFrame({
                    'index': test_indices,
                    f"{model_type}_fold_{fold_num}": [0.5] * len(test_indices),
                    f"auc_{fold_num}": [0.0] * len(test_indices)
                })
                all_test_predictions.append(test_predictions)
                
                # 更新折的指標
                fold_metrics[f"fold_{fold_num}"] = {
                    "error": "模型檔案未找到"
                }
            
        except Exception as e:
            print(f"Error training {model_type} for fold {fold_num}: {str(e)}")
            # 將錯誤資訊添加到指標中
            fold_metrics[f"fold_{fold_num}"] = {
                "error": str(e)
            }
            
            # 為出錯的fold填充默認預測
            for idx in val_idx:
                val_fold_predictions[int(idx)] = (fold_num, 0.5)
            
            # 測試集預測保持不變，但設置AUC為0（這樣加權時會被忽略）
            test_indices = np.arange(len(test_tensor_frame)).tolist()
            test_predictions = pd.DataFrame({
                'index': test_indices,
                f"{model_type}_fold_{fold_num}": [0.5] * len(test_indices),
                f"auc_{fold_num}": [0.0] * len(test_indices)
            })
            all_test_predictions.append(test_predictions)
    
    # 合併所有折的驗證集預測，按照索引順序排列
    if val_fold_predictions:
        # 創建按索引排序的DataFrame
        sorted_indices = sorted(val_fold_predictions.keys())
        merged_val_df = pd.DataFrame({
            'index': sorted_indices,
            'fold': [val_fold_predictions[idx][0] for idx in sorted_indices],
            'prediction': [val_fold_predictions[idx][1] for idx in sorted_indices]
        })
    
    # 合併所有折的測試集預測並使用AUC加權
    if all_test_predictions:
        merged_test_df = all_test_predictions[0]
        for df in all_test_predictions[1:]:
            merged_test_df = pd.merge(merged_test_df, df, on='index', how='outer')
        
        # 計算AUC加權平均，作為最終預測
        pred_cols = [col for col in merged_test_df.columns if col.startswith(f"{model_type}_fold_")]
        auc_cols = [col for col in merged_test_df.columns if col.startswith("auc_")]
        
        # 創建加權矩陣
        weights = np.array([merged_test_df[auc_col].values for auc_col in auc_cols]).T
        predictions = np.array([merged_test_df[pred_col].values for pred_col in pred_cols]).T
        
        # 處理可能的0權重（避免除以0）
        row_sums = weights.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # 避免除以0
        
        # 計算加權平均
        weighted_preds = (predictions * weights).sum(axis=1) / row_sums.squeeze()
        merged_test_df[model_type] = weighted_preds
        
        # 計算平均AUC，用於最終ensemble
        model_mean_auc = np.mean([fold_aucs.get(fold, 0) for fold in range(1, len(cv_splits) + 1)])
        merged_test_df[f"{model_type}_mean_auc"] = model_mean_auc
    
    # 返回合併後的測試集預測和驗證集預測字典
    return merged_test_df, val_fold_predictions, model_mean_auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task_type', type=str, choices=[
            'binary_classification'
        ], default='binary_classification')
    parser.add_argument('--dl_epochs', type=int, default=30,
                        help='每個模型的訓練輪數')
    parser.add_argument('--dl_num_trials', type=int, default=5,
                        help='深度學習模型的超參數搜索次數')
    parser.add_argument('--lgbm_num_trials', type=int, default=20,
                        help='GBDT模型的超參數搜索次數')
    parser.add_argument('--seed', type=int, default=0,
                        help='隨機種子')
    parser.add_argument('--model_types', type=str, nargs='+', default=DL_MODEL_TYPES,
                        help='要訓練的模型類型，默認全部')
    parser.add_argument('--verbose', action='store_true',
                        help='是否輸出詳細訓練過程')
    args = parser.parse_args()

    # 設置設備和種子
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 固定任務類型為二元分類
    args.task_type = 'binary_classification'
    
    # 確保model_types有效
    model_types = []
    for model_type in args.model_types:
        if model_type in DL_MODEL_TYPES:
            model_types.append(model_type)
        else:
            print(f"警告: 未知的模型類型 '{model_type}'，將被忽略")
    
    if not model_types:
        print("錯誤: 沒有有效的模型類型被指定")
        return
    
    print(f"將訓練以下模型: {', '.join(model_types)}")

    # 載入訓練集和測試集資料
    print("\n載入訓練集...")
    df = pd.read_csv('final_meta_train.csv')
    
    # 保存id列供參考（如果存在）
    if 'id' in df.columns:
        train_ids = df['id'].copy()
        df = df.drop('id', axis=1)
    
    # 載入特徵類型映射
    print("載入特徵類型映射...")
    with open('feature_types.json', 'r') as f:
        loaded_col_to_stype = json.load(f)

    # 將字串轉換回stype
    def convert_to_stype(type_str):
        if type_str == 'numerical':
            return stype.numerical
        elif type_str == 'categorical':
            return stype.categorical
        else:
            raise ValueError(f"未知類型: {type_str}")

    # 將載入的字典轉換回原始格式
    col_to_stype = {k: convert_to_stype(v) for k, v in loaded_col_to_stype.items()}

    # 創建訓練集Dataset
    print("創建訓練集Dataset...")
    dataset = Dataset(df=df, col_to_stype=col_to_stype, target_col='target')
    dataset.materialize()
    
    # 設置交叉驗證
    full_dataset = dataset
    full_tensor_frame = full_dataset.tensor_frame
    
    # 獲取標籤用於分層抽樣
    y_array = full_tensor_frame.y.cpu().numpy()

    # 使用分層抽樣的KFold，保證每折中正負樣本比例一致
    print(f"準備 {NUM_CV_FOLDS} 折交叉驗證...")
    skf = StratifiedKFold(n_splits=NUM_CV_FOLDS, shuffle=True, random_state=args.seed)
    # 儲存拆分結果
    cv_splits = list(skf.split(np.arange(len(full_tensor_frame)), y_array))
    
    # 載入測試集
    test_dataset, test_ids = load_test_dataset(col_to_stype)
    
    # 用於整合所有模型的測試集預測和驗證集預測
    all_models_test_predictions = []
    all_models_val_predictions = {}  # 格式: {index: {model_type: prediction, 'fold': fold_num}}
    model_aucs = {}  # 存儲每個模型的平均AUC
    
    # 統計數據
    total_start_time = time.time()
    model_times = {}
    model_metrics = {}
    
    # 按照模型類型逐個訓練所有折
    for model_type in model_types:
        model_start_time = time.time()
        try:
            # 特殊處理GBDT類型模型 (LightGBM)
            if model_type in ['LightGBM']:
                # 訓練GBDT模型
                test_predictions, val_predictions_dict, mean_auc = train_gbdt_model(
                    gbdt_model_type=model_type,  # 傳遞具體的GBDT模型類型
                    cv_splits=cv_splits, 
                    test_dataset=test_dataset, 
                    full_dataset=full_dataset, 
                    full_tensor_frame=full_tensor_frame, 
                    num_trials=args.lgbm_num_trials, 
                    device=device, 
                    seed=args.seed, 
                    verbose=args.verbose
                )
            else:
                # 訓練深度學習模型（原有代碼）
                test_predictions, val_predictions_dict, mean_auc = train_deep_learning_model(
                    model_type=model_type, 
                    cv_splits=cv_splits, 
                    test_dataset=test_dataset, 
                    full_dataset=full_dataset, 
                    full_tensor_frame=full_tensor_frame, 
                    dl_epochs=args.dl_epochs, 
                    dl_num_trials=args.dl_num_trials, 
                    device=device, 
                    seed=args.seed, 
                    verbose=args.verbose
                )
            
            # 記錄模型平均AUC
            model_aucs[model_type] = mean_auc
            
            # 保存這個模型的測試集預測
            all_models_test_predictions.append(test_predictions[['index', model_type]])
            
            # 合併這個模型的驗證集預測到全局字典
            for idx, (fold, pred) in val_predictions_dict.items():
                if idx not in all_models_val_predictions:
                    all_models_val_predictions[idx] = {'fold': fold}
                all_models_val_predictions[idx][model_type] = pred
            
            # 記錄訓練時間
            model_end_time = time.time()
            model_times[model_type] = model_end_time - model_start_time
            
            # 載入並記錄模型指標
            model_metrics[model_type] = {
                'mean_val_auc': mean_auc
            }
                        
        except Exception as e:
            print(f"訓練 {model_type} 時發生錯誤: {str(e)}")
            model_end_time = time.time()
            model_times[model_type] = model_end_time - model_start_time
            model_metrics[model_type] = {"error": str(e)}
            model_aucs[model_type] = 0.0  # 設置失敗模型的AUC為0
        
        # 在每個模型訓練完成後保存當前的中間結果
        
        # 1. 保存當前的驗證集預測
        if all_models_val_predictions:
            print(f"\n保存當前驗證集預測 (包含 {model_type})...")
            current_val_data = []
            for idx in sorted(all_models_val_predictions.keys()):
                row_data = {'index': idx, 'fold': all_models_val_predictions[idx]['fold']}
                for mt in model_types:
                    if mt in all_models_val_predictions[idx]:
                        row_data[mt] = all_models_val_predictions[idx][mt]
                current_val_data.append(row_data)
            
            # 創建驗證集DataFrame
            current_val_df = pd.DataFrame(current_val_data)
            
            # 計算驗證集上的ensemble預測 (如果有多個模型)
            valid_model_types = [mt for mt in model_types if mt in model_aucs and model_aucs[mt] > 0 and mt in current_val_df.columns]
            
            if len(valid_model_types) > 1:
                # 獲取有效模型權重
                weights = np.array([model_aucs[mt] for mt in valid_model_types])
                if np.sum(weights) > 0:
                    normalized_weights = weights / np.sum(weights)
                    
                    # 計算驗證集ensemble
                    current_val_df['ensemble'] = 0
                    for i, mt in enumerate(valid_model_types):
                        current_val_df['ensemble'] += current_val_df[mt].values * normalized_weights[i]
            elif len(valid_model_types) == 1:
                # 如果只有一個有效模型，其預測就是ensemble預測
                current_val_df['ensemble'] = current_val_df[valid_model_types[0]]
            
            # 添加真實的target值到驗證集
            current_val_df['target'] = current_val_df['index'].apply(lambda idx: full_tensor_frame.y[int(idx)].item())

            # 保存當前驗證集預測（包含ensemble）
            current_val_df.to_csv(COMBINED_VAL_FILE, index=False)
            print(f"當前驗證集預測已保存至 {COMBINED_VAL_FILE}")
        
        # 2. 保存當前的測試集預測
        if all_models_test_predictions:
            print(f"\n保存當前測試集預測 (包含 {model_type})...")
            # 合併目前為止所有模型的預測
            current_test_df = all_models_test_predictions[0].copy()
            for df in all_models_test_predictions[1:]:
                current_test_df = pd.merge(current_test_df, df, on='index', how='outer')
            
            # 計算目前為止的ensemble (如果有多個模型)
            valid_model_types = [mt for mt in model_types if mt in model_aucs and model_aucs[mt] > 0 and mt in current_test_df.columns]
            
            if len(valid_model_types) > 1:
                # 獲取有效模型權重
                weights = np.array([model_aucs[mt] for mt in valid_model_types])
                if np.sum(weights) > 0:
                    normalized_weights = weights / np.sum(weights)
                    
                    # 計算ensemble
                    current_test_df_with_ensemble = current_test_df.copy()
                    weighted_preds = np.zeros(len(current_test_df_with_ensemble))
                    for i, mt in enumerate(valid_model_types):
                        weighted_preds += current_test_df_with_ensemble[mt].values * normalized_weights[i]
                    
                    current_test_df_with_ensemble['ensemble'] = weighted_preds
                    
                    # 添加ID
                    current_test_df_with_ensemble['id'] = test_ids.values[current_test_df_with_ensemble['index']].astype(int)
                    
                    # 重新排列列順序
                    cols = ['id'] + [col for col in current_test_df_with_ensemble.columns if col not in ['index', 'id']]
                    current_test_df_with_ensemble = current_test_df_with_ensemble[cols]
                    
                    # 保存
                    current_test_df_with_ensemble.to_csv(FINAL_TEST_FILE, index=False)
                    print(f"當前測試集預測已保存至 {FINAL_TEST_FILE}")
            elif len(valid_model_types) == 1:
                # 如果只有一個有效模型
                current_test_df_with_ensemble = current_test_df.copy()
                current_test_df_with_ensemble['ensemble'] = current_test_df_with_ensemble[valid_model_types[0]]
                
                # 添加ID
                current_test_df_with_ensemble['id'] = test_ids.values[current_test_df_with_ensemble['index']].astype(int)
                
                # 重新排列列順序
                cols = ['id'] + [col for col in current_test_df_with_ensemble.columns if col not in ['index', 'id']]
                current_test_df_with_ensemble = current_test_df_with_ensemble[cols]
                
                # 保存
                current_test_df_with_ensemble.to_csv(FINAL_TEST_FILE, index=False)
                print(f"當前測試集預測已保存至 {FINAL_TEST_FILE}")
        
        # 3. 保存當前的模型指標
        print(f"\n保存當前模型指標 (包含 {model_type})...")
        current_metrics = {
            'model_metrics': {},
            'model_weights': {},
            'training_progress': {
                'completed_models': model_types.index(model_type) + 1,
                'total_models': len(model_types),
                'elapsed_time_seconds': time.time() - total_start_time
            }
        }
        
        # 添加每個已處理模型的指標
        for mt in model_types:
            if mt in model_aucs:
                current_metrics['model_metrics'][mt] = {
                    'mean_val_auc': model_aucs[mt]
                }
        
        # 計算模型權重 (如果有多個有效模型)
        valid_model_types = [mt for mt in model_types if mt in model_aucs and model_aucs[mt] > 0]
        if len(valid_model_types) > 1:
            weights = np.array([model_aucs[mt] for mt in valid_model_types])
            if np.sum(weights) > 0:
                normalized_weights = weights / np.sum(weights)
                for i, mt in enumerate(valid_model_types):
                    current_metrics['model_weights'][mt] = float(normalized_weights[i])
        
        # 保存
        with open(COMBINED_METRICS_FILE, 'w') as f:
            json.dump(current_metrics, f, indent=2)
        print(f"當前模型指標已保存至 {COMBINED_METRICS_FILE}")
        
        # 顯示當前進度摘要
        elapsed_time = time.time() - total_start_time
        completed = model_types.index(model_type) + 1
        remaining = len(model_types) - completed
        
        print(f"\n進度摘要: 已完成 {completed}/{len(model_types)} 個模型")
        print(f"已用時間: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分鐘)")
        
        if remaining > 0 and completed > 0:
            avg_time_per_model = elapsed_time / completed
            est_remaining_time = avg_time_per_model * remaining
            print(f"預估剩餘時間: {est_remaining_time:.2f} 秒 ({est_remaining_time/60:.2f} 分鐘)")
    
    # 合併所有模型的驗證集預測到一個CSV文件 (最終版本)
    if all_models_val_predictions:
        print("\n合併所有模型的驗證集預測...")
        # 按照索引排序
        sorted_indices = sorted(all_models_val_predictions.keys())
        
        # 構建DataFrame
        val_data = []
        for idx in sorted_indices:
            row_data = {'index': idx, 'fold': all_models_val_predictions[idx]['fold']}
            for model_type in model_types:
                if model_type in all_models_val_predictions[idx]:
                    row_data[model_type] = all_models_val_predictions[idx][model_type]
            val_data.append(row_data)
        
        combined_val_df = pd.DataFrame(val_data)
        
        # 計算最終的驗證集ensemble預測
        valid_model_types = [mt for mt in model_types if mt in model_aucs and model_aucs[mt] > 0 and mt in combined_val_df.columns]
        
        if len(valid_model_types) > 1:
            # 獲取有效模型權重
            weights = np.array([model_aucs[mt] for mt in valid_model_types])
            if np.sum(weights) > 0:
                normalized_weights = weights / np.sum(weights)
                
                # 打印模型權重，便於記錄
                print("\n驗證集ensemble權重:")
                for i, mt in enumerate(valid_model_types):
                    print(f"  {mt}: {normalized_weights[i]:.4f}")
                
                # 計算加權預測
                combined_val_df['ensemble'] = 0
                for i, mt in enumerate(valid_model_types):
                    combined_val_df['ensemble'] += combined_val_df[mt].values * normalized_weights[i]
            else:
                # 如果所有模型AUC都為0，使用簡單平均
                print("警告: 所有模型AUC為0，驗證集使用簡單平均")
                combined_val_df['ensemble'] = combined_val_df[valid_model_types].mean(axis=1)
        elif len(valid_model_types) == 1:
            # 如果只有一個有效模型
            combined_val_df['ensemble'] = combined_val_df[valid_model_types[0]]
        
        # 添加真實的target值到驗證集
        current_val_df['target'] = current_val_df['index'].apply(lambda idx: full_tensor_frame.y[int(idx)].item())

        # 保存合併後的驗證集預測
        combined_val_df.to_csv(COMBINED_VAL_FILE, index=False)
        print(f"所有模型的驗證集預測已合併保存至 {COMBINED_VAL_FILE}")
    
    # 合併所有模型的測試集預測，使用模型AUC加權 (最終版本)
    if all_models_test_predictions:
        print("\n合併所有模型的測試集預測...")
        final_test_df = all_models_test_predictions[0]
        for df in all_models_test_predictions[1:]:
            final_test_df = pd.merge(final_test_df, df, on='index', how='outer')
        
        # 計算所有模型的AUC加權平均作為最終整合結果
        if len(model_types) > 1:
            # 獲取有效模型和對應的AUC
            valid_model_types = [mt for mt in model_types if mt in final_test_df.columns and model_aucs.get(mt, 0) > 0]
            
            if valid_model_types:
                # 獲取模型權重
                model_weights = np.array([model_aucs.get(mt, 0) for mt in valid_model_types])
                # 標準化權重
                if np.sum(model_weights) > 0:
                    model_weights = model_weights / np.sum(model_weights)
                    
                    # 打印模型權重，便於記錄
                    print("\n模型ensemble權重:")
                    for i, mt in enumerate(valid_model_types):
                        print(f"  {mt}: {model_weights[i]:.4f}")
                    
                    # 計算加權預測
                    final_test_df_final = final_test_df.copy()  # 創建副本避免警告
                    weighted_preds = np.zeros(len(final_test_df_final))
                    for i, mt in enumerate(valid_model_types):
                        weighted_preds += final_test_df_final[mt].values * model_weights[i]
                    
                    final_test_df_final['ensemble'] = weighted_preds
                else:
                    # 如果所有模型AUC都為0，使用簡單平均
                    print("警告: 所有模型AUC為0，使用簡單平均")
                    final_test_df_final = final_test_df.copy()
                    final_test_df_final['ensemble'] = final_test_df_final[valid_model_types].mean(axis=1)
            else:
                final_test_df_final = final_test_df.copy()
                print("警告: 沒有有效的模型進行ensemble")
        else:
            final_test_df_final = final_test_df.copy()
            if len(model_types) == 1 and model_types[0] in final_test_df_final.columns:
                final_test_df_final['ensemble'] = final_test_df_final[model_types[0]]
        
        # 添加原始ID
        final_test_df_final['id'] = test_ids.values[final_test_df_final['index']].astype(int)
        
        # 移除index欄位
        final_columns = ['id'] + [col for col in final_test_df_final.columns 
                                if col not in ['index', 'id']]
        final_test_df_final = final_test_df_final[final_columns]
        
        # 保存最終的測試集預測
        final_test_df_final.to_csv(FINAL_TEST_FILE, index=False)
        print(f"最終整合的測試集預測已保存至 {FINAL_TEST_FILE}")
    else:
        print("沒有生成任何測試集預測!")
    
    # 合併所有模型的指標到一個文件 (最終版本)
    print("\n整合所有模型的指標...")
    combined_metrics = {
        'model_metrics': {},
        'model_weights': {},
        'validation_metrics': {}  # 新增：包含驗證集上的整體評估
    }

    # 添加每個模型的指標
    for model_type in model_types:
        # 添加模型的平均AUC
        if model_type in model_aucs:
            combined_metrics['model_metrics'][model_type] = {
                'mean_val_auc': model_aucs[model_type]
            }
    
    # 如果有進行ensemble，添加模型權重
    if 'valid_model_types' in locals() and 'model_weights' in locals() and len(valid_model_types) > 1:
        for i, mt in enumerate(valid_model_types):
            combined_metrics['model_weights'][mt] = float(model_weights[i])

    # 添加ensemble的整體信息
    if 'valid_model_types' in locals() and len(valid_model_types) > 0:
        combined_metrics['ensemble'] = {
            'model_count': len(valid_model_types),
            'models_used': valid_model_types
        }
        
        # 嘗試計算驗證集上ensemble的整體表現
        if 'combined_val_df' in locals() and 'ensemble' in combined_val_df.columns:
            # 如果有原始標籤，可以計算整體的AUPRC
            # 這裡假設我們沒有原始標籤，但可以添加如果有的話
            combined_metrics['validation_metrics']['has_ensemble'] = True

    # 保存合併的指標
    with open(COMBINED_METRICS_FILE, 'w') as f:
        json.dump(combined_metrics, f, indent=2)
    print(f"所有模型的指標已整合保存至 {COMBINED_METRICS_FILE}")

    # 計算總運行時間
    total_time = time.time() - total_start_time
    
    # 輸出運行統計
    print("\n" + "="*50)
    print("訓練完成摘要")
    print("="*50)
    print(f"總運行時間: {total_time:.2f} 秒 ({total_time/60:.2f} 分鐘)")
    print("\n模型訓練時間:")
    for model_type, time_taken in model_times.items():
        print(f"  {model_type}: {time_taken:.2f} 秒 ({time_taken/60:.2f} 分鐘)")
    
    print("\n模型驗證AUC:")
    for model_type, metrics in model_metrics.items():
        if "error" in metrics:
            print(f"  {model_type}: 訓練失敗 - {metrics['error']}")
        else:
            print(f"  {model_type}: {metrics.get('mean_val_auc', 'N/A'):.4f}")

    # 如果有驗證集ensemble的結果，輸出其表現
    if 'combined_val_df' in locals() and 'ensemble' in combined_val_df.columns:
        print("\n驗證集ensemble結果已保存")
        print(f"驗證集樣本數: {len(combined_val_df)}")

if __name__ == '__main__':
    main()