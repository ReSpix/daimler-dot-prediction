import os, torch, numpy as np
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from data_prep import load_properties, merge_scenario_data, ScenarioDataset
from model import DeepSetsPredictor
import pandas as pd
from paths import TRAIN_PATH, PROPS_PATH, SAVE_DIR

def train():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Device: {DEVICE}")
    
    # 1. Загрузка и подготовка
    props_df, prop_cols = load_properties(PROPS_PATH)
    train_df, valid_props = merge_scenario_data(pd.read_csv(TRAIN_PATH), props_df, prop_cols)
    
    dataset = ScenarioDataset(train_df, valid_props, fit_scaler=True)
    print(f"📊 Data: {len(dataset)} scenarios, {len(valid_props)} properties, max {dataset.max_n} components")
    
    # 2. Кросс-валидация
    gkf = GroupKFold(n_splits=5)
    groups = dataset.scenario_ids
    best_score = np.inf
    best_state = None
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(dataset.scenario_ids, dataset.targets if dataset.targets is not None else None, groups)):
        print(f"\n🔹 Fold {fold+1}")
        tr_dl = DataLoader(torch.utils.data.Subset(dataset, tr_idx), batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
        val_dl = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=64, shuffle=False, num_workers=0)
        
        model = DeepSetsPredictor(n_props=len(valid_props)).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        # Используем только ReduceLROnPlateau для стабильности
        scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=4)
        criterion = nn.HuberLoss(delta=1.0)
        
        fold_best_loss = np.inf
        patience = 12
        patience_counter = 0

        for epoch in range(80):
            model.train()
            train_loss = 0.0
            for batch in tr_dl:
                props_t = batch['props'].to(DEVICE)
                mask_t = batch['mask'].to(DEVICE)
                conc_t = batch['conc'].to(DEVICE)
                cond_t = batch['conditions'].to(DEVICE)
                target_t = batch['target'].to(DEVICE)
                
                props_t = torch.nan_to_num(props_t, nan=0.0, posinf=1.0, neginf=-1.0)
                opt.zero_grad()
                v_pred, o_pred = model(props_t, mask_t, conc_t, cond_t)
                loss = criterion(v_pred, target_t[:,0]) + criterion(o_pred, target_t[:,1])
                if torch.isnan(loss): continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                train_loss += loss.item()

            model.eval()
            val_loss = 0.0
            v_loss, o_loss = 0.0, 0.0
            with torch.no_grad():
                for batch in val_dl:
                    v_pred, o_pred = model(batch['props'].to(DEVICE), batch['mask'].to(DEVICE), 
                                        batch['conc'].to(DEVICE), batch['conditions'].to(DEVICE))
                    vl = criterion(v_pred, batch['target'][:,0].to(DEVICE))
                    ol = criterion(o_pred, batch['target'][:,1].to(DEVICE))
                    val_loss += (vl + ol).item()
                    v_loss += vl.item()
                    o_loss += ol.item()
            
            avg_val = val_loss / max(len(val_dl), 1)
            scheduler.step(avg_val)
            current_lr = opt.param_groups[0]['lr']

            if epoch % 5 == 0 or epoch < 3:
                print(f"  Epoch {epoch:02d} | LR: {current_lr:.1e} | Train: {train_loss/len(tr_dl):.3f} | Val: {avg_val:.3f}")

            # ✅ Логика Early Stopping и сохранения лучшего фолда
            if avg_val < fold_best_loss:
                fold_best_loss = avg_val
                torch.save(model.state_dict(), SAVE_DIR / f"fold_{fold+1}_best.pt")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  🛑 Early stopping at epoch {epoch}")
                    break
        
        if fold_best_loss < best_score:
            best_score = fold_best_loss
            best_state = model.state_dict()
            print(f"  ✅ New global best! Val Loss: {fold_best_loss:.3f}")

    # 3. Финальное сохранение
    torch.save(best_state, SAVE_DIR / "best_model.pt")
    torch.save({
        'prop_scaler': dataset.prop_scaler,
        'cond_scaler': dataset.cond_scaler,
        'target_scaler': dataset.target_scaler,
        'prop_cols': valid_props
    }, SAVE_DIR / "scalers.pt")
    print(f"\n🎉 Training done. Saved to {SAVE_DIR}")

if __name__ == '__main__':
    train()