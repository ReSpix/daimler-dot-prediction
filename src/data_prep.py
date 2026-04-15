import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import re

COL_MAP = {
    'Массовая доля, %': 'conc',
    'Температура испытания | ASTM D445 Daimler Oxidation Test (DOT), °C': 'temp',
    'Время испытания | - Daimler Oxidation Test (DOT), ч': 'time',
    'Количество биотоплива | - Daimler Oxidation Test (DOT), % масс': 'biofuel',
    'Дозировка катализатора, категория': 'catalyst',
    'Delta Kin. Viscosity KV100 - relative | - Daimler Oxidation Test (DOT), %': 'target_vis',
    'Oxidation EOT | DIN 51453 Daimler Oxidation Test (DOT), A/cm': 'target_ox'
}

def clean_prop_name(name):
    if pd.isna(name) or not isinstance(name, str):
        return "unknown"
    name = re.sub(r'\s*\|.*', '', name).strip().lower()
    name = re.sub(r'[^a-z0-9_]', '_', name).strip('_')
    return name if name else "unknown"

def load_properties(props_path):
    df = pd.read_csv(props_path)
    df = df.dropna(subset=['Наименование показателя'])
    df['prop_clean'] = df['Наименование показателя'].apply(clean_prop_name)
    
    # Чистим числа: меняем запятые, убираем символы типа "<", "≈", "нет"
    clean_vals = df['Значение показателя'].astype(str).str.replace(',', '.', regex=False)
    clean_vals = clean_vals.replace(to_replace=r'^[^0-9\.\-]*$', value=np.nan, regex=True)
    df['value'] = pd.to_numeric(clean_vals, errors='coerce')
    
    pivoted = df.pivot_table(
        index=['Компонент', 'Наименование партии'], 
        columns='prop_clean', 
        values='value', 
        aggfunc='first'
    ).reset_index()
    
    num_cols = [c for c in pivoted.columns if c not in ['Компонент', 'Наименование партии']]
    return pivoted[['Компонент', 'Наименование партии'] + num_cols], num_cols

def merge_scenario_data(mix_df, props_df, prop_cols):
    mix_df = mix_df.rename(columns=COL_MAP)
    mix_df['conc'] = pd.to_numeric(mix_df['conc'].astype(str).str.replace(',', '.'), errors='coerce')
    
    merged = mix_df.merge(props_df, on=['Компонент', 'Наименование партии'], how='left')
    merged = merged.drop(columns=['Наименование партии'], errors='ignore')
    
    valid_props = [c for c in prop_cols if c in merged.columns]
    return merged, valid_props

class SafeScaler:
    """Ручной скейлер, устойчивый к NaN и нулевой дисперсии"""
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, x, mask=None):
        # x может быть (B, N, P) или (B, C)
        if mask is not None and x.ndim > 2:
            mask = mask[..., np.newaxis]  # (B, N, 1) для broadcast с (B, N, P)
            
        valid = x if mask is None else np.where(mask, x, np.nan)
        axes = tuple(range(valid.ndim - 1)) # считаем по всем осям кроме последней (признаки)
        
        self.mean = np.nanmean(valid, axis=axes)
        self.std = np.nanstd(valid, axis=axes)
        
        self.mean = np.nan_to_num(self.mean, nan=0.0)
        self.std = np.nan_to_num(self.std, nan=1.0)
        self.std[self.std < 1e-6] = 1.0
        
    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, y):
        """
        Обратное преобразование.
        Если transform: y = (x - mean) / std,
        то inverse: x = y * std + mean
        """
        return y * self.std + self.mean

class ScenarioDataset(Dataset):
    def __init__(self, df, prop_cols, fit_scaler=False):
        self.df = df
        self.prop_cols = prop_cols
        self.scenario_ids = df['scenario_id'].unique().tolist()
        self.groups = df.groupby('scenario_id')
        self.max_n = int(self.groups.size().max())
        n_scenarios = len(self.scenario_ids)
        n_props = len(prop_cols)

        self.props = np.full((n_scenarios, self.max_n, n_props), np.nan, dtype=np.float32)
        self.conc = np.zeros((n_scenarios, self.max_n), dtype=np.float32)
        self.conditions = np.zeros((n_scenarios, 4), dtype=np.float32)
        self.mask = np.zeros((n_scenarios, self.max_n), dtype=bool)

        self.type_to_id = {"Базовое_масло": 0, "Детергент": 1, "Антиоксидант": 2, "Противоизносная_присадка": 3, 
                       "Дисперсант": 4, "Загуститель": 5, "Депрессорная_присадка": 6, "Антипенная_присадка": 7, "Соединение_молибдена": 8}
        self.type_ids = np.zeros((n_scenarios, self.max_n), dtype=np.int64)
        
        for i, sid in enumerate(self.scenario_ids):
            grp = self.groups.get_group(sid)
            n = len(grp)
            for j, comp_name in enumerate(grp['Компонент'].values):
                comp_type = next((k for k in self.type_to_id if comp_name.startswith(k)), -1)
                self.type_ids[i, j] = self.type_to_id.get(str(comp_type), 9)  # 9 = unknown

        has_targets = 'target_vis' in df.columns
        self.targets = np.zeros((n_scenarios, 2), dtype=np.float32) if has_targets else None

        for i, sid in enumerate(self.scenario_ids):
            grp = self.groups.get_group(sid)
            n = len(grp)
            self.mask[i, :n] = True
            self.props[i, :n, :] = grp[prop_cols].values
            self.conc[i, :n] = grp['conc'].values
            self.conditions[i, 0] = grp['temp'].iloc[0]
            self.conditions[i, 1] = grp['time'].iloc[0]
            self.conditions[i, 2] = grp['biofuel'].iloc[0]
            self.conditions[i, 3] = grp['catalyst'].iloc[0]
            if self.targets is not None:
                self.targets[i, 0] = grp['target_vis'].iloc[0]
                self.targets[i, 1] = grp['target_ox'].iloc[0]

        self.prop_scaler = SafeScaler()
        self.cond_scaler = SafeScaler()
        self.target_scaler = SafeScaler()

        if fit_scaler:
            # 1. Временная замена NaN/Inf на 0 для безопасного скейлинга
            temp_props = np.nan_to_num(self.props, nan=0.0, posinf=0.0, neginf=0.0)
            self.prop_scaler.fit(temp_props.reshape(-1, n_props))
            # 2. Трансформируем
            self.props = self.prop_scaler.transform(temp_props.reshape(-1, n_props)).reshape(self.props.shape)
            # 3. ЖЁСТКО обнуляем паддинг (mask=False)
            self.props[~self.mask] = 0.0
            
            self.cond_scaler.fit(self.conditions)
            self.conditions = self.cond_scaler.transform(self.conditions)
            
            if self.targets is not None:
                self.target_scaler.fit(self.targets)
                self.targets = self.target_scaler.transform(self.targets)

    def _transform(self, scalers):
        self.prop_scaler = scalers['prop']
        self.cond_scaler = scalers['cond']
        self.target_scaler = scalers['target']
        
        temp_props = np.nan_to_num(self.props, nan=0.0, posinf=0.0, neginf=0.0)
        self.props = self.prop_scaler.transform(temp_props.reshape(-1, self.props.shape[-1])).reshape(self.props.shape)
        # ✅ Критически важно: обнуляем паддинг и для тестового набора
        self.props[~self.mask] = 0.0
        
        self.conditions = self.cond_scaler.transform(self.conditions)
        if self.targets is not None:
            self.targets = self.target_scaler.transform(self.targets)
            
    def __len__(self): return len(self.scenario_ids)
    
    def __getitem__(self, idx):
        out = {
            'props': torch.tensor(self.props[idx]),
            'conc': torch.tensor(self.conc[idx]).unsqueeze(-1),
            'conditions': torch.tensor(self.conditions[idx]),
            'mask': torch.tensor(self.mask[idx]),
            'type_ids': torch.tensor(self.type_ids[idx], dtype=torch.long)
        }
        if self.targets is not None:
            out['target'] = torch.tensor(self.targets[idx])
        return out