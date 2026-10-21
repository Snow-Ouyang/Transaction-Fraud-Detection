"""Deterministic adapters. Vocabulary training never sees a future window."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'core_baseline/.deps'))
from catboost import CatBoostClassifier
from xgboost import XGBClassifier


def schema_for(fit, features, categorical):
    return {'features':features,'categorical':{
        c:sorted(fit[c].astype(str).unique().tolist()) for c in features if c in categorical}}


def transform(data, schema, family):
    x=data[schema['features']].copy()
    if family=='xgboost':
        for c,levels in schema['categorical'].items():
            x[c]=pd.Categorical(x[c].astype(str),categories=levels)
    else:
        for c in schema['categorical']: x[c]=x[c].astype(str)
    return x


def parameters(protocol, family, config):
    p={**protocol[family+'_fixed'],**{k:v for k,v in config.items() if k!='id'},'random_seed' if family=='catboost' else 'random_state':protocol['seed']}
    return p


def make_model(protocol, family, config, features):
    p=parameters(protocol,family,config)
    if family=='catboost':
        return CatBoostClassifier(**p,cat_features=[c for c in features if c in protocol['categorical_features']],verbose=200)
    return XGBClassifier(**p)


def suffix(family): return '.cbm' if family=='catboost' else '.ubj'


def load_model(path, family):
    m=CatBoostClassifier() if family=='catboost' else XGBClassifier()
    m.load_model(str(path))
    return m


def predict(model, data, schema, family):
    return model.predict_proba(transform(data,schema,family))[:,1].astype('float64')


def unknown_counts(data,schema):
    return {c:int((~data[c].astype(str).isin(levels)).sum()) for c,levels in schema['categorical'].items()}


def importance(model, family, features):
    vals=model.get_feature_importance(type='PredictionValuesChange') if family=='catboost' else model.feature_importances_*100
    return pd.DataFrame({'feature':features,'importance':vals,
                         'method':'PredictionValuesChange' if family=='catboost' else 'normalized_gain'})
