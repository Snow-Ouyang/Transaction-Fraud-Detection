"""Inspect realized D3 tree shapes after choices; never fit or score data."""
import numpy as np
import pandas as pd
from model_io import load_model,suffix


def summarize(folder,runs,ids):
    rows=[]
    for rid in dict.fromkeys(ids):
        r=runs[rid];family=r['family'];fold=next(f for f in r['folds'] if f['fold']=='D3')
        model=load_model(folder/'runs'/rid/('D3'+suffix(family)),family)
        if family=='catboost':
            leaves=np.asarray(model.get_tree_leaf_counts(),dtype=int)
            depths=np.log2(leaves)
            occupied=int(np.count_nonzero(np.asarray(model.get_leaf_weights())>0))
        else:
            tree_frame=model.get_booster().trees_to_dataframe()
            leaves=tree_frame.loc[tree_frame.Feature.eq('Leaf')].groupby('Tree').size().to_numpy()
            occupied=int((tree_frame.Feature.eq('Leaf') & tree_frame.Cover.gt(0)).sum())
            depths=[]
            for tree,g in tree_frame.groupby('Tree',sort=True):
                d={f'{tree}-0':0}
                for node in g.sort_values('Node').itertuples(index=False):
                    if node.Feature!='Leaf':
                        d[str(node.Yes)]=d[node.ID]+1;d[str(node.No)]=d[node.ID]+1
                depths.append(max(d.values()))
            depths=np.asarray(depths)
        rows.append({'run_id':rid,'family':family,'feature_count':len(r['features']),'tree_count':len(leaves),
                     'total_leaves':int(leaves.sum()),'mean_leaves_per_tree':float(leaves.mean()),
                     'nonempty_leaves':occupied,'nonempty_fraction':occupied/int(leaves.sum()),
                     'maximum_realized_depth':float(np.max(depths)),'mean_tree_depth':float(np.mean(depths)),
                     'model_megabytes':fold['model_bytes']/1024**2,'fit_seconds':fold['fit_seconds'],
                     'mean_development_delta_per_10000':r['mean_delta_per_10000']})
    return pd.DataFrame(rows)
