"""Plot and report existing progression results; never fit models or select candidates."""
from __future__ import annotations
import base64
import json
from pathlib import Path
import numpy as np
import pandas as pd


def render_report(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    import markdown

    folder = Path(folder)
    read = lambda name: pd.read_csv(folder/'tables'/f'{name}.csv')
    result = json.loads((folder/'results.json').read_text(encoding='utf-8'))
    protocol = json.loads((folder/'protocol.json').read_text(encoding='utf-8'))
    grid, progression = read('parameter_grid'), read('feature_progression')
    metrics = read('metrics')
    development = metrics.loc[metrics.role == 'development_selection'].copy()
    june = read('june_diagnostic')
    selected_id, final_id = result['selected_parameter_run'], result['final_retained_run']
    final_features = result['final_features']
    final = june.loc[june.run_id == final_id].iloc[0]
    original = june.loc[june.run_id == 'core_default'].iloc[0]
    tuned = june.loc[june.run_id == selected_id].iloc[0]
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        fm.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = [fm.FontProperties(fname=str(font)).get_name(), 'DejaVu Sans']
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10, 'savefig.dpi': 165,
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .15})
    blue, red, green, muted = '#316a9a', '#c55345', '#378373', '#778491'
    figures = []
    short_names = {'core_default': '原始核心', 'core_d4_l3': '深4 / L2=3', 'core_d4_l10': '深4 / L2=10',
                   'core_d6_l10': '深6 / L2=10', 'E1_amount_relative': '金额偏离', 'E2_age_weekday': '年龄/星期',
                   'E3_behavior_history': '其他历史', 'E4_merchant': '商户', 'E5_identity_proxies': '身份代理'}
    title_for = lambda run_id: short_names.get(run_id, run_id)

    def finish(fig, name, title, caption):
        fig.suptitle(title, x=.025, ha='left', fontsize=15)
        fig.tight_layout(rect=[0, 0, 1, .89])
        fig.savefig(folder/'figures'/f'{name}.png', bbox_inches='tight', facecolor='white')
        plt.close(fig)
        figures.append({'file': f'figures/{name}.png', 'title': title, 'caption': caption})

    ids = grid.run_id.tolist()
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(ids))
    for i, (fold, color) in enumerate(zip(['D1', 'D2', 'D3'], [blue, muted, red])):
        values = development.loc[development.fold == fold].set_index('run_id').reindex(ids).delta_per_10000
        ax.bar(x+(i-1)*.23, values, width=.23, color=color, label=fold)
    ax.plot(x, grid.set_index('run_id').reindex(ids).mean_delta_per_10000, 'kD', label='等窗口平均')
    ax.axhline(0, color='#222', lw=.8)
    ax.set_xticks(x, [title_for(v) for v in ids]); ax.set_ylabel('相对全部放行的效用增量 / 万笔 ($)'); ax.legend()
    finish(fig, '01_parameter_grid', '四组核心参数的开发窗口比较', '仅用三个开发窗口选参数；原始深6、L2=3模型直接复用。没有根据六月结果选择参数。')

    fig, ax = plt.subplots(figsize=(11, 5))
    values = progression.mean_paired_delta_per_10000
    colors = [green if bool(v) else muted for v in progression.retained]
    ax.bar(np.arange(len(progression)), values, color=colors)
    for i, row in progression.iterrows():
        ax.annotate(f'{int(row.improved_windows)}/3 窗口改善\n'+('保留' if row.retained else '不保留'),
                    (i, row.mean_paired_delta_per_10000), xytext=(0, 8 if row.mean_paired_delta_per_10000 >= 0 else -30),
                    textcoords='offset points', ha='center', fontsize=9)
    ax.set_xticks(range(len(progression)), progression.group_name)
    ax.axhline(0, color='#222', lw=.8); ax.set_ylabel('相对当前保留父版的平均效用变化 / 万笔 ($)')
    ax.margins(y=.25)
    finish(fig, '02_feature_increments', '逐组扩展：相对当前保留版本的增量', '平均变化为正且至少 2/3 开发窗口改善才保留；未保留的组不进入下一步。保留是研究路径决定，不代表上线或独立测试通过。')

    all_ids = result['run_ids']
    values = development.pivot(index='run_id', columns='fold', values='delta_per_10000').reindex(all_ids)
    fig, ax = plt.subplots(figsize=(9, 7))
    limit = max(abs(values.to_numpy().min()), abs(values.to_numpy().max()))
    im = ax.imshow(values, aspect='auto', cmap='RdBu', vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(values.columns)), values.columns)
    ax.set_yticks(range(len(values)), [title_for(v) for v in values.index]); ax.grid(False)
    for i in range(len(values)):
        for j in range(len(values.columns)):
            ax.text(j, i, f'{values.iloc[i,j]:,.0f}', ha='center', va='center', fontsize=9,
                    color='white' if abs(values.iloc[i,j]) > limit*.55 else '#222')
    fig.colorbar(im, ax=ax, label='效用增量 / 万笔 ($)', fraction=.04)
    finish(fig, '03_development_windows', '各候选在不同开发窗口的表现', '所有窗口均参与选择，图中数值是选择期表现，不是全新测试表现。不同扩展试验的父版见决策表。')

    diagnostic_ids = june.run_id.tolist()
    fig, ax = plt.subplots(figsize=(12, 5))
    vals = june.set_index('run_id').reindex(diagnostic_ids)
    bars = ax.bar(range(len(vals)), vals.utility, color=[green if v == final_id else blue if v == selected_id else muted for v in diagnostic_ids])
    for bar, run_id in zip(bars, diagnostic_ids):
        if run_id == final_id: bar.set_edgecolor('#162e3c'); bar.set_linewidth(2)
    ax.axhline(final.always_normal_utility, color=red, ls='--', label='全部放行')
    ax.axhline(0, color='#222', lw=.8)
    ax.set_xticks(range(len(vals)), [title_for(v)+('（最终）' if v == final_id else '') for v in diagnostic_ids], rotation=25, ha='right')
    ax.set_ylabel('六月复用诊断效用 ($)'); ax.legend()
    finish(fig, '04_june_reused_diagnostic', '六月仅在全部选择锁定后复用诊断', '六月已在 v1 中被观察和评估。本图不用于重新挑参数、特征或阈值；最终版本由此前开发窗口决定。')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, run_id, label in zip(axes, ['core_default', final_id], ['原始核心', '开发期选定最终版']):
        row = june.loc[june.run_id == run_id].iloc[0]
        count = np.array([[row.TN, row.FP], [row.FN, row.TP]], int)
        ax.imshow(np.log1p(count), cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{count[i,j]:,}', ha='center', va='center', fontsize=16,
                        color='white' if np.log1p(count[i,j]) > np.log1p(count).max()*.65 else '#222')
        ax.set_xticks([0,1], ['放行', '拦截']); ax.set_yticks([0,1], ['实际正常', '实际欺诈']); ax.set_title(label); ax.grid(False)
    finish(fig, '05_june_confusion', '原始核心与最终保留版本的六月诊断', '文字为原始计数，颜色为 log1p 计数；所有阈值均由各自四月窗口锁定。')

    from run_core import utility
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, run_id, label in zip(axes, [selected_id, final_id], ['调优后核心', '最终保留版']):
        prediction = pd.read_csv(folder/'runs'/run_id/'D3_threshold.csv.gz')
        decision = json.loads((folder/'runs'/run_id/'D3_decision.json').read_text(encoding='utf-8'))
        grid_t = np.unique(np.r_[np.geomspace(1e-5, .99, 150), decision['threshold'], .5, 5/155])
        ax.plot(grid_t, [utility(prediction.is_fraud, prediction.score >= t) for t in grid_t], color=blue)
        ax.axvline(decision['threshold'], color=red, ls='--', label=f'锁定 {decision["threshold"]:.5g}')
        ax.axhline(-100*prediction.is_fraud.sum(), color=muted, lw=.8)
        ax.set_xscale('log'); ax.set_xlabel('模型分数阈值'); ax.set_ylabel('四月阈值窗口效用 ($)'); ax.set_title(label); ax.legend()
    finish(fig, '06_threshold_selection', '扩展前后的四月阈值选择', '展示网格用于可视化；实际最优阈值扫描全部不同分数并处理并列。阈值窗口结果是选择样本内结果。')

    importance = pd.read_csv(folder/'runs'/final_id/'D3_feature_importance.csv').sort_values('importance')
    fig, ax = plt.subplots(figsize=(10, max(4.5, .32*len(importance)+1.2)))
    ax.barh(importance.feature, importance.importance, color=blue)
    ax.set_xlabel('CatBoost PredictionValuesChange')
    finish(fig, '07_feature_importance', '最终保留版本的拟合特征重要性', '只作为模型内部描述；相关特征会分担或替代重要性，不能解释成因果影响或独立增量。')

    base_slices = pd.read_csv(folder/'runs/core_default/June_slices.csv')
    final_slices = pd.read_csv(folder/'runs'/final_id/'June_slices.csv')
    base_cat = base_slices.loc[base_slices.slice_feature == 'category'].set_index('slice_value')
    final_cat = final_slices.loc[final_slices.slice_feature == 'category'].set_index('slice_value')
    order = base_cat.FP.add(final_cat.FP, fill_value=0).sort_values().index
    fig, ax = plt.subplots(figsize=(11, 6))
    pos = np.arange(len(order))
    ax.barh(pos-.18, base_cat.reindex(order).FP, height=.36, label='原始核心', color=muted)
    ax.barh(pos+.18, final_cat.reindex(order).FP, height=.36, label='最终保留版', color=green)
    ax.set_yticks(pos, order); ax.set_xlabel('六月误报数'); ax.legend()
    finish(fig, '08_category_false_positives', '误报是否仍集中在类别支持变化处', '切片仅作选择后的诊断，没有根据切片更换最终保留版本。')

    daily_rows = []
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for run_id, label, color in [('core_default','原始核心',muted), (selected_id,'调优后核心',blue), (final_id,'最终保留版',green)]:
        pred = pd.read_csv(folder/'runs'/run_id/'June_diagnostic.csv.gz')
        pred['date'] = pd.to_datetime(pred.unix_time, unit='s').dt.floor('D')
        daily = pred.groupby('date').apply(lambda g: utility(g.is_fraud, g.decision.astype(bool)), include_groups=False)
        ax.plot(daily.index, daily.values, label=label, color=color)
        daily_rows.extend({'run_id':run_id,'date':str(date.date()),'utility':int(value)} for date,value in daily.items())
    pd.DataFrame(daily_rows).to_csv(folder/'tables/june_daily_utility.csv',index=False,encoding='utf-8-sig')
    ax.axhline(0,color='#222',lw=.8); ax.set_ylabel('每日效用 ($)'); ax.legend(); fig.autofmt_xdate()
    finish(fig, '09_june_daily_variation', '六月复用诊断中的逐日变化', '同客户事件会聚集，逐日变化用于观察不稳定性，不当作独立重复实验或置信保证。')

    _write_narrative(folder, result, protocol, grid, progression, development, june, figures, markdown)
    (folder/'figure_index.json').write_text(json.dumps(figures, ensure_ascii=False, indent=2),encoding='utf-8')
    print('REPORT COMPLETE: 9 figures, report_zh.md and report_zh.html',flush=True)


def _write_narrative(folder, result, protocol, grid, progression, development, june, figures, markdown):
    # Narrative is generated after all model and feature decisions, solely from saved statistics.
    from narrative_progression import write_narrative
    write_narrative(folder, result, protocol, grid, progression, development, june, figures, markdown)
