"""Build the Chinese report and figures from the read-only audit metrics."""
from pathlib import Path
import base64
import json
import os
import sqlite3
import sys

DEPS=Path(os.environ.get('V05_REPORT_DEPS',r'C:\tsbyttn\.tmp\v05-report-deps'))
if DEPS.exists(): sys.path.insert(0,str(DEPS))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import markdown

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
A=json.loads((OUT/'audit_metrics.json').read_text(encoding='utf-8'))
G=pd.DataFrame(A['groups'])
C=pd.DataFrame(A['consistency'])
AG=pd.DataFrame(A['aggregate'])
W=pd.DataFrame(A['weight_stats'])
BEST='stageA-b9-identity'
REF='stageA-b12-identity'
TAGS=[BEST,'stageA-b10-identity','stageA-b6-identity','stageB-b9-random-1',REF,'stageB-b9-reverse']
NAMES={BEST:'b9 identity','stageA-b10-identity':'b10 identity','stageA-b6-identity':'b6 identity',REF:'b12 full','stageB-b9-random-1':'b9 random-1','stageB-b9-reverse':'b9 reverse'}
COLORS=['#2563a6','#35a0a0','#758490','#8f78ad','#d08b2c','#bf665a']
PVALUES=[.005,.01,.02,.03,.04,.05]


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def group(kind,tag=BEST):
    return G[(G.kind==kind)&(G.tag==tag)].sort_values('p')


def agg(kind,tag):
    return AG[(AG.kind==kind)&(AG.tag==tag)].iloc[0]


def pc(x): return f'{100*x:.2f}%'
def num(x): return f'{x:,.0f}'


def savefig(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=160,bbox_inches='tight',facecolor='white')
    plt.close(fig)


plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#334155','text.color':'#1e293b','axes.titleweight':'bold','figure.constrained_layout.use':True})
fig,axs=plt.subplots(1,2,figsize=(11.5,4.2),sharex=True)
for ax,kind,title in zip(axs,['natural','stress'],['Natural iid X noise','Fixed-weight stress, six priors']):
    vals=[agg(kind,t)['mean'] for t in TAGS]
    ax.barh(range(len(TAGS)),vals,color=COLORS,height=.64)
    ax.set_yticks(range(len(TAGS)),[NAMES[t] for t in TAGS]); ax.invert_yaxis()
    for j,v in enumerate(vals): ax.text(v+.04,j,f'{v:.3f}',va='center',fontsize=9)
    ax.set_xlim(0,3.2); ax.set_xlabel('Mean online latency (s / decode)'); ax.set_title(title,pad=12)
    ax.grid(axis='x',alpha=.14); ax.set_axisbelow(True)
savefig(fig,'latency_comparison')

fig,axs=plt.subplots(1,2,figsize=(11.5,4.1),sharey=True)
for ax,kind,title in zip(axs,['natural','stress'],['Natural iid X noise','Fixed-weight stress']):
    g=group(kind)
    for col,label,color,marker in [('mean','Mean','#2563a6','o'),('p95','P95','#35a0a0','s'),('p99','P99','#d08b2c','^')]:
        ax.plot(g.p,g[col],label=label,color=color,marker=marker,linewidth=1.6,markersize=4)
    ax.set_title('b9 identity: '+title,pad=10);ax.set_xlabel('Physical p' if kind=='natural' else 'Decoder prior p')
    ax.set_ylabel('Online latency (s / decode)');ax.grid(alpha=.2);ax.legend(frameon=False)
    ax.set_ylim(.19,.42)
savefig(fig,'b9_tail_latency')

fig,axs=plt.subplots(1,2,figsize=(11.5,4.2))
for tag,color,label,shift in [(BEST,'#2563a6','b9 identity',-.00035),(REF,'#d08b2c','b12 full',.00035)]:
    g=group('natural',tag)
    lows=np.array([v[0] for v in g.ler_wilson95]); highs=np.array([v[1] for v in g.ler_wilson95])
    # Zero observed errors are plotted as zero, with their upper confidence bound retained.
    axs[0].errorbar(g.p+shift,100*g.ler,yerr=np.maximum(0,100*np.array([g.ler-lows,highs-g.ler])),color=color,marker='o',markersize=4,label=label,capsize=3,linewidth=1.1)
axs[0].set_xlabel('Physical p');axs[0].set_ylabel('Observed logical error rate (%)');axs[0].set_title('Natural noise: 5,000 shots per p');axs[0].legend(frameon=False);axs[0].grid(alpha=.18)
for weight,color in zip([2,3,4,5],COLORS):
    w=W[(W.kind=='stress')&(W.tag==BEST)&(W.weight==weight)].sort_values('p')
    axs[1].plot(w.p,100*w.ler,marker='o',label=f'weight {weight}',color=color,markersize=4)
axs[1].set_title('Stress: same 1,250 samples per weight');axs[1].set_xlabel('Decoder prior p');axs[1].set_ylabel('Observed conditional failure rate (%)');axs[1].legend(frameon=False);axs[1].grid(alpha=.18)
savefig(fig,'logical_failure_rates')

db=ROOT/'validation_partA_natural/results.sqlite'
conn=sqlite3.connect(db.as_uri()+'?mode=ro&immutable=1',uri=True)
drift=pd.read_sql_query("SELECT shot_id,wall_seconds FROM shots WHERE layout_tag='stageB-b9-reverse' AND dataset_name='natural_p0p05' AND oracle_calls=2 ORDER BY shot_id",conn)
conn.close()
fig,ax=plt.subplots(figsize=(11.5,3.2))
ax.scatter(drift.shot_id,drift.wall_seconds,s=3,alpha=.38,color='#758490',label='Individual shots, exactly 2 oracle calls',rasterized=True)
roll=drift.groupby(drift.shot_id//100).agg(shot=('shot_id','mean'),time=('wall_seconds','median'))
ax.plot(roll.shot,roll.time,color='#bf665a',linewidth=2,label='Median within each 100-shot ID window')
ax.set_xlabel('Shot ID (zero-based)');ax.set_ylabel('Online latency (s)');ax.set_title('Timing drift: b9 reverse, natural p=0.05',pad=10);ax.grid(alpha=.15);ax.legend(frameon=False,loc='upper left',fontsize=9)
savefig(fig,'reverse_timing_drift')

bn=agg('natural',BEST); bs=agg('stress',BEST); rn=agg('natural',REF); rs=agg('stress',REF)
txt=[]
def add(s):txt.append(s.strip())

add(f'''# BB72 多 syndrome 验证结果分析报告

分析日期：2026-09-23。实验记录覆盖 2026-09-19 至 2026-09-22（新加坡时间）。

数据来源：`exact_bnb/v0_5_multisyndrome_validation` 下 7 个 SQLite 数据库、14 个 CSV、7 组运行清单与冻结数据集，以及两份作业日志。报告以逐 shot 数据库重算为准。

**结果已经完整；b9 identity 的速度优势在此次多 syndrome 验证中保持成立。数值正确性应保留限定：扇区选择存在少量差异，低先验概率压力测试也没有全部通过原来的 1e-10 概率一致性阈值。**

- 预期和实际均为 **360,000 次解码**，72 个 `(layout, dataset)` 组合各 5,000 条，未发现缺失、重复或汇总失配。
- 自然噪声下 b9 identity 平均 **{bn['mean']:.4f} s/shot**，相对 b12 完整后验基准快 **{rn['mean']/bn['mean']:.2f} 倍**；压力测试下为 **{bs['mean']:.4f} s/shot**，快 **{rs['mean']/bs['mean']:.2f} 倍**。
- b9 在全部 12 个数据条件下均为平均耗时最低的已测布局。自然噪声的平均 oracle 调用为 **{bn['calls_mean']:.4f}**，压力测试为 **{bs['calls_mean']:.4f}**。
- b9 与 b12 的扇区差异分别为 **103/30,000**（自然噪声）和 **271/30,000**（压力测试）。b9 的概率一致性阈值超限分别为 **0** 和 **125** 条。
- 全部 5 个候选布局对 b12 的 300,000 次配对比较，共 **1,860** 次扇区差异、**635** 次概率差超过 1e-10。两种计数有重叠，均不是独立物理样本数。

## 1. 实验范围与完整性

码与通道：BB72 `[[72,12,6]]`，独立 X 错误，用 H_Z 和 logical-Z observables 进行解码。日志记录设备为 **NVIDIA A100-SXM4-40GB**，单 GPU，Torch 2.10.0+cu128，cuTensorNet，float64。这里测到的是 A100 性能。

共验证 6 个布局：b9 identity（9+3）、b10 identity（10+2）、b6 identity（6+6）、b12 full（12）、b9 random-1（9+3 的另一种逻辑划分）、b9 reverse（9+3 的反向逻辑顺序）。报告中的简称对应原始 `stageA-*` / `stageB-*` 标签。
''')
rows=[]
for a in A['audits']:
    label='Part A：自然噪声，6 个 p' if 'partA' in a['run'] else 'Part B：压力，p='+a['run'].split('_p')[-1]
    rows.append([label,num(a['expected']),num(a['actual']),'完整',f"{a['online_hours']:.2f}"])
add(table(['数据组','预期解码数','实际解码数','状态','在线耗时之和（h）'],rows))
add('''检查包括 SQLite `integrity_check`、主键唯一性、每组 shot_id 精确覆盖 0–4999、数值有限性、耗时/概率范围、冻结数据的 shape 与权重，以及数据库的 true_lambda、错误权重、syndrome 权重与 NPZ 逐条一致。`logical_success` 与 `best_lambda == true_lambda` 一致；每条记录均满足叶节点记账合计 4096、拓扑调用数合计等于 oracle_calls。14 个已有 CSV 的全部指标均与数据库重算结果一致。

上述检查核验了已有结果之间的一致性。本次没有重新在 GPU 上收缩张量网络，也没有重新构造 H_Z/L_Z 来独立验证冻结文件中的 syndrome 映射。

**样本数应按实验设计解释：** Part A 是 6×5,000=30,000 次自然噪声抽样，每个样本给 6 个布局解码。Part B 是同一批 5,000 次固定权重抽样，在 6 个解码先验下分别给 6 个布局解码。本次核对了全部数组，6 个压力数据集的 errors、syndromes、true_lambdas 和权重完全相同。因此总计是 35,000 次原始抽样、60,000 个“样本与先验组合”、360,000 次布局解码，不能当作 360,000 个独立噪声样本。

README 描述的是只运行一个压力先验时每布局 35,000 次解码；本目录的 PBS 实际扩展到了 6 个压力先验，因此是每布局 60,000 次。Part A 的空 stress 文件与 Part B 的空 natural 文件对应 shots=0，是正常产物。

## 2. 耗时与布局选择

以下耗时是数据库中的 `wall_seconds`：从 `ExactBlockBnB.search()` 开始到输出最优逻辑扇区，包含同步 oracle 调用和搜索控制。路径搜索、decoder 构造、warm-up、search 之前的输入转换、SQLite 提交和汇总输出不在此指标内。`oracle_seconds` 还包括调用包装与同步，不能当作纯 GPU kernel 时间。

### 各布局的总体性能
''')
rows=[]
for tag in TAGS:
    n=agg('natural',tag);s=agg('stress',tag)
    rows.append([NAMES[tag],f"{n['mean']:.4f}",f"{n['p99']:.4f}",f"{s['mean']:.4f}",f"{s['p99']:.4f}",f"{rn['mean']/n['mean']:.2f}×",f"{n['hours']+s['hours']:.2f}"])
add(table(['布局','自然平均 s','自然 P99 s','压力平均 s','压力 P99 s','自然对 b12 加速','60,000 次在线 h'],rows))
add('''![各布局平均解码耗时](latency_comparison.png)

自然与压力各有 30,000 次解码/布局，汇总均按真实 shot 加权；此处各 p 的样本数相同。压力汇总用于描述本次工作负载的耗时，不能据此计算 30,000 个独立样本的误差条。
''')
add(f'''b9 identity 相对 b10 identity 的平均时间减少 **{100*(1-bn['mean']/agg('natural','stageA-b10-identity')['mean']):.1f}%**（自然）和 **{100*(1-bs['mean']/agg('stress','stageA-b10-identity')['mean']):.1f}%**（压力）。b10 虽然调用次数稍少，单次收缩和根块输出的代价更高。b9 random-1 与 reverse 的调用数也与 identity 接近，但自然平均耗时分别是它的 **{agg('natural','stageB-b9-random-1')['mean']/bn['mean']:.2f} 倍**和 **{agg('natural','stageB-b9-reverse')['mean']/bn['mean']:.2f} 倍**。因此，优化目标需要同时考虑块划分对应的收缩成本和实际存活前缀数，不能只按 query 数排序。

原 P5.1 在 easy/hard 两个 syndrome 上给出的 b9 平均为 0.3005 s，此次自然总体均值更低。这反映样本难度分布不同，不能把两者的差值解释为新的代码提速。

### b9 identity 随 p 的性能
''')
rows=[]
for _,n in group('natural').iterrows():
    s=group('stress').set_index('p').loc[n.p]
    rows.append([f'{n.p:g}',f'{n["mean"]:.4f}',f'{n.p99:.4f}',f'{n.calls_mean:.4f}',f'{s["mean"]:.4f}',f'{s.p99:.4f}',f'{s.calls_mean:.4f}'])
add(table(['p','自然平均 s','自然 P99 s','自然平均调用','压力平均 s','压力 P99 s','压力平均调用'],rows))
add('''![b9 耗时分布](b9_tail_latency.png)

自然 p 从 0.005 到 0.05，b9 平均耗时从 0.2075 s 增至 0.2176 s，仅增约 4.9%，但 P99 从 0.2076 s 增至 0.3913 s。说明高 p 下主要增加了少量困难样本的尾部开销。压力样本固定后，平均耗时约 0.213–0.214 s，对当前先验范围不敏感。

## 3. Query 数与剪枝的实际含义

此实现的一个 oracle call 会返回一整个逻辑块的质量张量。b9 根调用返回 512 个九位前缀质量，随后每个存活前缀的一次调用返回 8 个剩余扇区质量。它与旧 sweep 的“一个 parity mask 一次标量 query”不是同一种工作量。
''')
add(table(['指标','自然噪声 b9','压力测试 b9'],[
    ['平均 oracle 调用',f'{bn.calls_mean:.4f}',f'{bs.calls_mean:.4f}'],
    ['恰好 2 次调用',f'{int(bn.two_calls):,}/30,000（{pc(bn.two_calls/30000)}）',f'{int(bs.two_calls):,}/30,000（{pc(bs.two_calls/30000)}）'],
    ['平均第二块存活前缀 N_survive',f'{bn.calls_mean-1:.4f}',f'{bs.calls_mean-1:.4f}'],
    ['最大调用次数',int(bn.calls_max),int(bs.calls_max)],
    ['平均 leaves_evaluated',f'{bn.leaves_mean:.4f}',f'{bs.leaves_mean:.4f}'],
    ['oracle 占在线耗时',pc(bn.oracle_share),pc(bs.oracle_share)],
]))
add('''相同 9+3 划分如果不剪枝，完整遍历需要 1+512=513 次块 oracle 调用；当前自然均值 2.0359 次，相对于这种遍历减少约 99.60% 的调用。这个比例只比较相同块 oracle 的调用数，不能等同于 FLOP 减少或端到端加速。

b12 仅需一次调用，但直接形成全部 4096 个扇区质量，其单次收缩明显更贵。`leaves_evaluated` 只计搜索程序实际到访/比较的叶节点，不代表张量收缩实际生成的叶概率数；b12 的该指标接近 1，也仍然计算了完整的 4096 输出。旧 sweep 的 4095 parity queries 与这里约 2 次块调用同样不能直接换算成约 2000 倍提速。

## 4. 与完整后验基准的一致性

`consistency_vs_b12.csv` 使用两个不同判据：`same_lambda` 是选出的 12 位扇区完全相同；`prob_close_1e-10` 是各自归一化最优概率的绝对差不超过 1e-10。这里的 b12 是同一套 float64 TN 实现的完整输出基准，仍会受浮点误差影响。
''')
rows=[]
for tag in TAGS:
    if tag==REF:continue
    ns=C[(C.kind=='natural')&(C.tag==tag)];ss=C[(C.kind=='stress')&(C.tag==tag)]
    rows.append([NAMES[tag],int(ns.different.sum()),int(ss.different.sum()),int(ns.probability_not_close.sum()),int(ss.probability_not_close.sum()),f'{max(ns.max_probability_gap.max(),ss.max_probability_gap.max()):.3e}'])
add(table(['候选布局','自然扇区差异 /30k','压力扇区差异 /30k','自然概率超限','压力概率超限','最大概率绝对差'],rows))
add('''### b9 的差异集中在哪里
''')
rows=[]
for _,r in C[C.tag==BEST].sort_values(['kind','p']).iterrows():
    rows.append(['自然' if r.kind=='natural' else '压力',f'{r.p:g}',r.different,r.probability_not_close,f'{r.max_probability_gap:.3e}',r.candidate_only_success,r.reference_only_success])
add(table(['类型','p','扇区差异 /5000','概率超限 /5000','最大概率差','仅 b9 成功','仅 b12 成功'],rows))
add('''自然噪声中，b9 的 103 条扇区差异对应的最优概率差最大为 **3.744×10⁻¹³**。所有候选布局的自然扇区差异，其最优概率差最大也只有 4.141×10⁻¹³。这与并列或近似并列最大值受浮点收缩顺序影响的情况相符。代码的 incumbent 更新只用严格的 `mass > best_mass`，没有跨布局统一的容差平局规则。

但数据库只保存最终扇区和该扇区的概率，没有保存完整 posterior、runner-up 概率、候选扇区在同一基准张量下的概率。因此，**不能仅凭两个最优概率接近就证明每一个不同扇区都是合法的并列 MAP 解，也不能把扇区不一致率直接当成算法错误率。**

在低 p 压力数据上，浮点消减问题更明显。635 条概率超限中，581 条来自压力 p=0.005、53 条来自压力 p=0.01，其余 1 条来自自然 p=0.03。它们涉及错误权重 5 的 569 次比较、权重 4 的 65 次比较和权重 7 的 1 次比较。其中 594 次仍然选中了相同扇区，41 次同时发生扇区差异。

### 一个具体复核案例

压力 `p=0.005`、`shot_id=827`（从 0 开始计数），错误权重 5、syndrome 权重 9：

| 项目 | b9 identity | b12 full |
| --- | --- | --- |
| best_lambda | `000000000000` | `001010111000` |
| true_lambda | `001010111000` | `001010111000` |
| best_probability | 0.4997853025536222 | 0.4997852695767487 |
| logical_success | 0 | 1 |
| root_mass | 1258.0145005967775 | 1258.0145856758368 |

这条的最优概率差约 **3.298×10⁻⁸**，有扇区选择差异，也有归一化分母差异。其他几个候选布局同样给出约 0.49978530255 的最优概率，因此不能直接认定 b12 的概率数值更准确，需要在统一收缩/更高精度的结果下交叉求值。

`root_mass` 是带公共 TN 缩放因子的未归一化质量，数值可以大于 1；不能把它直接当作 syndrome 概率。应该比较归一化概率或同一缩放下的相对误差。

### 数值保护是否足够
''')
rows=[]
for kind,label in [('natural','自然 b9'),('stress','压力 b9')]:
    r=agg(kind,BEST)
    rows.append([label,f'{int(r.clipped_shots):,}/30,000',f'{r.max_negative_root_ratio:.3e}',f'{r.max_guard_root_ratio:.3e}'])
rows.append(['压力 b12',f'{int(rs.clipped_shots):,}/30,000',f'{rs.max_negative_root_ratio:.3e}',f'{rs.max_guard_root_ratio:.3e}'])
add(table(['分组','出现负质量裁剪的 shot','最大 |负质量|/root','最大 local_guard/root'],rows))
add('''负值的绝对大小依赖 TN 缩放，相对 root 的比例更有解释力。出现裁剪本身不等于解码失败。当前程序在负质量超过 root 的 1e-6 时才中止；本次没有触发中止。可是“没有中止”并不等于通过了 1e-10 的跨布局概率一致性检查。

代码按已观察到的负质量量级构造剪枝保护项，这是实际数值防护；它没有为所有正质量收缩误差提供严格上界。因此，现有结果支持“数值上高度接近、速度优势稳定”，尚不能作为对全部浮点剪枝决策的严格精度证明。

## 5. 逻辑解码成功率

这里的 `logical_success=1` 定义为预测逻辑扇区 `best_lambda` 等于冻结物理错误对应的 `true_lambda`。它表示这次样本的逻辑解码成功，和任务是否完整、是否与 b12 一致是三个不同指标。即使精确 MAP 解码，也可能对某次实际错误解码失败。

### 自然噪声 LER

以下每行各有 5,000 个自然噪声样本。95% 区间采用逐 p 的二项 Wilson 区间；没有将不同 p 或不同布局混合成独立试验。
''')
rows=[]
for _,n in group('natural').iterrows():
    ref=group('natural',REF).set_index('p').loc[n.p]
    lo,hi=n.ler_wilson95
    if n.failure==0:lo=0.
    rows.append([f'{n.p:g}',int(n.failure),pc(n.ler),f'{100*lo:.3f}%–{100*hi:.3f}%',int(ref.failure),pc(ref.ler)])
add(table(['p','b9 失败数','b9 LER','b9 95% 区间','b12 失败数','b12 LER'],rows))
add('''p=0.005 的 0/5,000 是“未观察到失败”，并非证明真实 LER 为零；Wilson 95% 上界约为 0.0768%。自然样本总计 b9 比 b12 多 13 次失败，配对分解为 35 次仅 b9 成功、48 次仅 b12 成功。由于这些差异集中在近似平局样本，现阶段不能据此断言一种布局的物理纠错能力更好。

### 压力数据按错误权重分层

固定权重数据中，p 只表示解码器使用的独立噪声先验，实际样本始终按权重 2、3、4、5 各 1,250 个构成。这里报告的是该压力分布上的条件失败率，不能当作自然噪声 LER 曲线。
''')
rows=[]
for p in PVALUES:
    row=[f'{p:g}']
    for weight in [2,3,4,5]:
        v=W[(W.kind=='stress')&(W.tag==BEST)&(W.p==p)&(W.weight==weight)].iloc[0]
        row.append(f'{int(v.failure)}/1250（{pc(v.ler)}）')
    rows.append(row)
add(table(['解码先验 p','权重 2','权重 3','权重 4','权重 5'],rows))
add('''![自然 LER 与压力条件失败率](logical_failure_rates.png)

权重 2 在所有先验下均未观察到失败。权重 5 的失败率约 24%，是压力失败的主要来源。先验从 0.005 调到 0.05 后，耗时和条件失败率变化都较小；由于重复使用同一批样本，且平局选择尚未统一，不宜按这些微小波动选择“最优先验”。

## 6. 耗时异常与资源账目

自然 `p=0.05` 的 b9 reverse 有一段额外变慢。限定 oracle_calls 恰好为 2，shot_id 0–999 的平均耗时为 **2.6503 s**，而 4000–4999 为 **4.0810 s**。调用次数未增加仍然变慢，说明这部分不能由 BnB 扩展更多前缀来解释。

![相同调用数下的运行时漂移](reverse_timing_drift.png)

现有日志没有 GPU 利用率、频率、温度、其他作业占用等时间序列，无法据此确定外部负载或硬件状态是否是原因。本报告保留全部观测值，没有删除慢样本。该现象需要在比较尾延迟或进行更细的布局竞速时单独复测。
''')
add(f'''全部逐 shot 在线时间之和为 **{A['total_online_hours']:.2f} GPU·h**（单 GPU 时间口径），其中自然约 66.60 h、压力约 66.74 h。这不等同于 PBS 计费时间，初始化、提交、空闲和中断间隔不在其中。Part A 日志从 9 月 19 日 10:08:19 到 9 月 22 日 04:48:54，约 66.68 h；Part B 当前日志从中途续跑开始，前面若干组直接 skip，因此不能拿这份日志的首尾差当作全部 Part B 耗时。

仅 b9 identity 的 60,000 次在线解码共 **{bn.hours+bs.hours:.2f} h**。若后续只跑已测最优布局，相同规模的已记录在线工作量会远少于本次六布局验证；这是基于本次相同数据条件的工作量估计，不包含新增运行的系统开销，也不是新 syndrome 分布的保证。

## 7. 建议的后续验证

1. **保留 b9 identity 作为性能基线，优先复核差异样本。** 先检查低 p 压力 shot 827 及其他概率超限/扇区差异样本，无需马上重跑完整 360,000 次验证。报告目录的差异清单保留 run、布局、dataset、shot_id、概率与扇区，原始错误和 syndrome 可由 NPZ 定位。
2. **在同一 b12 posterior 下评价候选扇区。** 对差异样本保存 P_ref(lambda_b9)、P_ref(lambda_b12)、top-2 gap、归一化残差和 root 质量差，区分并列最优、数值归一化差异和真正次优解。必要时用另一条收缩路径或更高精度交叉校验。当前 `best_probability` 的接近只能作为线索。
3. **统一平局规则后再比较逐 shot 成功率。** 明确绝对/相对容差与固定的扇区排序，记录候选最优集合；只修改平局裁决不能消除正质量误差，需要先完成上述交叉评价。不要仅放宽 1e-10 阈值来宣称“完全通过”。
4. **若继续性能优化，先测每种拓扑的单次成本。** b9 当前约 99% 的在线时间仍在 oracle 包装调用内。应分别记录根块与第二块耗时、存活前缀数和 GPU 状态，同时复测 reverse 的时段漂移。进一步减少 query 的收益应按总时间评估。

从当前记录可得出的明确结论是：**b9 identity 的速度优胜已在此多样本工作负载上重现；完整性合格；严格逐扇区一致性与低 p 压力数值精度仍需针对差异样本补验证。**

## 8. 复核依据与附件

- 逐 shot 来源：7 个 `validation_*/results.sqlite` 的 `shots` 和 `layout_meta` 表。每个 shot 的 `wall_seconds`、`oracle_seconds`、`oracle_calls`、成功标志和数值诊断都在数据库中。
- 设置和样本：各目录的 `run_manifest.json`、`datasets/dataset_manifest.json`、`datasets/*.npz`。基础随机种子为 20260919。
- 指标定义：[validation_utils.py](../validation_utils.py) 中 `write_summaries` 与建表定义；[run_bb72_multisyndrome_validation.py](../run_bb72_multisyndrome_validation.py) 中 `decode_once`、success 计算与逐 shot 写入；[exact_block_bnb.py](../exact_block_bnb.py) 中 `search`、`_dfs`、`_sanitize`、`_update_incumbent`；[multi_logical_tensor_network_decoder.py](../multi_logical_tensor_network_decoder.py) 的未归一化质量说明。
- 执行记录：[Part A 日志](../output-partA-natural.txt)、[Part B 日志](../output-partB-stress.txt)。数据库 `created_at` 使用 SQLite UTC，报告中的日历时间已按新加坡 UTC+8 解读。
- [重算指标与概率超限明细](audit_metrics.json)：含输入数据库 SHA-256、完整性结果、72 组统计与全部 635 条概率超限比较。
- [扇区差异明细](sector_mismatches.json)：全部 1,860 条候选布局与 b12 的扇区差异比较，可按 run、tag、dataset_name、shot_id 精确定位。

本次只读取原始结果与代码，并在独立的 `analysis_20260923` 目录生成分析产物；没有更改原始运行代码、数据库或 CSV。
''')

md='\n\n'.join(txt)+'\n'
(OUT/'BB72_v0_5_分析报告.md').write_text(md,encoding='utf-8')
body=markdown.markdown(md,extensions=['tables','fenced_code','toc'])
for name in ['latency_comparison','b9_tail_latency','logical_failure_rates','reverse_timing_drift']:
    uri='data:image/png;base64,'+base64.b64encode((OUT/f'{name}.png').read_bytes()).decode()
    body=body.replace(f'src="{name}.png"',f'src="{uri}"')
style='''html{scroll-behavior:smooth}body{margin:0;background:#f1f5f9;color:#1e293b;font-family:"Microsoft YaHei","PingFang SC",Arial,sans-serif;line-height:1.85}main{max-width:1060px;margin:32px auto;background:white;padding:42px 52px 60px;box-shadow:0 4px 24px #0f172a0a}h1{font-size:29px;line-height:1.45;margin-top:0}h2{font-size:22px;margin-top:46px;border-bottom:1px solid #dbe3ee;padding-bottom:10px}h3{font-size:18px;margin-top:28px}p,li{font-size:15px}a{color:#245b91}strong{color:#183b61}code{font-family:Consolas,monospace;background:#f3f6fa;padding:2px 5px;border-radius:3px;overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:13px;margin:20px 0;line-height:1.6}th{background:#edf2f7;text-align:left;padding:10px 9px;border-bottom:2px solid #bac8d9}td{padding:9px;border-bottom:1px solid #e4eaf0;vertical-align:top}tr:nth-child(even){background:#f8fafc}img{max-width:100%;height:auto;display:block;margin:20px auto}li{padding-bottom:7px}pre{overflow:auto} @media(max-width:800px){main{padding:22px 18px;margin:0}h1{font-size:24px}table{display:block;overflow:auto;white-space:nowrap}}@media print{body{background:white}main{margin:0;padding:0;box-shadow:none}h2,h3{break-after:avoid}img,table{break-inside:avoid}a{color:inherit}}'''
html='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BB72 多 syndrome 验证结果分析</title><style>'+style+'</style></head><body><main>'+body+'</main></body></html>'
(OUT/'BB72_v0_5_分析报告.html').write_text(html,encoding='utf-8')
assert len(G)==72 and A['total_rows']==360000
assert all(not a['issues'] for a in A['audits']) and not A['csv_issues']
assert len(A['probability_exceptions'])==635
assert sum(C.different)==1860
assert sum(C.n)==300000
assert all(group(kind).set_index('p').loc[p]['mean']==G[(G.kind==kind)&(G.p==p)]['mean'].min() for kind in ['natural','stress'] for p in PVALUES)
print(json.dumps({'report':str(OUT/'BB72_v0_5_分析报告.html'),'markdown_characters':len(md),'figures':4,'checks':'PASS'},ensure_ascii=True))
