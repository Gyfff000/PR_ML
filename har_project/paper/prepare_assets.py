"""从已保存的 CSV/JSON/OOF 预测生成论文表格和图；不训练或推断。
运行环境：NumPy + Matplotlib。必须在完整 har_project 仓库中运行。
已生成的 figures/tables 可独立编译，不要求重新运行本脚本。
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
sources = set()
BLUE, ORANGE = '#32688E', '#C3643C'

def read_csv(relative):
    p = ROOT / relative
    sources.add(p)
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def read_json(relative):
    p = ROOT / relative
    sources.add(p)
    return json.loads(p.read_text(encoding='utf-8-sig'))

def table(name, caption, label, columns, header, rows, note=''):
    text = '\\begin{table}[htbp]\\centering\\small\n'
    text += '\\caption{' + caption + '}\\label{' + label + '}\n'
    text += '\\begin{tabular}{' + columns + '}\\toprule\n'
    text += ' & '.join(header) + r'\\\midrule' + '\n'
    text += '\n'.join(' & '.join(map(str,row)) + r'\\' for row in rows)
    text += '\n\\bottomrule\\end{tabular}\n'
    if note:
        text += '\\tabnote{' + note + '}\n'
    text += '\\end{table}\n'
    (PAPER/'tables'/f'{name}.tex').write_text(text,encoding='utf-8')

def style():
    for f in ['C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/simhei.ttf']:
        if Path(f).exists():
            font_manager.fontManager.addfont(f)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=f).get_name()
            break
    else:
        names = {f.name for f in font_manager.fontManager.ttflist}
        for name in ['Noto Sans CJK SC','Noto Sans CJK JP','SimHei']:
            if name in names:
                plt.rcParams['font.family']=name
                break
        else:
            raise RuntimeError('需要中文字体，例如微软雅黑或 Noto Sans CJK。')
    plt.rcParams.update({'font.size':10,'axes.unicode_minus':False,
        'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})

def save(fig, name):
    fig.savefig(PAPER/'figures'/f'{name}.pdf',bbox_inches='tight',pad_inches=.05)
    fig.savefig(PAPER/'figures'/f'{name}.png',dpi=240,bbox_inches='tight',pad_inches=.05)
    plt.close(fig)

def architecture():
    fig,ax=plt.subplots(figsize=(6.1,4.4))
    ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.009',
            facecolor=color,edgecolor='#82929E'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10,linespacing=1.5)
    def arrow(a,b):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,
            color='#536877',shrinkA=2,shrinkB=2))
    box(.025,.84,.43,.13,'人工特征：561 维','#EDF3F8')
    box(.545,.84,.43,.13,'时序窗口：9 × 128','#EDF3F8')
    box(.025,.40,.43,.35,'MLP 编码器\nLinear：561 → 128 → 64\nReLU + Dropout(0.3)','#E8F0F6')
    box(.545,.40,.43,.35,'CNN 编码器\n通道：9 → 32 → 64 → 128\n卷积核：7 / 5 / 3\nBN、ReLU、两次池化\n全局均值池化、Dropout(0.3)','#E8F0F6')
    box(.22,.18,.56,.12,'拼接：64 + 128 = 192 维','#F8EBE3')
    box(.32,.015,.36,.10,'线性分类：192 → 6','#F8EBE3')
    arrow((.24,.83),(.24,.76)); arrow((.76,.83),(.76,.76))
    arrow((.24,.39),(.35,.31)); arrow((.76,.39),(.65,.31))
    arrow((.50,.17),(.50,.125))
    fig.subplots_adjust(left=0,right=1,top=1,bottom=0)
    save(fig,'architecture')

def main():
    style()
    metrics=read_json('outputs/final_test/final_test_metrics.json')
    config=read_json('models/final/experiment_config.json')
    source=ROOT/'fusion_model.py'; sources.add(source)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==config['fusion_source_sha256']
    report={m:read_json(f'outputs/final_test/{m}_classification_report.json')
            for m in ['logistic','fusion']}
    names=metrics['class_names']
    oof=ROOT/'outputs/fusion_oof_predictions.npz'; sources.add(oof)
    with np.load(oof,allow_pickle=False) as d:
        labels=d['labels']; subjects=d['subjects']; folds=d['folds']
        counts=np.bincount(labels,minlength=6)
        assert len(labels)==7352 and len(np.unique(subjects))==21
        for k in range(1,6):
            assert not np.intersect1d(subjects[folds==k],subjects[folds!=k]).size
    testcounts=[int(report['logistic'][n]['support']) for n in names]
    table('data','官方训练集与测试集的类别分布','tab:data','lrrr',
        ['类别','训练窗口数','测试窗口数','总数'],
        [[n,int(a),b,int(a)+b] for n,a,b in zip(names,counts,testcounts)]
        +[['合计',int(sum(counts)),sum(testcounts),int(sum(counts))+sum(testcounts)]])
    entries=[]
    for name,f in [('逻辑回归','logistic'),('SVM','svm'),('随机森林','forest')]:
        r=max(read_csv(f'outputs/{f}_grid_cv.csv'),key=lambda r:float(r['cv_macro_f1']))
        entries.append((name,float(r['train_macro_f1']),float(r['cv_macro_f1']),float(r['cv_f1_std'])))
    for name,f in [('MLP','mlp'),('1D CNN','cnn'),('融合模型','fusion')]:
        rows=read_csv(f'outputs/{f}_group_cv.csv')
        vals=np.array([float(r['val_macro_f1']) for r in rows])
        entries.append((name,np.mean([float(r['train_macro_f1']) for r in rows]),float(vals.mean()),float(vals.std(ddof=0))))
    table('cv','六类模型的五折分组开发结果','tab:cv','lrrr',
        ['模型','训练 F1 均值','验证 F1 均值','折间标准差'],
        [[name,*[f'{v:.4f}' for v in values]] for name,*values in entries],
        'F1 均为宏平均；神经网络基础种子为 42。传统模型在同一组折上搜索参数，分数存在选择乐观性。')
    ordered=sorted(entries,key=lambda r:r[2],reverse=True)
    fig,ax=plt.subplots(figsize=(6.1,2.8),layout='constrained')
    y=np.arange(6)
    for i,(name,train,val,std) in enumerate(ordered):
        ax.errorbar(val,i,xerr=std,fmt='o',capsize=4,color=ORANGE if name=='融合模型' else BLUE,ms=5)
    ax.set_yticks(y,[r[0] for r in ordered]); ax.invert_yaxis()
    ax.set(xlim=(.85,1.005),xlabel='验证宏平均 F1（均值 ± 折间标准差）')
    ax.grid(axis='x',alpha=.2); save(fig,'cv_comparison')
    pca=read_csv('outputs/pca_compare_cv.csv')
    tab=[]
    for r in pca:
        lo,hi=int(r['min_components']),int(r['max_components'])
        dims=str(lo) if lo==hi else f'{lo}--{hi}'
        method=r['method'].replace('No PCA','不使用 PCA').replace('%',r'\%')
        tab.append([method,dims,f"{float(r['cv_macro_f1']):.4f}",f"{float(r['cv_f1_std']):.4f}",f"{float(r['seconds']):.1f}"])
    table('pca','固定逻辑回归 $C=0.1$ 的 PCA 对照','tab:pca','lrrrr',
        ['设置','各折维数范围','验证 F1','折间标准差','耗时/s'],tab,
        '保留比例指累计解释方差；各设置未重新搜索分类器参数，耗时为本机单次五折运行记录。')
    seeds=[]; means=[]
    for seed,prefix in [(42,'outputs'),(2024,'outputs/fusion_seed_2024'),(2025,'outputs/fusion_seed_2025')]:
        rows=read_csv(f'{prefix}/fusion_group_cv.csv')
        v=np.array([float(r['val_macro_f1']) for r in rows]); means.append(float(v.mean()))
        seeds.append([seed,f'{v.mean():.4f}',f'{v.std(ddof=0):.4f}',', '.join(r['selected_epochs'] for r in rows)])
    table('stability','融合模型三个基础种子的分组验证结果','tab:seeds','lrrl',
        ['基础种子','五折平均 F1','折间标准差','各折选定轮数'],seeds,
        '三次五折均值平均为 '+f'{np.mean(means):.4f}'+
        '，均值间样本标准差（ddof=1）为 '+f'{np.std(means,ddof=1):.4f}'+'；外层与内层受试者划分均固定。')
    table('test','冻结模型的官方独立测试结果','tab:test','lrrr',
        ['模型','准确率','宏平均 F1','错误窗口数'],
        [[name,f"{metrics['models'][m]['accuracy']:.4f}",f"{metrics['models'][m]['macro_f1']:.4f}",metrics['models'][m]['errors']]
        for m,name in [('logistic','逻辑回归'),('fusion','融合模型')]],
        '共 2947 个窗口、9 名受试者。两模型配置均在独立测试前固定。')
    for model in ['logistic','fusion']:
        rel=f'outputs/final_test/{model}_confusion_matrix.csv'
        # First column contains row labels.
        rows=read_csv(rel)
        cm=np.array([[int(r[n]) for n in names] for r in rows])
        normalized=cm/cm.sum(axis=1,keepdims=True)
        assert int(np.trace(cm))==metrics['test_samples']-metrics['models'][model]['errors']
        fig,ax=plt.subplots(figsize=(5.9,4.0),layout='constrained')
        im=ax.imshow(normalized,cmap='Blues',vmin=0,vmax=1)
        ax.set_xticks(range(6),names); ax.set_yticks(range(6),names)
        ax.set(xlabel='预测类别',ylabel='真实类别')
        for i in range(6):
            for j in range(6):
                ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=10,
                    color='white' if normalized[i,j]>.5 else '#263747')
        cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.04); cb.set_label('类别内比例')
        save(fig,f'cm_{model}')
    fig,ax=plt.subplots(figsize=(6.1,3.0),layout='constrained')
    x=np.arange(6); width=.35
    for dx,m,name,color in [(-width/2,'logistic','逻辑回归',BLUE),(width/2,'fusion','融合模型',ORANGE)]:
        ax.bar(x+dx,[report[m][n]['f1-score'] for n in names],width,label=name,color=color)
    ax.set_xticks(x,names); ax.set(ylim=(0,1.07),ylabel='类别 F1')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,1.19),frameon=False,ncol=2); ax.grid(axis='y',alpha=.15)
    save(fig,'class_f1')
    architecture()
    preds=read_csv('outputs/final_test/test_predictions.csv')
    paired={'both_correct':0,'logistic_only_correct':0,'fusion_only_correct':0,'both_wrong':0}
    subjectstats={}
    for r in preds:
        a=r['logistic_correct'].lower()=='true'; b=r['fusion_correct'].lower()=='true'
        key='both_correct' if a and b else 'logistic_only_correct' if a else 'fusion_only_correct' if b else 'both_wrong'
        paired[key]+=1
        d=subjectstats.setdefault(r['subject_id'],{'n':0,'logistic_correct':0,'fusion_correct':0})
        d['n']+=1; d['logistic_correct']+=int(a); d['fusion_correct']+=int(b)
    assert paired=={'both_correct':2754,'logistic_only_correct':45,'fusion_only_correct':62,'both_wrong':86}
    (PAPER/'audit_statistics.json').write_text(json.dumps({'paired':paired,'by_subject':subjectstats,
        'fusion_seed_means':means,'fusion_seed_mean':float(np.mean(means)),
        'fusion_seed_mean_sample_std':float(np.std(means,ddof=1))},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    manifest={'purpose':'Paper assets generated from saved results, without model training or inference',
        'versions':{'python':sys.version.split()[0],'numpy':np.__version__,'matplotlib':matplotlib.__version__},
        'source_sha256':{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}}
    (PAPER/'asset_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Generated 5 tables, 5 figure assets; paired statistics:',paired)

if __name__=='__main__':
    main()
