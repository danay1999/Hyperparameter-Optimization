"""Build the frozen October 30 paper from its Markdown narrative and saved tables.

Run from any directory: python report/build_report.py
Requirements: pandas, numpy, matplotlib, pandoc, pdfLaTeX (including newtx fonts, setspace, geometry, and caption).
The script never runs or changes experiments. Numerical prose is snapshot-specific.
"""
from pathlib import Path
import re, subprocess, shutil, json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; FIG=ROOT/'figures'; BUILD=ROOT/'build'
for d in [FIG,BUILD]:d.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,'savefig.bbox':'tight'})
OPS=['grid','random','bayesian','halving']; NAMES={'grid':'Grid','random':'Random','bayesian':'TPE','halving':'Halving'}
MODELS=['elastic_net','svm','lightgbm']; MN={'elastic_net':'Linear','svm':'SVM','lightgbm':'LightGBM'}
COLORS=['#9b6b43','#697786','#245d91','#2f826e']
TITLES={'white_wine':'White Wine','superconductivity':'Superconductivity','communities_crime':'Communities/Crime','bank_marketing':'Bank Marketing','bioresponse':'Bioresponse','credit_default':'Credit Default','synthetic_linear':'Synthetic linear','synthetic_friedman':'Friedman','synthetic_balanced':'Balanced synthetic','synthetic_imbalanced':'Imbalanced synthetic'}
def read(name):return pd.read_csv(DATA/name)
def save(fig,name):
 fig.savefig(FIG/f'{name}.pdf');fig.savefig(FIG/f'{name}.png',dpi=220);plt.close(fig)

runtime_summary=json.loads((DATA/'runtime_summary.json').read_text())
runtime_dataset=read('runtime_by_dataset.csv');runtime_optimizer=read('runtime_by_optimizer.csv');runtime_model=read('runtime_by_model.csv')
assert abs(runtime_summary['all_recorded_executions']['hours']-164.698672588959)<1e-8
assert abs(runtime_summary['primary_paper_snapshot']['hours']-135.71748028560455)<1e-8
ranks=read('table_03_average_ranks.csv');trade=read('table_17_tradeoff_summary.csv');budget=read('budget_gains_per_cell_seed.csv');robust=read('table_12_robustness_paired.csv');perf=read('table_02_performance.csv');coverage=read('table_01_coverage.csv')
# Frozen-snapshot invariants keep figures and narrative from silently drifting.
assert coverage.paired_seeds.sum()==448
assert len(budget)==896 # 224 cell-seed pairs times four optimizers
assert ranks.groupby(['model','budget']).apply(lambda g:g.loc[g.mean_rank.idxmin(),'optimizer'],include_groups=False).eq('bayesian').all()

fig,axes=plt.subplots(1,2,figsize=(7.1,2.4),sharey=True)
for ax,b in zip(axes,[25,100]):
 for j,op in enumerate(OPS):
  vals=[ranks[(ranks.model==m)&(ranks.budget==b)&(ranks.optimizer==op)].mean_rank.iloc[0] for m in MODELS]
  ax.bar(np.arange(3)+(j-1.5)*.18,vals,width=.18,color=COLORS[j],label=NAMES[op])
 ax.set(xticks=np.arange(3),xticklabels=[MN[m] for m in MODELS],ylim=(0,4),title=f'Budget {b}');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
axes[0].set_ylabel('Dataset-block average rank (lower is better)');fig.legend(*axes[1].get_legend_handles_labels(),ncol=4,loc='lower center',bbox_to_anchor=(.5,-.03),frameon=False);fig.tight_layout(rect=[0,.10,1,1]);save(fig,'ranks')

fig,axes=plt.subplots(2,3,figsize=(7.1,3.7))
for i,task in enumerate(['regression','classification']):
 for j,model in enumerate(MODELS):
  ax=axes[i,j];sub=budget[(budget.task==task)&(budget.model==model)]
  values=[sub[sub.optimizer==op].relative_gain_percent.dropna() for op in OPS]
  bp=ax.boxplot(values,positions=range(4),patch_artist=True,showfliers=False,widths=.65)
  for box,color in zip(bp['boxes'],COLORS):box.set_facecolor(color);box.set_alpha(.7)
  ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(4),[NAMES[o] for o in OPS],rotation=25);ax.set_title(f'{MN[model]} / {task}');ax.grid(axis='y',alpha=.2)
  if j==0:ax.set_ylabel('Relative improvement (%)')
fig.tight_layout();save(fig,'budget')

fig,axes=plt.subplots(1,3,figsize=(7.1,2.4))
for ax,model in zip(axes,MODELS):
 sub=trade[(trade.model==model)&(trade.budget==100)].set_index('optimizer')
 ax.barh(range(4),[sub.loc[o,'median_time_ratio'] for o in OPS],color=COLORS)
 ax.axvline(1,color='black',lw=.8,ls='--');ax.set_yticks(range(4),[NAMES[o] for o in OPS]);ax.set_title(MN[model]);ax.set_xlabel('Median time / random');ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
fig.tight_layout();save(fig,'runtime')

fig,axes=plt.subplots(2,2,figsize=(7.1,3.8))
for i,task in enumerate(['regression','classification']):
 for j,condition in enumerate(['noisy','small']):
  ax=axes[i,j];sub=robust[robust.task.eq(task)&robust.contrast.str.startswith(condition)]
  vals=[sub[sub.optimizer.eq(o)].relative_degradation_percent.dropna() for o in OPS]
  bp=ax.boxplot(vals,patch_artist=True,showfliers=False,widths=.6)
  for box,color in zip(bp['boxes'],COLORS):box.set_facecolor(color);box.set_alpha(.7)
  ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(1,5),[NAMES[o] for o in OPS],rotation=20);ax.set_title(f'{task.title()}: {condition} minus base');ax.set_ylabel('Relative degradation (%)');ax.grid(axis='y',alpha=.2)
fig.tight_layout();save(fig,'robustness')

curves=read('all_search_trajectories.csv');curves=curves[(curves.dataset=='synthetic_friedman')&(curves.condition=='base')&(curves.model=='lightgbm')&(curves.seed==1)]
fig,axes=plt.subplots(1,2,figsize=(7.1,2.5))
for op,color in zip(OPS,COLORS):
 sub=curves[curves.optimizer.eq(op)].sort_values('trial')
 for ax,x in zip(axes,['equivalents','elapsed_seconds']):ax.step(sub[x],-sub.incumbent,where='post',label=NAMES[op],color=color,lw=1.2)
axes[0].set(xlabel='Full-fidelity equivalents',ylabel='Best full-resource CV RMSE');axes[1].set(xlabel='Elapsed optimization seconds',ylabel='Best full-resource CV RMSE');axes[1].legend();fig.tight_layout();save(fig,'trajectory')

# Tables have a common content representation; only layout differs by target.
blocks={};tables=[]
def esc(s):
 s=str(s)
 for a,b in [('\\',r'\textbackslash{}'),('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:s=s.replace(a,b)
 return s

def table(token,caption,headers,rows,widths=None,wide=False):
 md='**'+caption+'**\n\n'+'| '+' | '.join(headers)+' |\n'+'| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
 env='table'
 spec=widths or ('l'+'r'*(len(headers)-1))
 latex='\\begin{'+env+'}[t]\n\\centering\n\\caption{'+esc(caption)+'}\n\\label{tab:'+token.lower()+'}\n\\fontsize{10}{12}\\selectfont\n\\setlength{\\tabcolsep}{3.3pt}\n\\begin{tabular}{'+spec+'}\n\\toprule\n'+' & '.join(esc(h) for h in headers)+r' \\'+ '\n\\midrule\n'+''.join(' & '.join(esc(x) for x in row)+r' \\'+'\n' for row in rows)+'\\bottomrule\n\\end{tabular}\n\\end{'+env+'}\n'
 blocks[token]=(md,latex);tables.append({'table':token,'caption':caption})

table('DATASETS','Datasets and training conditions. Row and feature counts are source values before cleaning.', ['Dataset (OpenML ID)','Task','Rows','Features','Conditions'],[
 ['White Wine (44971)','Reg.','4,898','11','Base, Noisy'],['Superconductivity (44964)','Reg.','21,263','81','Base, Small, Noisy'],['Communities/Crime (46286)','Reg.','1,994','122','Base, Noisy'],['Bank Marketing (1461)','Class.','45,211','16','Base, Small, Noisy'],['Bioresponse (4134)','Class.','3,751','1,776','Base, Noisy'],['Credit Default (42477)','Class.','30,000','23','Base, Small, Noisy'],['Four synthetic generators','Both','2,000 each','20','Base']],wide=True)
space_rows=[['Linear / regression','alpha: 0.0001-100 (log); l1_ratio: 0.05-1'],['Linear / classification','C: 0.0001-100 (log); l1_ratio: 0-1; class_weight: none/balanced'],['SVM / both','C: 0.001-1000 (log); gamma: 0.00001-1 (log)'],['SVM / task-specific','epsilon: 0.001-1 (log, regression); class_weight: none/balanced (classification)'],['LightGBM / structure','num_leaves: 8-128; max_depth: -1, 3, 5, 8, 12; min_child_samples: 5-100'],['LightGBM / sampling','subsample: 0.5-1; colsample_bytree: 0.5-1'],['LightGBM / boosting','learning_rate: 0.01-0.3 (log); n_estimators: 50-500'],['LightGBM / regularization','reg_alpha and reg_lambda: 0.000001-10 (log); min_split_gain: 0-1']]
table('SPACES','Hyperparameter ranges used by the searches. For halving, LightGBM tree count is set by the resource stage.', ['Model / group','Parameters and bounds'],space_rows,widths=r'p{0.23\textwidth}p{0.70\textwidth}',wide=True)
cov=coverage.groupby('dataset')[['paired_seeds','expected_seeds']].sum()
table('COVERAGE','Available comparisons with all four optimizers in the main analysis group. Dataset counts include all model cells and budgets.', ['Dataset','Available / planned','Coverage'],[[TITLES[d],f'{int(r.paired_seeds)} / {int(r.expected_seeds)}',f'{100*r.paired_seeds/r.expected_seeds:.1f}%'] for d,r in cov.iterrows()])
rank_rows=[]
for m in MODELS:
 for b in [25,100]:rank_rows.append([MN[m],b]+[f'{ranks[(ranks.model==m)&(ranks.budget==b)&(ranks.optimizer==o)].mean_rank.iloc[0]:.3f}' for o in OPS])
table('RANKS','Average test ranks after grouping by dataset, with seven blocks per model-budget panel. Lower ranks are better.', ['Model','B','Grid','Random','TPE','Halving'],rank_rows)
examples=[('synthetic_linear','elastic_net'),('synthetic_linear','lightgbm'),('synthetic_friedman','lightgbm'),('superconductivity','lightgbm'),('credit_default','elastic_net'),('bioresponse','lightgbm')]
raw_rows=[]
for d,m in examples:
 sub=perf[(perf.dataset==d)&(perf.model==m)&(perf.condition=='base')&(perf.budget==100)].set_index('optimizer')
 for j,o in enumerate(OPS):
  raw_rows.append([TITLES[d]+' / '+MN[m] if j==0 else '',NAMES[o],int(sub.loc[o,'seeds']),f"{sub.loc[o,'mean_test_score']:.5f}",f"{sub.loc[o,'sd_test_score']:.5f}"])
table('RAW','Example Base-condition test scores at budget 100. Mean and standard deviation (SD) use the available seeds. Regression scores are RMSE; classification scores are ROC-AUC.', ['Dataset / model','Optimizer','Seeds','Mean','SD'],raw_rows)
time_rows=[]
for m in MODELS:
 for o in OPS:
  row=trade[(trade.model==m)&(trade.budget==100)&(trade.optimizer==o)].iloc[0]
  time_rows.append([MN[m],NAMES[o],f'{row.median_time_ratio:.2f}',f'{100*row.pareto_fraction:.1f}%',int(row.comparisons)])
table('TIME','Median runtime relative to random search at budget 100, and the percentage of matched groups where each method is Pareto efficient.', ['Model','Optimizer','Time / RS','Pareto','Groups'],time_rows)
# Absolute recorded compute is workload accounting, not an optimizer speed ranking.
all_hours=runtime_summary['all_recorded_executions']['hours'];primary_hours=runtime_summary['primary_paper_snapshot']['hours']
rows=[[TITLES[row.dataset],f'{row.hours:.2f}',f'{row.primary_hours:.2f}',f'{row.share_percent:.2f}%'] for row in runtime_dataset.sort_values('hours',ascending=False).itertuples()]
rows.append(['Total',f'{all_hours:.2f}',f'{primary_hours:.2f}','100.00%'])
table('TOTAL_DATASET','Recorded search time by dataset, including all conditions, models, budgets, and seeds. Main refers to the group used for performance analysis; All also includes older code versions. Duplicate copies are counted once.', ['Dataset','All hours','Main hours','Share of all'],rows)
rows=[[NAMES[row.optimizer],int(row.complete_records),f'{row.hours:.2f}',f'{row.primary_hours:.2f}',f'{row.share_percent:.2f}%'] for row in runtime_optimizer.sort_values('hours',ascending=False).itertuples()]
rows.append(['Total',runtime_summary['completed_recorded_executions']['execution_records'],f'{all_hours:.2f}',f'{primary_hours:.2f}','100.00%'])
table('TOTAL_OPTIMIZER','Recorded search time by optimizer. Counts are unique completed runs. Grid and halving run separately at each budget; random search and TPE share their histories.', ['Search algorithm','Runs','All hours','Main hours','Share of all'],rows)
rows=[[MN[row.model],int(row.complete_records),f'{row.hours:.2f}',f'{row.primary_hours:.2f}',f'{row.share_percent:.2f}%'] for row in runtime_model.sort_values('hours',ascending=False).itertuples()]
rows.append(['Total',runtime_summary['completed_recorded_executions']['execution_records'],f'{all_hours:.2f}',f'{primary_hours:.2f}','100.00%'])
table('TOTAL_MODEL','Recorded search time by model family. Workloads differ, and SVM is excluded from the larger Base and Noisy conditions.', ['Model family','Runs','All hours','Main hours','Share of all'],rows)
table('TIMELINE','Completed milestones and the proposed schedule for finishing the project.', ['Period','Milestone / next action'],[['Week 6 / Sep. 26','Scope and team coordination meeting (documented).'],['By Oct. 30','Runner, saved experiments, and executed analysis implemented.'],['Oct. 30-Nov. 1','Check report claims and submit the midterm PDF with cover letter.'],['Next course weeks','Resolve feasible coverage/convergence issues and review grid/resource semantics.'],['Weeks 13-14','Refresh a frozen snapshot; review results, limitations, and references.'],['Week 15','Finalize the paper and team presentation.']],widths=r'p{0.20\columnwidth}p{0.71\columnwidth}')

def figure(token,name,caption):
 md=f'![{caption}](figures/{name}.png)\n\n**Figure {len([x for x in blocks if x.startswith("FIG_")])+1}.** {caption}\n'
 tex=r'\begin{figure}[!htbp]'+'\n'+r'\centering'+'\n'+r'\includegraphics[width=\textwidth]{figures/'+name+'.pdf}\n'+r'\caption{'+esc(caption)+'}\n'+r'\label{fig:'+name+'}\n'+r'\end{figure}'+'\n'
 blocks[token]=(md,tex)
figure('FIG_RANKS','ranks','Average optimizer ranks across seven dataset blocks. Some cells have missing seeds. A rank does not show how large the score difference is.')
figure('FIG_BUDGET','budget','Test-score changes from budget 25 to 100, relative to the magnitude of the budget-25 score. Positive values mean improvement. Boxplots use the available matched groups, with outlier markers hidden. Relative AUC and RMSE changes have different interpretations.')
figure('FIG_TIME','runtime','Median search time relative to random search at budget 100. The dashed line marks equal time. Our halving schedule does not generally save elapsed time.')
figure('FIG_ROBUST','robustness','Changes in clean-test performance after altering the training data. Positive values mean worse performance. Both budgets and available model cells are included; outlier markers are hidden. The Small regression comparisons use Superconductivity only.')
figure('FIG_TRAJECTORY','trajectory','Best validation RMSE so far for Friedman regression with LightGBM, seed 1, budget 100. Only full-resource trials update the score, but time and resource totals include earlier low-resource work. This is one example, not a comparison of convergence rates across datasets.')
refs=[
('bergstra12','J. Bergstra and Y. Bengio, "Random Search for Hyper-Parameter Optimization," Journal of Machine Learning Research, vol. 13, pp. 281-305, 2012.','https://jmlr.org/papers/v13/bergstra12a.html'),
('bergstra11','J. Bergstra, R. Bardenet, Y. Bengio, and B. Kegl, "Algorithms for Hyper-Parameter Optimization," Advances in Neural Information Processing Systems, vol. 24, 2011.','https://proceedings.neurips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc635690-Abstract.html'),
('jamieson16','K. Jamieson and A. Talwalkar, "Non-stochastic Best Arm Identification and Hyperparameter Optimization," Proceedings of AISTATS, PMLR, vol. 51, pp. 240-248, 2016.','https://proceedings.mlr.press/v51/jamieson16.html'),
('li18','L. Li, K. Jamieson, G. DeSalvo, A. Rostamizadeh, and A. Talwalkar, "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization," Journal of Machine Learning Research, vol. 18, no. 185, pp. 1-52, 2018.','https://jmlr.org/papers/v18/16-558.html'),
('probst19','P. Probst, A.-L. Boulesteix, and B. Bischl, "Tunability: Importance of Hyperparameters of Machine Learning Algorithms," Journal of Machine Learning Research, vol. 20, no. 53, pp. 1-32, 2019.','https://jmlr.org/papers/v20/18-444.html'),
('bischl21','B. Bischl et al., "Hyperparameter Optimization: Foundations, Algorithms, Best Practices and Open Challenges," arXiv:2107.05847v3, 2021.','https://arxiv.org/abs/2107.05847'),
('demsar06','J. Demsar, "Statistical Comparisons of Classifiers over Multiple Data Sets," Journal of Machine Learning Research, vol. 7, pp. 1-30, 2006.','https://jmlr.org/papers/v7/demsar06a.html'),
('ke17','G. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," Advances in Neural Information Processing Systems, vol. 30, 2017.','https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html'),
('akiba19','T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A Next-generation Hyperparameter Optimization Framework," Proceedings of KDD, 2019; arXiv:1907.10902.','https://arxiv.org/abs/1907.10902'),
('pedregosa11','F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.','https://jmlr.org/papers/v12/pedregosa11a.html'),
('grinsztajn22','L. Grinsztajn, E. Oyallon, and G. Varoquaux, "Why do tree-based models still outperform deep learning on tabular data?" Advances in Neural Information Processing Systems, vol. 35, pp. 507-520, 2022.','https://arxiv.org/abs/2207.08815')]
blocks['REFERENCES']=('\n\n'.join(f'[{i}] {text} [Publisher / paper]({url}).' for i,(_,text,url) in enumerate(refs,1)), '\\begin{thebibliography}{11}\n\\interlinepenalty=10000\n'+''.join('\\bibitem{'+key+'} '+esc(text)+' \\href{'+url+'}{Online}.\n' for key,text,url in refs)+'\\end{thebibliography}\n')
source=(ROOT/'paper_source.md').read_text()
cover='''# Budgeted Hyperparameter Optimization on Tabular Data: Predictive Performance, Computational Cost, and Robustness

**MAP 6114 Machine Learning**

**Team: Optima Prime**

Danay Fernandes Alfonso  
Dilek Ozdemirci  
Ecil Teodoro

---

'''
md=cover+source
for key,(m,t) in blocks.items():md=md.replace('@@'+key+'@@',m)
assert '@@' not in md
(ROOT/'report.md').write_text(md)
# Extract abstract and convert the shared narrative to LaTeX; raw figure/table blocks
# survive Pandoc conversion and preserve the compact table layout.
abstract=source.split('## Abstract\n\n',1)[1].split('\n\n**Index terms:',1)[0]
body='## I. Introduction'+source.split('## I. Introduction',1)[1]
body=re.sub(r'^## [IVX]+\. ', '## ',body,flags=re.M)
body=re.sub(r'^### [A-Z]\. ', '### ',body,flags=re.M)
body=body.replace('## References\n\n@@REFERENCES@@','@@REFERENCES@@')
for key,(m,t) in blocks.items():body=body.replace('@@'+key+'@@',t)
converted=subprocess.run(['pandoc','--from=markdown+raw_tex','--to=latex','--shift-heading-level-by=-1','--wrap=none'],input=body,text=True,check=True,capture_output=True).stdout
converted=converted.replace('\\section{Appendix A. Reproducibility and Evidence Map}', '\\appendix\n\\renewcommand{\\thesection}{Appendix~\\Alph{section}}\n\\section{Reproducibility and Evidence Map}')
# Long hashes and artifact paths should line-break instead of overflowing a column.
converted=re.sub(r'\\texttt\{([^{}]+)\}',lambda m:r'\path{'+m.group(1).replace(r'\_','_')+'}',converted)
# Turn numbered literature references into numbered bibliography citations without changing MD.
for i,(key,_,_) in enumerate(refs,1):converted=converted.replace(f'[{i}]',r'\cite{'+key+'}').replace('{[}'+str(i)+'{]}',r'\cite{'+key+'}')
abs_tex=subprocess.run(['pandoc','-f','markdown','-t','latex','--wrap=none'],input=abstract,text=True,check=True,capture_output=True).stdout.strip()
title=source.splitlines()[0][2:]
header=r'''\documentclass[12pt,letterpaper]{article}
\usepackage[margin=1in,headheight=15pt]{geometry}
\usepackage{newtxtext,newtxmath}
\usepackage{setspace}
\usepackage[section]{placeins}
\usepackage[font=normalsize,labelfont=bf]{caption}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,graphicx,booktabs,array}
\usepackage{microtype}
\usepackage{xurl}
\usepackage[hidelinks]{hyperref}
\urlstyle{tt}
\newcommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\setlength{\emergencystretch}{1.5em}
\interdisplaylinepenalty=2500
\raggedbottom
\graphicspath{{figures/}}
\hypersetup{pdftitle={Budgeted Hyperparameter Optimization on Tabular Data},pdfauthor={Optima Prime}}
\renewcommand{\thesection}{\Roman{section}}
\renewcommand{\thesubsection}{\Alph{subsection}}
\makeatletter
\renewcommand{\@seccntformat}[1]{\csname the#1\endcsname.\quad}
\makeatother
\renewcommand{\thetable}{\Roman{table}}
\renewenvironment{abstract}{\section*{Abstract}}{\par}
\pagestyle{plain}
\setlength{\parindent}{0.25in}
\setlength{\parskip}{0pt}
\doublespacing
\begin{document}
\thispagestyle{empty}
\begin{center}
\singlespacing
\vspace*{1.1in}
{\LARGE\bfseries Budgeted Hyperparameter Optimization\\
on Tabular Data:\\[0.15in]
Predictive Performance, Computational Cost,\\
and Robustness\par}
\vspace{0.65in}
{\Large MAP 6114 Machine Learning\par}
\vspace{0.45in}
{\large\bfseries Team: Optima Prime\par}
\vspace{0.3in}
{\large Danay Fernandes Alfonso\par}
\vspace{0.1in}
{\large Dilek Ozdemirci\par}
\vspace{0.1in}
{\large Ecil Teodoro\par}
\end{center}
\clearpage
\setcounter{page}{1}
\title{\bfseries\fontsize{18}{22}\selectfont '''+esc(title)+r'''}
\author{Danay Fernandes Alfonso, Dilek Ozdemirci, and Ecil Teodoro\\
\normalsize Team Optima Prime, MAP 6114 Machine Learning}
\date{}
\begingroup
\singlespacing
\maketitle
\endgroup
\begin{abstract}
'''+abs_tex+r'''
\end{abstract}
\noindent\textbf{Keywords:} 
Hyperparameter optimization, random search, Bayesian optimization, successive halving, tabular learning, computational budgets.
\par
'''
(ROOT/'report.tex').write_text(header+converted+'\n\\end{document}\n')
for _ in range(2):
 subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory=build','report.tex'],cwd=ROOT,check=True,stdout=(BUILD/'compile_stdout.txt').open('w'),stderr=subprocess.STDOUT)
shutil.copy2(BUILD/'report.pdf',ROOT/'report.pdf')
inputs=[ROOT/'paper_source.md',ROOT/'scan_runtime.py',Path(__file__),*sorted(DATA.glob('*'))]
(ROOT/'build_manifest.json').write_text(json.dumps({'course':'MAP 6114 Machine Learning','instructor':'Dr. Jia Liu','team':'Optima Prime','hardware':runtime_summary['hardware'],'recorded_optimization_hours':runtime_summary['all_recorded_executions']['hours'],'primary_recorded_optimization_hours':runtime_summary['primary_paper_snapshot']['hours'],'format':{'body_font_points':12,'line_spacing':'double','columns':1,'margins_inches':1},'snapshot_date':'2026-10-30','source_cohort':'5f5b6e60da15ed1c5be07110bdfa0c1147b0502ee46a819fca83e9e5d938fdb4','input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'outputs':['report.md','report.tex','report.pdf'],'figure_data':'data/*.csv','cited_sources':[url for _,_,url in refs]},indent=2)+'\n')
print('Built report.md, report.tex, and report.pdf.')
