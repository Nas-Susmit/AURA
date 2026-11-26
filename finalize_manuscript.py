from pathlib import Path
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('AURA_Revised.tex'); s=p.read_text()
def rep(a,b):
 global s
 if a not in s: raise ValueError(a)
 s=s.replace(a,b)
rep(r'\usepackage{cite}',r'''\usepackage{cite}
\newcommand{\doi}[1]{\href{https://doi.org/#1}{doi: \nolinkurl{#1}}}''')
rep(r'\author{\IEEEauthorblockN{Anonymous author(s)}\IEEEauthorblockA{Author details withheld for review}}',r'''\author{\IEEEauthorblockN{Susmit Naskar}\IEEEauthorblockA{Department of Computer Science and Engineering\\Institute of Engineering and Management, Kolkata, India}}''')
# Add a real named supplementary artifact rather than an invented repository URL.
rep('The accompanying source package includes the deterministic generator, original and revision experiment scripts, table/figure generation script, one generated population, per-seed metrics, sensitivity, permutation and attribution-stability files, audit files and literature matrix.',r'The AURA research package (\texttt{AURA\_Final\_Research\_Package.zip}) accompanies this manuscript. It contains the generator, experiment and figure scripts, one generated population, per-seed results, sensitivity and attribution analyses, and literature matrix. A public repository URL is not yet available.')
a=s.index('OpenAI ChatGPT (GPT-6.2)'); b=s.index('\n\\begin{thebibliography}',a)
s=s[:a]+'''OpenAI ChatGPT was used extensively to draft and revise the Abstract, Sections I--VIII, Data and Code Availability, and this acknowledgment; to help identify and check references; and to generate the LaTeX, Fig.~1, and Python code for the simulator, analyses, Tables IV--XI and Figs.~2--3. The tool also assisted in preparing Tables I--III. This assistance included research design, implementation, interpretation and writing, not only language editing. The human author is responsible for verifying the references, code, assumptions, results and conclusions and approving the submitted work.'''+s[b:]
# Language and substantive framing.
rep('This study uses AURA as an auditable test bed rather than claiming a novel state-of-the-art architecture. Its contribution is an investigation of the limits of a methodological claim:', 'AURA is a test bed for examining whether disagreement improves selective prediction. This study makes four contributions:')
rep('A negative result is informative precisely because a plausible mechanism need not improve selective prediction.', 'The experiments test whether a plausible uncertainty measure offers additional predictive value.')
rep('nearest methodological and clinical precedents identified in a targeted scoping search','most relevant studies found in a targeted literature search')
rep('44 publications/resources','44 publications and resources')
rep('Our specific question is whether weighted prediction dispersion adds incremental error-ranking value on an explicit axSpA-inspired stress benchmark after controlling for confidence, availability and quality.', 'We test whether weighted dispersion improves error ranking on an axSpA-inspired benchmark after accounting for confidence, availability and quality.')
rep('The inverse predicted error is a transparent heuristic:', 'Inverse predicted error is used as a weighting heuristic:')
rep('The central result is a limit on the methodological claim: a learned ranker improves integrated risk for late fusion, but weighted dispersion contributes no clearly established additional benefit after matching confidence, quality and availability.', 'The main finding is that learned ranking reduces integrated risk for late fusion, but weighted dispersion does not show a clear additional benefit once confidence, quality and availability are included.')
rep("The gate's small gains against equal and quality weighting are supported by paired intervals, but are not evidence of clinical importance.", "The gate's small gains over equal and quality weighting are supported by paired intervals, but have no established clinical significance. Early logistic regression is the simpler choice for this benchmark; the generator's linear links may favor it. Late fusion may still be useful when imaging and laboratory models are maintained separately, when inputs arrive at different times, or when data-sharing restrictions prevent raw feature concatenation. These are possible engineering motivations, not benefits demonstrated by the experiments or guarantees of privacy.")
rep('All data therefore share a limited model of the world.', 'The results are therefore specific to the assumptions of the generator.')
rep('This is a lightweight exploratory study with no preregistration.', '')
rep('These results constrain the methodological claim that disagreement is independently useful: neither its benefit nor its absence in other settings is established.', 'These results do not establish an independent benefit of disagreement, nor do they rule out a benefit in other settings.')
rep('The study evaluates algorithmic behavior in this generator, not axSpA diagnostic performance, clinical evidence relationships or readiness.', 'The findings describe performance on simulated records, not diagnostic accuracy or clinical utility in axSpA.')
rep('These results are a negative methodological finding within an illustrative generator, not evidence about real axSpA diagnostic relationships or clinical effectiveness.', 'The negative finding is limited to this generator and does not establish how real axSpA diagnostic evidence behaves.')
rep('The DESIR cohort is a potential source of longitudinal diagnostic evidence, but access is subject to request and governance', 'The DESIR cohort has been used to examine changes in clinical diagnosis over time; access to its data requires a request and approval')
# math notation: standard indicator, unambiguous braces and spacing.
s=s.replace(r'\mathbb{1}',r'\mathbb{I}')
rep(r'\sum_km_k=0\ \text{or}\ u>\tau',r'\sum_{k} m_k=0\text{ or }u>\tau')
rep(r'\ell_k=\bigl(p_k-Y\bigr)^2',r'\ell_k=(p_k-Y)^2')
rep(r't_0=[H_2(\hat p),\sum_kw_kv_k,\tfrac{1}{3}\sum_km_k,\bar q,m_C,m_L,m_I]',r'\begin{aligned} t_0=[&H_2(\hat{p}),\ \sum_{k}w_k v_k,\ \tfrac{1}{3}\sum_{k}m_k,\\ &\bar{q},\ m_C,\ m_L,\ m_I] \end{aligned}')
rep(r'\bar q=\sum_k m_k q_k/\sum_km_k',r'\bar{q}=(\sum_{k}m_kq_k)/(\sum_{k}m_k)')
# Table I spacing, Table II numerics in math mode, Table III ensure prose separate.
rep(r'\begin{tabular}{@{}p{0.19\columnwidth}',r'\renewcommand{\arraystretch}{1.2}'+'\n'+r'\begin{tabular}{@{}p{0.19\columnwidth}')
a=s.index('Stiffness & B'); b=s.index(r'\bottomrule',a)
rows=s[a:b].splitlines(); new=[]
for row in rows:
 if not row.strip():continue
 row=row.removesuffix(r'\\'); parts=row.split(' & ')
 for i in range(2,len(parts)):
  v=parts[i].strip('$'); v=re.sub(r'^(-?)\.',r'\g<1>0.',v)
  if v in ['-1','-2']:v+='.0'
  parts[i]='$'+v+'$'
 new.append(' & '.join(parts)+r'\\')
s=s[:a]+'\n'.join(new)+s[b:]
rep(r'$q_k\sim\mathrm{Beta}(2,4)$; more observation noise',r'$q_k\sim\mathrm{Beta}(2,4)$; more observation noise')
rep(' & BRIER & ',' & Brier & ')
rep(r'H = entropy.',r'$H$ = entropy.')
rep(r'No $D$--H',r'No $D$--$H$')
rep('[-0.00162, -0.00000]',r'[$-0.001625$, $-0.000002$]')
rep('Shared features; unadjusted 95\\% intervals.', 'Shared features; unadjusted 95\\% intervals. Six decimals are shown for the RF interval near zero.')
rep('clip=.01','clip=0.01');rep('clip=.10','clip=0.10');rep('C=.1','C=0.1')
# Table XI: compact stacked rows with four decimals instead of slash-compressed columns.
a=s.index(r'\begin{table}[t]'+'\n'+r'\caption{Expert performance');b=s.index(r'\end{table}',a)+len(r'\end{table}')
ex=pd.read_csv('revision_results/experts.csv').groupby('modality')[['auroc','auprc','raw_brier','brier','raw_ece','ece']].mean()
rows=[]
for label,(_,x) in zip(['Clinical','Laboratory','Imaging'],ex.iterrows()):
 rows.append(f'{label} & {x.auroc:.4f} & {x.auprc:.4f} & {x.raw_brier:.4f} & {x.brier:.4f}'+r'\\')
# Use two panels in same table, 4 decimals for all.
rows2=[f'{label} & {x.raw_ece:.4f} & {x.ece:.4f}'+r'\\' for label,(_,x) in zip(['Clinical','Laboratory','Imaging'],ex.iterrows())]
s=s[:a]+r'''\begin{table}[t]
\caption{Expert performance on available records. Calibration is fitted separately per expert.}
\label{tab:expert}
\centering\footnotesize
\renewcommand{\arraystretch}{1.12}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrrr@{}}\toprule
 & & & \multicolumn{2}{c}{Brier} & \multicolumn{2}{c}{ECE}\\\cmidrule(lr){4-5}\cmidrule(lr){6-7}
Expert & AUROC & AUPRC & Raw & Cal. & Raw & Cal.\\\midrule
'''+ '\n'.join(f'{label} & {x.auroc:.4f} & {x.auprc:.4f} & {x.raw_brier:.4f} & {x.brier:.4f} & {x.raw_ece:.4f} & {x.ece:.4f}'+r'\\' for label,(_,x) in zip(['Clinical','Laboratory','Imaging'],ex.iterrows()))+r'''
\bottomrule
\end{tabular}
\end{table}'''+s[b:]
rep('Mean AUROC across stress conditions, with labels matching the scenario table.', 'Mean AUROC across the seven conditions in Table~\\ref{tab:scen}.')
# Bibliography: verified metadata; clickable identifiers.
rep(r"``MRI in axial spondyloarthritis: understanding an `ASAS-positive MRI' and the ASAS classification criteria,'' 2022. [Online]. Available: \url{https://pmc.ncbi.nlm.nih.gov/articles/PMC9283193/}",r"``MRI in axial spondyloarthritis: understanding an `ASAS-positive MRI' and the ASAS classification criteria,'' \emph{Skeletal Radiol.}, vol. 51, pp. 1721--1730, 2022, doi: 10.1007/s00256-022-04018-4.")
rep('K. G. Hermann','K. G. Hermann')
rep('E. Gvozdenovic','E. Gvozdenovi\\'+'{\\v c}') if False else None
rep(r'\emph{Joint Bone Spine}, 2026,',r'\emph{Joint Bone Spine}, vol. 93, no. 3, art. 106036, 2026,')
rep(r'\emph{Diagnostics}, vol. 16, art. 1037',r'\emph{Diagnostics}, vol. 16, no. 7, art. 1037')
rep(r'\emph{Big Data Cogn. Comput.}, vol. 10, art. 367',r'\emph{Big Data Cogn. Comput.}, vol. 10, no. 11, art. 367')
rep(r"``HeMIS: Hetero-modal image segmentation,'' 2016, arXiv:1607.05194.",r"``HeMIS: Hetero-modal image segmentation,'' in \emph{Proc. MICCAI}, LNCS, vol. 9901, pp. 469--477, 2016, doi: 10.1007/978-3-319-46723-8_54.")
rep(r'\emph{Proc. CVPR}, 2022.',r'\emph{Proc. IEEE/CVF CVPR}, pp. 20707--20717, 2022, doi: 10.1109/CVPR52688.2022.02005.')
rep(r'\emph{Quant. Imaging Med. Surg.}, 2024,',r'\emph{Quant. Imaging Med. Surg.}, vol. 14, no. 8, pp. 5845--5860, 2024,')
rep(r'L. J. L. Lopez,',r'L. J. Lechuga Lopez,')
rep(r'PMLR, vol. 333, pp. 794--833, 2026.',r'PMLR, vol. 333, pp. 794--833, 2026. [Online]. Available: \url{https://proceedings.mlr.press/v333/lopez26a.html}')
rep(r'\bibitem{desir} ``Diagnosis',r'\bibitem{desir} A. Molto, C. Serrand, S. Alonso, F. Berenbaum, P. Claudepierre, B. Combe, L. Gossec, A. Ruyssen-Witrand, A. Saraux, D. Wendling, T. Lequerre, and M. Dougados, ``Diagnosis')
rep(r'\emph{RMD Open}, vol. 10, e004484',r'\emph{RMD Open}, vol. 10, no. 3, e004484')
rep(r'A. N. Nguyen',r'A. Nguyen')
# DOI formatting to real clickable links, allowing line breaks via nolinkurl.
s=re.sub(r'doi: (10\.[^\s]+?)(?=\.\n)',r'\\doi{\1}',s)
# Link non-DOI proceedings via the official landing pages.
links={'selective':'https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html','guo':'https://proceedings.mlr.press/v70/guo17a.html','ensembles':'https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html','ovadia':'https://proceedings.neurips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html','shap':'https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html'}
for k,url in links.items():
 s=re.sub(r'(\\bibitem\{'+k+r'\}[^\n]+)\n',lambda m:m.group(1)+r' \href{'+url+r'}{[Proceedings]}\n'.replace('\\n','\n'),s)
# Figure 3 regenerate directly from preserved data; results untouched.
d=pd.read_csv('revision_results/metrics.csv'); scenarios=['complete','mcar30','mcar50','quality_shift','selective_acquisition','mimic_shift','combined_shift']; labels=['Complete','MCAR30','MCAR50','Quality','Selective','Mimic','Combined']
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(3.45,2.5));z=d[d.ranking=='entropy'].groupby(['scenario','model']).auroc.mean().unstack().loc[scenarios,['Equal fusion','Quality fusion','Early logistic','AURA']];z.plot(ax=ax,marker='o',markersize=3,linewidth=1);ax.set_xticks(range(7),labels,rotation=26,ha='right');ax.set_ylabel('AUROC');ax.set_xlabel('Condition');ax.legend(frameon=False,fontsize=6);fig.tight_layout();fig.savefig('revision_figures/robustness.pdf');plt.close(fig)
Path('AURA_Final.tex').write_text(s)
