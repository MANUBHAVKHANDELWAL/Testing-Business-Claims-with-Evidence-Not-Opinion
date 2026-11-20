"""RetailStat AI — reproducible advanced statistics project.
Run from the project directory: python analysis.py
All 7,084 supplied rows are considered; no random sample is selected.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; OUT=ROOT/'outputs'; CHARTS=OUT/'charts'
for p in (DATA,OUT,CHARTS):p.mkdir(parents=True,exist_ok=True)
BLUE='#2563EB'; NAVY='#172554'; TEAL='#0D9488'; AMBER='#F59E0B'; RED='#DC2626'; GREY='#64748B'; BG='#F8FAFC'
plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':BG,'axes.edgecolor':'#CBD5E1','axes.labelcolor':NAVY,'text.color':NAVY,'xtick.color':GREY,'ytick.color':GREY,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':160})

def chart(name,title,draw):
 fig,ax=plt.subplots(figsize=(9,5));draw(ax);ax.set_title(title,loc='left',weight='bold',fontsize=15,pad=14);fig.tight_layout();fig.savefig(CHARTS/(name+'.png'),bbox_inches='tight');plt.close(fig)

def clean_data(frame):
 x=frame.drop_duplicates(subset='Transaction_ID',keep='first').copy()
 x['Region']=x.Region.astype(str).str.strip().str.title().replace({'Nort':'North','Sth':'South'})
 x['Gender']=x.Gender.astype(str).str.strip().str.title().replace({'M':'Male','F':'Female'})
 for col in ['Order_Date','Delivery_Date']:
  x[col]=pd.to_datetime(x[col],errors='coerce',format='mixed')
 x.loc[x.Delivery_Date<x.Order_Date,'Delivery_Date']=pd.NaT
 x.loc[~x.Age.between(18,100),'Age']=np.nan
 x.loc[~x.Discount.between(0,0.5),'Discount']=np.nan
 x.loc[(x.Income<=0)|(x.Income>500000),'Income']=np.nan

 return x

def main():
 original=pd.read_csv(DATA/'retail_customer_sales_dataset.csv')
 x=clean_data(original);x.to_csv(DATA/'retail_sales_cleaned.csv',index=False)
 numeric=['Sales','Profit','Age','Income']
 desc=x[numeric].agg(['count','mean','median','std','var','min','max']).T
 desc['mode']=x[numeric].mode(dropna=True).iloc[0]
 for pct in [25,75]:desc[f'p{pct}']=x[numeric].quantile(pct/100)
 desc['IQR']=desc.p75-desc.p25
 desc.round(3).to_csv(OUT/'descriptive_statistics.csv')
 # CI: t-based interval for a mean across observed transaction records
 sales=x.Sales.dropna();n=len(sales);m=sales.mean();sem=stats.sem(sales)
 ci=stats.t.interval(0.95,df=n-1,loc=m,scale=sem)
 north=x.loc[x.Region=='North','Sales'].dropna();south=x.loc[x.Region=='South','Sales'].dropna()
 tt=stats.ttest_ind(north,south,equal_var=False)
 groups=[x.loc[x.Region==region,'Sales'].dropna() for region in ['North','South','East','West']]
 anova=stats.f_oneway(*groups)
 ct=pd.crosstab(x.Customer_Segment,x.Product_Category)
 chi,pchi,dof,exp=stats.chi2_contingency(ct)
 corr_data=x[['Discount','Profit']].dropna()
 r,pr=stats.pearsonr(corr_data.Discount,corr_data.Profit)
 s1=x[['Sales','Quantity']].dropna(); model1=sm.OLS(s1.Sales,sm.add_constant(s1[['Quantity']])).fit(cov_type='HC3')
 s2=x[['Sales','Quantity','Discount','Income']].dropna();model2=sm.OLS(s2.Sales,sm.add_constant(s2[['Quantity','Discount','Income']])).fit(cov_type='HC3')
 results={'raw_rows':len(original),'unique_transactions':len(x),'duplicates_removed':len(original)-len(x),'invalid_age_or_missing_after_clean':int(x.Age.isna().sum()),'invalid_or_missing_discount':int(x.Discount.isna().sum()),'invalid_or_missing_income':int(x.Income.isna().sum()),'mean_sales':float(m),'ci95_sales':[float(ci[0]),float(ci[1])],'confidence_n':n,'region_means':{k:float(x.loc[x.Region==k,'Sales'].mean()) for k in ['North','South','East','West']},'ttest':{'statistic':float(tt.statistic),'pvalue':float(tt.pvalue),'n_north':len(north),'n_south':len(south),'method':'Welch independent two-sample t-test'},'anova':{'statistic':float(anova.statistic),'pvalue':float(anova.pvalue),'method':'one-way ANOVA'},'chi_square':{'statistic':float(chi),'pvalue':float(pchi),'dof':int(dof),'min_expected':float(exp.min()),'method':'Pearson chi-square independence test'},'correlation':{'coefficient':float(r),'pvalue':float(pr),'n':len(corr_data),'method':'Pearson correlation'},'simple_regression':{'n':len(s1),'r2':float(model1.rsquared),'coefficients':{k:float(v) for k,v in model1.params.items()},'pvalues':{k:float(v) for k,v in model1.pvalues.items()}},'multiple_regression':{'n':len(s2),'r2':float(model2.rsquared),'coefficients':{k:float(v) for k,v in model2.params.items()},'pvalues':{k:float(v) for k,v in model2.pvalues.items()}},'alpha':0.05,'note':'Full supplied dataset; no random sampling, no causal claims. OLS HC3 heteroskedasticity-robust standard errors; observational and potentially repeated customers.'}
 (OUT/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
 ct.to_csv(OUT/'segment_category_contingency.csv')
 with open(OUT/'statistical_results.txt','w',encoding='utf-8') as f:
  f.write('RetailStat AI | Statistical execution results\nAll valid provided records analyzed; no random subset.\n\n')
  f.write(json.dumps(results,indent=2));f.write('\n\nSIMPLE MODEL\n'+model1.summary().as_text()+'\n\nMULTIPLE MODEL\n'+model2.summary().as_text())
 # Consistent charts; no charts sample points randomly
 chart('01_sales_histogram','Univariate | Sales distribution',lambda a:(a.hist(x.Sales,bins=35,color=BLUE,edgecolor='white'),a.set_xlabel('Sales (dataset units)'),a.set_ylabel('Transactions')))
 chart('02_profit_boxplot','Univariate | Profit distribution',lambda a:(a.boxplot(x.Profit,patch_artist=True,boxprops={'facecolor':TEAL},medianprops={'color':NAVY}),a.set_ylabel('Profit (dataset units)'),a.set_xticks([1],['Profit'])))
 chart('03_sales_by_region','Bivariate | Average sales by region',lambda a:(a.bar(list(results['region_means']),list(results['region_means'].values()),color=BLUE),a.set_ylabel('Mean sales (dataset units)'),a.set_xlabel('Region')))
 chart('04_sales_region_box','Bivariate | Sales spread across regions',lambda a:(a.boxplot(groups,patch_artist=True,boxprops={'facecolor':TEAL},medianprops={'color':NAVY},showfliers=False),a.set_xticks(range(1,5),['North','South','East','West']),a.set_ylabel('Sales (dataset units); outliers hidden visually')))
 chart('05_category_segment_heatmap','Bivariate | Segment × category transaction counts',lambda a:draw_heatmap(a,ct.values,ct.columns.tolist(),ct.index.tolist(),'Count'))
 chart('06_discount_profit_scatter','Bivariate | Discount versus profit',lambda a:(a.scatter(corr_data.Discount,corr_data.Profit,s=7,color=BLUE,alpha=.17),a.set_xlabel('Discount fraction'),a.set_ylabel('Profit (dataset units)')))
 chart('07_simple_regression','Bivariate | Quantity and predicted sales',lambda a:(a.scatter(s1.Quantity,s1.Sales,s=7,color=BLUE,alpha=.12),a.plot(np.arange(1,11),model1.predict(sm.add_constant(pd.DataFrame({'Quantity':np.arange(1,11)}))),color=TEAL,lw=3),a.set_xlabel('Quantity'),a.set_ylabel('Sales (dataset units)')))
 chart('08_multiple_regression','Multivariate | Predicted versus actual sales',lambda a:(a.scatter(model2.fittedvalues,s2.Sales,s=8,color=BLUE,alpha=.18),a.plot([0,max(s2.Sales)],[0,max(s2.Sales)],color=TEAL,lw=2),a.set_xlabel('Predicted sales'),a.set_ylabel('Actual sales')))
 chart('09_correlation_heatmap','Multivariate | Numerical variable correlations',lambda a:draw_corr(a,x[['Sales','Profit','Quantity','Discount','Income']].corr()))
 chart('10_residuals','Regression diagnostic | Residuals versus fitted values',lambda a:(a.scatter(model2.fittedvalues,model2.resid,s=7,alpha=.18,color=BLUE),a.axhline(0,color=TEAL,lw=2),a.set_xlabel('Predicted sales'),a.set_ylabel('Residual (actual − predicted)')))
 chart('11_income_histogram','Univariate | Income distribution after validation',lambda a:(a.hist(x.Income.dropna(),bins=35,color=BLUE,edgecolor='white'),a.set_xlabel('Annual income (dataset units)'),a.set_ylabel('Transactions')))
 chart('12_segment_category_stacked','Multivariate | Product category mix by segment',lambda a:draw_stacked(a,ct))
 print('Analyzed',len(x),'unique transactions; charts:',len(list(CHARTS.glob('*.png'))));print('T-test p',tt.pvalue,'ANOVA p',anova.pvalue,'chi p',pchi,'correlation',r,pr);return results

def draw_heatmap(ax,array,cols,rows,label):
 im=ax.imshow(array,cmap='Blues',aspect='auto');ax.set_xticks(range(len(cols)),cols,rotation=25,ha='right');ax.set_yticks(range(len(rows)),rows)
 for i in range(len(rows)):
  for j in range(len(cols)):ax.text(j,i,str(array[i,j]),ha='center',va='center',fontsize=9,color='white' if array[i,j]>.65*array.max() else NAVY)
 ax.figure.colorbar(im,ax=ax,label=label)

def draw_corr(ax,data):
 im=ax.imshow(data.values,cmap='PuBuGn',vmin=-1,vmax=1);ax.set_xticks(range(len(data)),list(data),rotation=35,ha='right');ax.set_yticks(range(len(data)),list(data.index))
 for i in range(len(data)):
  for j in range(len(data)):ax.text(j,i,f'{data.iloc[i,j]:.2f}',ha='center',va='center',fontsize=9,color=NAVY)
 ax.figure.colorbar(im,ax=ax,label='Pearson r')

def draw_stacked(ax,ct):
 q=ct.div(ct.sum(axis=1),axis=0)*100;bottom=np.zeros(len(q));colors=[NAVY,BLUE,TEAL,'#67B7AE',AMBER]
 for i,c in enumerate(q.columns):ax.bar(q.index,q[c],bottom=bottom,label=c,color=colors[i]);bottom+=q[c].to_numpy()
 ax.set_ylabel('Share of segment (%)');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3,frameon=False)

if __name__=='__main__':main()
