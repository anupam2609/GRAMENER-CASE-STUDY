#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')



# Loading data
loan_data=pd.read_csv('loan.csv',encoding = "ISO-8859-1",low_memory=False)



# Dropping all columns with only null values
loan_data_new=loan_data.dropna(axis=1,how='all')
loan_data_new.dtypes



#Remove columns with only one unique values
loan_data_new= loan_data_new.loc[:,loan_data_new.nunique()!=1]
loan_data_new.shape



# Drop columns with more than 50% null values
loan_data_new=loan_data_new.loc[:,round(loan_data_new.isnull().sum()/len(loan_data_new)*100,2)<50]
loan_data_new.shape



drop_columns=['loan_amnt','funded_amnt_inv','emp_title','earliest_cr_line','url','total_acc','open_acc','desc',
              'title','zip_code','issue_d','sub_grade','last_credit_pull_d','last_pymnt_d']
loan_data_new=loan_data_new.drop(drop_columns,axis=1)
loan_data_new.dtypes

drop_columns1=['total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee','last_pymnt_amnt']
loan_data_new=loan_data_new.drop(drop_columns1,axis=1)


# As we only want to find out potential defaults, we should remove 'current' from loan status
loan_data_new=loan_data_new[loan_data_new.loan_status !='Current']
loan_data_new=loan_data_new.loc[:,loan_data_new.nunique()!=1]



print(loan_data_new.dtypes)
print(loan_data_new.shape)


# Converting revol_util and int_rate into a numeric type
loan_data_new.revol_util=loan_data_new.revol_util.str.rstrip('%').astype('float')
loan_data_new.int_rate=loan_data_new.int_rate.str.rstrip('%').astype('float')

# Creating a profit and loss column
loan_data_new['PnL']=round((loan_data_new['total_pymnt']-loan_data_new['funded_amnt'])*100/loan_data_new['funded_amnt'],2)

# Creating a column whihc is ratio of funded amount and annual income
loan_data_new['loan_inc_ratio']=round(loan_data_new.funded_amnt*100/loan_data_new.annual_inc,0)

print(loan_data_new.groupby(['grade'])['PnL'].median())
print(loan_data_new[(loan_data_new['inq_last_6mths']<3)].groupby('grade')['PnL'].median())
print(loan_data_new[(loan_data_new['grade']!='G')|(loan_data_new['home_ownership']!='OWN')].groupby('grade')['PnL'].median())
print(loan_data_new[(loan_data_new['grade']!='G')|(loan_data_new['home_ownership']!='OWN')].groupby('grade')['PnL'].median())
print(loan_data_new[(loan_data_new['grade']!='G')|(loan_data_new['purpose']!='medical')].groupby('grade')['PnL'].median())
print(loan_data_new[(loan_data_new['grade']!='G')|(loan_data_new['purpose']!='renewable_energy')].groupby('grade')['PnL'].median())
print(loan_data_new[(loan_data_new['grade']!='G')|(loan_data_new['purpose']!='small_business')].groupby('grade')['PnL'].median())
print(loan_data_new[(loan_data_new['grade']!='G')|((loan_data_new['home_ownership']!='OWN')&(~loan_data_new['purpose'].isin(['small_business','medical','renewable_energy'])))].groupby('grade')['PnL'].median())
print(loan_data_new[(loan_data_new['grade']=='G')&(loan_data_new['emp_length']=='< 1 year')]['purpose'].value_counts())



print("% of data points left",round(loan_data_new.size/loan_data.size*100,2))
print("% defaults in the remaining data set:",round(sum(loan_data_new.loan_status=='Charged Off')*100/len(loan_data_new),1))




#Relationship of Funded Amount with defaults
plt.figure(figsize=(15,15))

plt.subplot(1,2,1)
plt.title('Default')
sns.boxplot(y=loan_data_new[loan_data_new.loan_status=='Charged Off'].PnL)


plt.subplot(1,2,2)
plt.title('Fully Paid')
sns.boxplot(y=loan_data_new[loan_data_new.loan_status=='Fully Paid'].PnL)
plt.show()


# Relationship of default with funded amount with a barplot
plt.figure(figsize=(15,15))

sns.barplot(x='loan_status',y='PnL',data=loan_data_new)
plt.xlabel("Loan Status")
plt.ylabel("Profit and Loss")

plt.title("Profit n Loss vs status relationship")

plt.show()


plt.figure(figsize=(15,15))

sns.barplot(x='loan_status',y='loan_inc_ratio',hue='purpose',data=loan_data_new)

plt.show()




# Relationship of default with term with a barplot
plt.figure(figsize=(15,15))

plt.subplot(1,2,1)
sns.barplot(x='term',y='PnL',data=loan_data_new,hue='grade')
plt.xlabel("Term")
plt.ylabel("Profit and Loss")

plt.title("Profit and Loss vs term relationship")


plt.subplot(1,2,2)
sns.countplot('term',hue='loan_status',data=loan_data_new)
plt.xlabel("Term")
plt.ylabel("Count")
plt.title("Term vs Default relationship")
plt.show()



# Relationship of default with term with a barplot
plt.figure(figsize=(15,30))

plt.subplot(2,1,1)
sns.barplot(y='emp_length',x='PnL',data=loan_data_new,hue='grade')
plt.xlabel("Emp Length")
plt.ylabel("Funded Amount")

plt.title("Loan amount vs Emp Length relationship")


plt.subplot(2,1,2)
sns.countplot(y='emp_length',hue='loan_status',data=loan_data_new)
plt.xlabel("Emp Length")
plt.ylabel("Count")
plt.title("Emp Length vs Default relationship")
plt.show()


# In[ ]:


plt.figure(figsize=(15,15))

plt.subplot(1,2,1)
sns.barplot(x='home_ownership',y='PnL',data=loan_data_new,hue='grade')
plt.xlabel("Home Ownership")
plt.ylabel("Profit and Loss")

plt.title("Funded amount vs Home Ownership relationship")


plt.subplot(1,2,2)
sns.countplot(x='home_ownership',hue='loan_status',data=loan_data_new)
plt.xlabel("Home Ownership")
plt.ylabel("Count")
plt.title("Home Ownership vs Default relationship")
plt.show()


# In[ ]:


plt.figure(figsize=(15,30))

plt.subplot(2,1,1)
sns.barplot(y='purpose',x='PnL',data=loan_data_new,hue='grade')
plt.ylabel("Purpose")
plt.xlabel("Profit and Loss")

plt.title("Funded amount vs Purpose relationship")


plt.subplot(2,1,2)
sns.countplot(y='purpose',hue='loan_status',data=loan_data_new)
plt.ylabel("Purpose")
plt.xlabel("Count")
plt.title("Purpose vs Default relationship")
plt.show()


# In[ ]:


plt.figure(figsize=(15,45))

plt.subplot(2,1,1)
sns.barplot(y='addr_state',x='PnL',data=loan_data_new)
plt.ylabel("State")
plt.xlabel("Profit and Loss")

plt.title("Funded amount vs Purpose relationship")


plt.subplot(2,1,2)
sns.countplot(y='addr_state',hue='loan_status',data=loan_data_new)
plt.ylabel("Purpose")
plt.xlabel("Count")
plt.title("Purpose vs Default relationship")
plt.show()


# In[ ]:


# Analize to see influence of loan amount and DTI on profitability of individual loans

plt.figure(figsize=(15,15))

sns.lmplot(x='dti',y='PnL',col='loan_status',hue='purpose',fit_reg=False,data=loan_data_new)

plt.show()

plt.figure(figsize=(15,15))

sns.lmplot(x='dti',y='PnL',row='grade',hue='loan_status',fit_reg=False,data=loan_data_new)

plt.show()

plt.figure(figsize=(15,15))

sns.lmplot(x='funded_amnt',y='PnL',hue='loan_status',row='grade',fit_reg=False,data=loan_data_new)

plt.show()

plt.figure(figsize=(15,15))

sns.lmplot(x='inq_last_6mths',y='PnL',hue='loan_status',row='grade',fit_reg=False,data=loan_data_new)

plt.show()


# In[ ]:


plt.figure(figsize=(15,45))


sns.lmplot(x='revol_util',y='PnL',col='loan_status',hue='purpose',fit_reg=False,data=loan_data_new)
plt.ylabel("Revol_util")
plt.xlabel("Profit and Loss")

plt.show()


# In[ ]:


# Profitibility vs %age default for employment length


D1=loan_data_new.groupby(['emp_length','loan_status']).agg({'PnL':'count'})
D2=D1.groupby(level=0).apply(lambda x:round(x*100/x.sum(),1))

# Creating a dataframe for default rate
default_purpose={'default_rate':{}}

for name in loan_data_new.emp_length.dropna().unique():
    default_purpose['default_rate'][name]=D2['PnL'][name]['Charged Off']
D2=pd.DataFrame(default_purpose)

# Merging default dataframe with median and mad
D1=loan_data_new.groupby(['emp_length']).agg({'PnL':['median','mad']})

D3=pd.merge(D1['PnL'],D2,left_index=True,right_index=True,how='inner')

print(D3.sort_values('default_rate'))

D4=loan_data_new.groupby(['emp_length','grade']).agg({'PnL':'median'})

median_profitability={}

for name in loan_data_new.emp_length.dropna().unique():
    median_profitability[name]={}
    for gr in loan_data_new.grade.dropna().unique():
        try:  
            median_profitability[name][gr]=D4['PnL'][name][gr]
        except:
            median_profitability[name][gr]=pd.NaT
print(pd.DataFrame(median_profitability))


# In[ ]:


# Profitibility vs %age default for Purpose


D1=loan_data_new.groupby(['purpose','loan_status']).agg({'PnL':'count'})
D2=D1.groupby(level=0).apply(lambda x:round(x*100/x.sum(),1))

# Creating a dataframe for default rate
default_purpose={'default_rate':{}}

for name in loan_data_new.purpose.dropna().unique():
    default_purpose['default_rate'][name]=D2['PnL'][name]['Charged Off']
D2=pd.DataFrame(default_purpose)

# Merging default dataframe with median and mad
D1=loan_data_new.groupby(['purpose']).agg({'PnL':['median','mad']})

D3=pd.merge(D1['PnL'],D2,left_index=True,right_index=True,how='inner').sort_values('default_rate')

print(D3)

D4=loan_data_new.groupby(['purpose','grade']).agg({'PnL':'median'})
#print(D4)
median_profitability={}

for name in loan_data_new.purpose.dropna().unique():
    median_profitability[name]={}
    for gr in loan_data_new.grade.dropna().unique():
        try:  
            median_profitability[name][gr]=D4['PnL'][name][gr]
        except:
            median_profitability[name][gr]=pd.NaT
print(pd.DataFrame(median_profitability))




D1=loan_data_new.groupby(['home_ownership','loan_status']).agg({'PnL':'count'})
D2=D1.groupby(level=0).apply(lambda x:round(x*100/x.sum(),1))

# Creating a dataframe for default rate
default_purpose={'default_rate':{}}

for name in loan_data_new.home_ownership.dropna().unique():
    try:
        default_purpose['default_rate'][name]=D2['PnL'][name]['Charged Off']
    except:
        default_purpose['default_rate'][name]=pd.NaT
D2=pd.DataFrame(default_purpose)

# Merging default dataframe with median and mad
D1=loan_data_new.groupby(['home_ownership']).agg({'PnL':['median','mad']})

D3=pd.merge(D1['PnL'],D2,left_index=True,right_index=True,how='inner').sort_values('default_rate')

print(D3)

D4=loan_data_new.groupby(['home_ownership','grade']).agg({'PnL':'median'})
median_profitability={}

for name in loan_data_new.home_ownership.dropna().unique():
    median_profitability[name]={}
    for gr in loan_data_new.grade.dropna().unique():
        try:  
            median_profitability[name][gr]=D4['PnL'][name][gr]
        except:
            median_profitability[name][gr]=pd.NaT
print(pd.DataFrame(median_profitability))




D1=loan_data_new.groupby(['addr_state','loan_status']).agg({'PnL':'count'})
D2=D1.groupby(level=0).apply(lambda x:round(x*100/x.sum(),1))

# Creating a dataframe for default rate
default_purpose={'default_rate':{}}

for name in loan_data_new.addr_state.dropna().unique():
    try:
        default_purpose['default_rate'][name]=D2['PnL'][name]['Charged Off']
    except:
        default_purpose['default_rate'][name]=pd.NaT
        
D2=pd.DataFrame(default_purpose)

# Merging default dataframe with median and mad
D1=loan_data_new.groupby(['addr_state']).agg({'PnL':['median','mad']})

D3=pd.merge(D1['PnL'],D2,left_index=True,right_index=True,how='inner').sort_values('default_rate')

print(D3)

D4=loan_data_new.groupby(['addr_state','grade']).agg({'PnL':'median'})
median_profitability={}

for name in loan_data_new.addr_state.dropna().unique():
    median_profitability[name]={}
    for gr in loan_data_new.grade.dropna().unique():
        try:  
            median_profitability[name][gr]=D4['PnL'][name][gr]
        except:
            median_profitability[name][gr]=pd.NaT
print(pd.DataFrame(median_profitability))


# In[ ]:


D1=loan_data_new.groupby(['verification_status','loan_status']).agg({'PnL':'count'})
D2=D1.groupby(level=0).apply(lambda x:round(x*100/x.sum(),1))

# Creating a dataframe for default rate
default_purpose={'default_rate':{}}

for name in loan_data_new.verification_status.dropna().unique():
    try:
        default_purpose['default_rate'][name]=D2['PnL'][name]['Charged Off']
    except:
        default_purpose['default_rate'][name]=pd.NaT
        
D2=pd.DataFrame(default_purpose)

# Merging default dataframe with median and mad
D1=loan_data_new.groupby(['verification_status']).agg({'PnL':['median','mad']})

D3=pd.merge(D1['PnL'],D2,left_index=True,right_index=True,how='inner').sort_values('default_rate')

print(D3)

D4=loan_data_new.groupby(['verification_status','grade']).agg({'PnL':'median'})
median_profitability={}

for name in loan_data_new.verification_status.dropna().unique():
    median_profitability[name]={}
    for gr in loan_data_new.grade.dropna().unique():
        try:  
            median_profitability[name][gr]=D4['PnL'][name][gr]
        except:
            median_profitability[name][gr]=pd.NaT
print(pd.DataFrame(median_profitability))


# In[ ]:


#loan_data_new['delinq_2yrs']=loan_data_new['delinq_2yrs'].astype('category')
D1=loan_data_new.groupby(['delinq_2yrs','loan_status']).agg({'PnL':'count'})
D2=D1.groupby(level=0).apply(lambda x:round(x*100/x.sum(),1))

#print(D1)
# Creating a dataframe for default rate
default_purpose={'default_rate':{}}

for name in loan_data_new.delinq_2yrs.dropna().unique():
    try:
        default_purpose['default_rate'][name]=D2['PnL'][name]['Charged Off']
    except:
        default_purpose['default_rate'][name]=pd.NaT
        
D2=pd.DataFrame(default_purpose)

# Merging default dataframe with median and mad
D1=loan_data_new.groupby(['delinq_2yrs']).agg({'PnL':['median','mad']})

D3=pd.merge(D1['PnL'],D2,left_index=True,right_index=True,how='inner').sort_values('default_rate')

print(D3)

D4=loan_data_new.groupby(['delinq_2yrs','grade']).agg({'PnL':'median'})
median_profitability={}

for name in loan_data_new.delinq_2yrs.dropna().unique():
    median_profitability[name]={}
    for gr in loan_data_new.grade.dropna().unique():
        try:  
            median_profitability[name][gr]=D4['PnL'][name][gr]
        except:
            median_profitability[name][gr]=pd.NaT
print(pd.DataFrame(median_profitability))


# In[ ]:


#loan_data_new['delinq_2yrs']=loan_data_new['delinq_2yrs'].astype('category')
D1=loan_data_new.groupby(['inq_last_6mths','loan_status']).agg({'PnL':'count'})
D2=D1.groupby(level=0).apply(lambda x:round(x*100/x.sum(),1))

#print(D1)
# Creating a dataframe for default rate
default_purpose={'default_rate':{}}

for name in loan_data_new.pub_rec_bankruptcies.dropna().unique():
    try:
        default_purpose['default_rate'][name]=D2['PnL'][name]['Charged Off']
    except:
        default_purpose['default_rate'][name]=pd.NaT
        
D2=pd.DataFrame(default_purpose)

# Merging default dataframe with median and mad
D1=loan_data_new.groupby(['pub_rec_bankruptcies']).agg({'PnL':['median','mad']})

D3=pd.merge(D1['PnL'],D2,left_index=True,right_index=True,how='inner').sort_values('default_rate')

print(D3)

D4=loan_data_new.groupby(['pub_rec_bankruptcies','grade']).agg({'PnL':'median'})
median_profitability={}

for name in loan_data_new.pub_rec_bankruptcies.dropna().unique():
    median_profitability[name]={}
    for gr in loan_data_new.grade.dropna().unique():
        try:  
            median_profitability[name][gr]=D4['PnL'][name][gr]
        except:
            median_profitability[name][gr]=pd.NaT
print(pd.DataFrame(median_profitability))


# In[ ]:


#loan_data_new['delinq_2yrs']=loan_data_new['delinq_2yrs'].astype('category')
D1=loan_data_new.groupby(['inq_last_6mths','loan_status']).agg({'PnL':'count'})
D2=D1.groupby(level=0).apply(lambda x:round(x*100/x.sum(),1))

#print(D1)
# Creating a dataframe for default rate
default_purpose={'default_rate':{}}

for name in loan_data_new.inq_last_6mths.dropna().unique():
    try:
        default_purpose['default_rate'][name]=D2['PnL'][name]['Charged Off']
    except:
        default_purpose['default_rate'][name]=pd.NaT
        
D2=pd.DataFrame(default_purpose)

# Merging default dataframe with median and mad
D1=loan_data_new.groupby(['inq_last_6mths']).agg({'PnL':['median','mad']})

D3=pd.merge(D1['PnL'],D2,left_index=True,right_index=True,how='inner').sort_values('default_rate')

print(D3)

D4=loan_data_new.groupby(['inq_last_6mths','grade']).agg({'PnL':'median'})
median_profitability={}

for name in loan_data_new.inq_last_6mths.dropna().unique():
    median_profitability[name]={}
    for gr in loan_data_new.grade.dropna().unique():
        try:  
            median_profitability[name][gr]=D4['PnL'][name][gr]
        except:
            median_profitability[name][gr]=pd.NaT
print(pd.DataFrame(median_profitability))



loan_data_new.dtypes

drop_columns2=['installment','term','total_pymnt_inv','total_rec_prncp']

loan_data_new_col_removed=loan_data_new.drop(drop_columns2,axis=1)



plt.figure(figsize=(10,10))
sns.countplot('emp_length',hue='loan_status',data=loan_data_new_col_removed)
plt.show()



# Converting employment length into a nominal variable
mapping_dict = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0

    },
    "grade":{
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": 6,
        "G": 7
    }
}

loan_data_new_col_removed=loan_data_new_col_removed.replace(mapping_dict)




# Creating bins for DTI,loan amount, revol_util and seeing their influence of profitability
loan_data_new['funded_amnt_bins']=pd.cut(loan_data_new['funded_amnt'],7)
print(loan_data_new.groupby(['grade','funded_amnt_bins'])['PnL'].median())


loan_data_new['dti_bins']= pd.cut(loan_data_new['dti'],7)
print(loan_data_new.groupby(['grade','dti_bins'])['PnL'].median())


loan_data_new['revol_util_bins']= pd.cut(loan_data_new['revol_util'],7)
print(loan_data_new.groupby(['grade','revol_util_bins'])['PnL'].median())



# Analyzing public records, Debt to Income ratio, inquires in last months, revolving utilization, revolving utilization

loan_data_new.corr()
plt.figure(figsize=(45,45))


fgrid=sns.lmplot(y='funded_amnt',x='annual_inc',fit_reg=False,hue='loan_status',col='term',data=loan_data_new)
ax=fgrid.axes[0][0]
plt.xscale('log')
plt.yscale('log')
plt.ylabel("Loan Inc Ratio")
plt.xlabel("Annual Inc")

plt.show()

#loan_data_new.columns

