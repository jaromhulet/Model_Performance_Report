
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import math



#create a exception class
class dftrain_dftest_type_mismatch(Exception):
    
    def __init__(self,*args):
        if args:
            self.message = args[0]
        else:
            self.message = None
    
    def __str__(self):

        if self.message:
            return 'customError, {0} '.format(self.message)
        else:
            return 'The by variable types between df_train and df_test are mismatches'
        

class performanceReport():
    
    def __init__(self,pred_col_name,actual_col_name,df_test='Empty',df_train='Empty'):
        
        self.df_test = df_test
        self.df_train = df_train
        self.pred_col_name = pred_col_name
        self.actual_col_name = actual_col_name
        
        #if one of the two dataframes is empty, create a variable that indicates that there is only one dataframe for the class instance
        if not isinstance(df_test, pd.DataFrame):
            self.one_df = 1
            self.df = df_train
        elif not isinstance(df_train, pd.DataFrame):
            self.one_df = 1
            self.df = df_test
        else:
            self.one_df = 0
        
       
   
   
    def perfChart(self,metric_function,metric_name,chart_title,by='Empty',filters=[],bins=5,barwidth=0,barWidth=0.25,color1='#7f6d5f',color2='#557f2d',figsize=[30,8],
                  x_tick_rotation=45,fig_path='fig1.png'):
        
        
        #first, condition on if there are two datasets
        if self.one_df == 0:
            
            #the filters variable must be a list of list. Each sublist must have the following three elements
            #1, the name of the column to be filtered
            #2, 'lt', 'lte', 'gt', 'gte' , 'e' and 'ne' to designate the type of equality/inequality
            #3, the value to filter by
            #for example filters = [['col1_name','lte','6'],[col2_name','e','VALUE1']]
            if len(filters)!=0:
                for i in filters:
                    if i[1] == 'lt':
                        df_train_temp = self.df_train[self.df_train[i[0]]<i[2]]
                        df_test_temp = self.df_test[self.df_test[i[0]]<i[2]]
                                                    
                    elif i[1] == 'lte':
                        df_train_temp = self.df_train[self.df_train[i[0]]<=i[2]]
                        df_test_temp = self.df_test[self.df_test[i[0]]<=i[2]]                    
                        
                    elif i[1] == 'gt':
                        df_train_temp = self.df_train[self.df_train[i[0]]>i[2]]
                        df_test_temp = self.df_test[self.df_test[i[0]]>i[2]]                    
                        
                    elif i[1] == 'gte':
                        df_train_temp = self.df_train[self.df_train[i[0]]>=i[2]]
                        df_test_temp = self.df_test[self.df_test[i[0]]>=i[2]]                    
                        
                    elif i[1] == 'e':
                        df_train_temp = self.df_train[self.df_train[i[0]]==i[2]]
                        df_test_temp = self.df_test[self.df_test[i[0]]==i[2]]
                        
                    elif i[1] == 'ne':
                        df_train_temp = self.df_train[self.df_train[i[0]]!=i[2]]
                        df_test_temp = self.df_test[self.df_test[i[0]]!=i[2]]
            else:
                df_train_temp = self.df_train
                df_test_temp = self.df_test
                    
            
            #if data is numeric cut into bins and calculate
            if is_numeric_dtype(self.df_train[by]) and is_numeric_dtype(self.df_test[by]):
                
                #binning logic
                temp_max = max(df_train_temp[by].max(),df_test_temp[by].max())
                temp_min = min(df_train_temp[by].min(),df_test_temp[by].min())
                step_size = (temp_max - temp_min)/(bins-1)
                
                bin_list = [temp_min]
                
                for i in range(0,bins):
                    temp_min = temp_min + step_size
                    bin_list.append(temp_min)
                    
                temp_col_name = by + '_grouped'
                
                df_train_temp[temp_col_name] = pd.cut(df_train_temp[by],bin_list)
                df_test_temp[temp_col_name] = pd.cut(df_test_temp[by],bin_list)
                    
       
                
            elif not is_numeric_dtype(self.df_train[by]) and not is_numeric_dtype(self.df_test[by]):
                
                train_unique_vals = df_train_temp[by].sort_values().unique()
                test_unique_vals = df_test_temp[by].sort_values().unique()
                unique_vals = np.union1d(train_unique_vals,test_unique_vals)
                #unique_vals = unique_vals.dropna()   
                
                temp_col_name = by + '_grouped'
                
                df_train_temp[temp_col_name] = df_train_temp[by]
                df_test_temp[temp_col_name] = df_test_temp[by]
                
                
            else:
                raise dftrain_dftest_type_mismatch
            
            
            
            #loop through all buckets and calculate performance for each bucket
            for j in unique_vals:
                
                train_perf = []
                test_perf = []
                label_list = []
            
                train_perf_all = metric_function(df_train_temp[self.pred_col_name],df_train_temp[self.actual_col_name])
                test_perf_all = metric_function(df_test_temp[self.pred_col_name],df_test_temp[self.actual_col_name])                
                
                df_train_temp2 = df_train_temp[df_train_temp[temp_col_name]==j]
                df_test_temp2 = df_test_temp[df_test_temp[temp_col_name]==j]
                
                if len(df_train_temp2) == 0:
                    temp_train_perf = 0
                else:
                    temp_train_perf = metric_function(df_train_temp2[self.pred_col_name],df_train_temp2[self.actual_col_name])
                    
                if len(df_test_temp2) == 0:
                    temp_test_perf = 0
                else:
                    temp_test_perf = metric_function(df_test_temp2[self.pred_col_name],df_test_temp2[self.actual_col_name])
                    
                    
                #append metrics to list
                train_perf.append(temp_train_perf)
                test_perf.append(temp_test_perf)
                
                label_list.append(j)
            
            #set position of bar on X-axis    
            r1 = np.arange(len(label_list))
            r2 = [x + barWidth for x in r1]
            r3 = [x + barWidth for x in r2]
            
            plt.bar(r1,train_perf, color=color1, width=barWidth,edgecolor='white',label=('Train %s'%metric_name))
            plt.bar(r2,test_perf, color=color2, width=barWidth,edgecolor='white',label=('Test %s'%metric_name))
            
            plt.xlabel(by, fontweight='bold')
            plt.xticks([r + barWidth for r in range(len(test_perf))],label_list)
            plt.ylabel(metric_name, fontweight='bold')
            
            plt.rcParams["figure.figsize"] = (figsize[0],figsize[1])
            
            plt.title(chart_title)
            plt.xticks(rotation=x_tick_rotation,ha='right')
            
            plt.legend()
            
            plt.savefig(fig_path)      
            
            plt.show()
            
        else:
            print('There is only one dataframe provided - I need to make the code to handle this')
            

#create test data to run through code
test_df = pd.read_csv('sample_data.csv')
train_df = pd.read_csv('sample_data.csv')
df = pd.read_csv('sample_data.csv')

test_df['PRED'] = test_df['PRED'].astype(float)
test_df['ACTUAL'] = test_df['ACTUAL'].astype(float)
train_df['PRED']=train_df['PRED'].astype(float)
train_df['ACTUAL'] = train_df['ACTUAL'].astype(float)

def rmse(y,yhat):
    temp_dif = y - yhat
    return temp_dif.mean()


test_report = performanceReport('PRED','ACTUAL',df_test= test_df, df_train=train_df)     

#print(test_df.head(10))

test_report.perfChart(metric_function=rmse,metric_name='RMSE',chart_title='RMSE by Group1',by='GROUP_1')
   
    #def perfChart(self,metric_function,metric_name,chart_title,by='Empty',filters=[],bins=5,barwidth=0,barWidth=0.25,color1='#7f6d5f',color2='#557f2d',figsize=[30,8],
     #             x_tick_rotation=45,fig_path='fig1.png'):
            
            
        
        
        



