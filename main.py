import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error



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
        
#create a exception class
class not_enough_colors_provided(Exception):
    
    def __init__(self,*args):
        if args:
            self._0 = args[0]
            self._1 = args[1]
        else:
            self.message = None
    
    def __str__(self):

        return 'The number of colors provided are %s, while the number of unique values in the columns are %s'%(self._0,self._1)        
        

class performanceReport():
    
    def __init__(self,pred_col_name,actual_col_name,df_test='Empty',df_train='Empty'):
        
        self.df_test = df_test
        self.df_train = df_train
        self.pred_col_name = pred_col_name
        self.actual_col_name = actual_col_name
        
        #if one of the two dataframes is empty, create a variable that indicates that there is only one dataframe for the class instance
        if not isinstance(df_test, pd.DataFrame):
            self.one_df = True
            self.df = df_train
        elif not isinstance(df_train, pd.DataFrame):
            self.one_df = True
            self.df = df_test
        else:
            self.one_df = False
            
        #create residuals
        if self.one_df:
            self.df['RESID'] = self.df[self.actual_col_name] - self.df[self.pred_col_name]
        else:
            self.df_test['RESID'] = self.df_test[self.actual_col_name] - self.df_test[self.pred_col_name]
            self.df_train['RESID'] = self.df_train[self.actual_col_name] - self.df_train[self.pred_col_name]
   
    #Method to apply temporary filters to datasets     
    def filterData(self,filters):
        
        if not self.one_df:
        
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
                    
            return df_train_temp, df_test_temp
        
        else:
            
            for i in filters:
                if i[1] == 'lt':
                    df_temp = self.df[self.df[i[0]]<i[2]]
                                                
                elif i[1] == 'lte':
                    df_temp = self.df[self.df[i[0]]<=i[2]]              
                    
                elif i[1] == 'gt':
                    df_temp = self.df[self.df[i[0]]>i[2]]                   
                    
                elif i[1] == 'gte':
                    df_temp = self.df[self.df[i[0]]>=i[2]]                   
                    
                elif i[1] == 'e':
                    df_temp = self.df[self.df[i[0]]==i[2]]
                    
                elif i[1] == 'ne':
                    df_temp = self.df[self.df[i[0]]!=i[2]]
                
            return df_temp
            
        
        
   
   
    def perfChart(self,metric_function,metric_name,chart_title,by='Empty',filters=[],bins=5,barwidth=0,barWidth=0.25,trainColor='#7f6d5f',testColor='#557f2d',figsize=[30,8],
                  x_tick_rotation=45,fig_path='fig1.png',all_metrics=True):
        
        
        #first, condition on if there are two datasets
        if not self.one_df:
            
            #the filters variable must be a list of list. Each sublist must have the following three elements
            #1, the name of the column to be filtered
            #2, 'lt', 'lte', 'gt', 'gte' , 'e' and 'ne' to designate the type of equality/inequality
            #3, the value to filter by
            #for example filters = [['col1_name','lte','6'],[col2_name','e','VALUE1']]
            if len(filters)!=0:
                
                df_train_temp, df_test_temp = self.filterData(filters)
                
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
                
                unique_vals_train = df_train_temp[temp_col_name].unique()
                unique_vals_test = df_test_temp[temp_col_name].unique()
                unique_vals = np.union1d(unique_vals_train,unique_vals_test)
                    
       
                
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
            train_perf_all = metric_function(df_train_temp[self.pred_col_name],df_train_temp[self.actual_col_name])
            test_perf_all = metric_function(df_test_temp[self.pred_col_name],df_test_temp[self.actual_col_name])                                
            
            if all_metrics:
                train_perf = [train_perf_all]
                test_perf = [test_perf_all]
                label_list = ['All']
            else:
                train_perf = []
                test_perf = []
                label_list = []                
                
            
            for j in unique_vals:
            
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

            
            plt.bar(r1,train_perf, color=trainColor, width=barWidth,edgecolor='white',label=('Train %s'%metric_name))
            plt.bar(r2,test_perf, color=testColor, width=barWidth,edgecolor='white',label=('Test %s'%metric_name))
            
            plt.xlabel(by, fontweight='bold')
            #plt.xticks([r + barWidth for r in range(len(test_perf))],label_list)
            plt.xticks([r + barWidth for r in range(len(test_perf))],label_list)
            plt.ylabel(metric_name, fontweight='bold')
            
            plt.rcParams["figure.figsize"] = (figsize[0],figsize[1])
            
            plt.title(chart_title)
            plt.xticks(rotation=x_tick_rotation,ha='right')
            
            plt.legend()
            
            plt.savefig(fig_path)      
            
            plt.show()
            
        else:
            
            #the filters variable must be a list of list. Each sublist must have the following three elements
            #1, the name of the column to be filtered
            #2, 'lt', 'lte', 'gt', 'gte' , 'e' and 'ne' to designate the type of equality/inequality
            #3, the value to filter by
            #for example filters = [['col1_name','lte','6'],[col2_name','e','VALUE1']]
            if len(filters)!=0:
                
                df_temp = self.filterData(filters)

            else:
                df_temp = self.df
                    
            
            #if data is numeric cut into bins and calculate
            if is_numeric_dtype(self.df[by]):
                
                #binning logic
                temp_max = df_temp[by].max()
                temp_min =df_temp[by].min()
                step_size = (temp_max - temp_min)/(bins-1)
                
                bin_list = [temp_min]
                
                for i in range(0,bins):
                    temp_min = temp_min + step_size
                    bin_list.append(temp_min)
                    
                temp_col_name = by + '_grouped'
                
                df_temp[temp_col_name] = pd.cut(df_temp[by],bin_list)
                    
                
            elif not is_numeric_dtype(self.df[by]):
                
                unique_vals = df_temp[by].sort_values().unique() 
                
                temp_col_name = by + '_grouped'
                
                df_temp[temp_col_name] = df_temp[by]
                
                
            else:
                
                raise dftrain_dftest_type_mismatch
            
            
            #loop through all buckets and calculate performance for each bucket
            perf_all = metric_function(df_temp[self.pred_col_name],df_temp[self.actual_col_name])
            
            if all_metrics:
                perfs = [perf_all]
                label_list = ['All']
            else:
                perfs = []
                label_list = []                
                
            
            for j in unique_vals:
                
                df_temp2 = df_temp[df_temp[temp_col_name]==j]

                
                if len(df_temp2) == 0:
                    temp_perf = 0
                else:
                    temp_perf = metric_function(df_temp2[self.pred_col_name],df_temp2[self.actual_col_name])
                    
                    
                #append metrics to list
                perfs.append(temp_perf)
                label_list.append(j)
                
            
            #set position of bar on X-axis    
            r1 = np.arange(len(label_list))
            r2 = [x + barWidth for x in r1]
            
            plt.bar(r1,perfs, color=trainColor, width=barWidth,edgecolor='white',label=('Train %s'%metric_name))
            
            plt.xlabel(by, fontweight='bold')
            #plt.xticks([r + barWidth for r in range(len(perfs))],label_list)
            plt.ylabel(metric_name, fontweight='bold')
            
            plt.rcParams["figure.figsize"] = (figsize[0],figsize[1])
            
            plt.title(chart_title)
            plt.xticks(rotation=x_tick_rotation,ha='right')
            
            plt.legend()
            
            plt.savefig(fig_path)      
            
            plt.show()     
            
            
    def residChart(self,by,chart_title,aggregation='None',font_size=25,figsize=[30,8],
                   boxplot=False,train=False,colorby='None',font_weight='bold',
                   bins=0,filters=[],colorbycolors=[],fig_path='fig1',xlabel_dict={},
                   ylabel_dict={},xticks_dict={},yticks_dict={},chart_title_dict={}):
        
        
        #assign a dataset as the to temp_df variable based on user input and the number of datasets provided
        #at class initiation
        if len(filters) != 0 and not self.one_df:
            df_train_temp, df_test_temp = self.filterData(filters)
            
            if train:
                temp_df = df_train_temp
            else:
                temp_df = df_test_temp
            
        elif len(filters) != 0 and self.one_df:
            temp_df = self.filterData(filters)
        
        elif len(filters) == 0 and self.one_df:
            temp_df = self.df
            
        elif len(filters) == 0 and not self.one_df:
            
            if train:
                temp_df = self.df_train
            else:
                temp_df = self.df_test


            
        if aggregation == 'None':
            
            if colorby != 'None':
                unique_vals = temp_df[colorby].unique()
                #raise error if not enough colors are provided
                if len(colorbycolors) != len(unique_vals) and len(colorbycolors) > 0:
                    raise not_enough_colors_provided(len(colorbycolors),len(unique_vals))
                
                color_index = 0
                
                for value in unique_vals:
                    temp_df2 = temp_df[temp_df[colorby]==value]
                    
                    if len(colorbycolors) != 0:
                        #color by user input list
                        plt.scatter(temp_df2[by],temp_df2['RESID'],
                                    fontsize=font_size,color=colorbycolors[color_index])
                        color_index += 1
                    else:
                        #allow default colors 
                        plt.scatter(temp_df2[by],temp_df2['RESID'])
            else:
                
                if not is_numeric_dtype(temp_df[by]):
                    unique_vals = temp_df[by].unique()
                    
                    for value in unique_vals:
                        
                        temp_df2 = temp_df[temp_df[by]==value]
                        
                        plt.scatter(temp_df2[by],temp_df2['RESID'])
                else:
                    plt.scatter(temp_df[by,temp_df['RESID']])
                
                        
        elif aggregation == 'mean':
            df_grouped = temp_df[[by,'RESID']].groupby(by=[by]).mean()
            
            #reset index to be a single index instead of double
            df_grouped.reset_index(0).reset_index(drop=True)
            df_grouped = df_grouped.reset_index()
            
            plt.scatter(df_grouped[by],df_grouped['PRED'])
            
        elif aggregation == 'median':
            df_grouped = temp_df[[by,'RESID']].groupby(by=[by]).median()
            
            #reset index to be a single index instead of double
            df_grouped.reset_index(0).reset_index(drop=True)
            df_grouped = df_grouped.reset_index()
            
            plt.scatter(df_grouped[by],df_grouped['PRED'])
            
        #xlabel logic
        if len(xlabel_dict) == 0:
            plt.xlabel(by,fontweight=font_weight,fontsize=font_size)
        else:
            plt.xlabel(**xlabel_dict)
            
        #ylabel logic
        if len(ylabel_dict) == 0:
            plt.ylabel('Residuals',fontweight=font_weight,fontsize=font_size)
        else:
            plt.ylabel(**ylabel_dict)
                
        #xticks logic
        if len(xticks_dict) == 0:
            plt.xticks(rotation=45,ha='right',fontsize=font_size)
        else:
            plt.xticks(**xticks_dict)
            
        #yticks logic
        if len(yticks_dict) == 0:
            plt.yticks(fontsize=font_size)
        else:
            plt.yticks(**yticks_dict)
            
        #title logic
        if len(chart_title_dict) == 0:
            plt.title(chart_title,fontsize=font_size)
        else:
            plt.title(**chart_title_dict)
        
        
        plt.rcParams["figure.figsize"] = (figsize[0],figsize[1])
        
        plt.legend()
        plt.savefig(fig_path)
        plt.show()
        
         

#create test data to run through code
test_df = pd.read_csv('sample_data.csv')
train_df = pd.read_csv('sample_data.csv')
df = pd.read_csv('sample_data.csv')

test_df['PRED'] = test_df['PRED'].astype(float)
test_df['ACTUAL'] = test_df['ACTUAL'].astype(float)
train_df['PRED']=train_df['PRED'].astype(float)
train_df['ACTUAL'] = train_df['ACTUAL'].astype(float)

def rmse(y,yhat):
    return math.sqrt(mean_squared_error(y,yhat))


test_report = performanceReport('PRED','ACTUAL',df_test= test_df,df_train=train_df)     

#print(test_df.head(10))

test_report.residChart(chart_title='RMSE by Group1',by='GROUP_1',
                       fig_path='scatter',xlabel_dict={'xlabel':'Group 3','fontsize':25})

#test_report.perfChart(metric_function=rmse,metric_name='RMSE',chart_title='RMSE by Group1',by='GROUP_1')
#test_report.residChart(chart_title='RMSE by Group1',by='GROUP_3',fig_path='scatter',xlabel_dict={'xlabel':'Group 3','fontsize':50})
   
    #def perfChart(self,metric_function,metric_name,chart_title,by='Empty',filters=[],bins=5,barwidth=0,barWidth=0.25,color1='#7f6d5f',color2='#557f2d',figsize=[30,8],
     #             x_tick_rotation=45,fig_path='fig1.png'):
            
            
        
        
        



