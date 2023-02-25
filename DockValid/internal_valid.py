import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from validation_process import validation_process
import warnings
warnings.filterwarnings('ignore')

class internal_valid:
                 
    def __init__(self, data, active_col, rescore_method, name,clean = None,figsize = None, savefig = False, verbose = None):
        self.data = data
        self.active_col = active_col
        self.rescore_method = rescore_method
        self.name = name
        self.figsize = figsize
        self.clean = clean
        if self.figsize == None:
            pass
        else:
            fig = plt.figure(figsize = self.figsize)
            background_color = "#F0F6FC"
            fig.patch.set_facecolor(background_color)
        #sns.set()
          # Create directory for raw result
        img_directory = 'Img'
        self.path_raw = os.path.join(str(os.getcwd()), img_directory)
        if os.path.isdir(self.path_raw) == False:
            os.mkdir(self.path_raw)
        else:
            pass
        
        self.savefig = savefig
        self.verbose = verbose
        
        
    def visualize_distribution(self):
        if self.clean == True:
            self.data = self.data[self.data<10]
        sns.set()
        plt.figure(figsize = (24,8))

        plt.subplot(121)
        ax = sns.histplot(data=self.data.drop([self.active_col], axis =1), kde = True)
        ax.set_ylabel(None)
        ax.set_xlabel("Docking score", fontsize = 16, weight = 'semibold')
        ax.set_title("Histogram", fontsize = 30, weight = 'semibold')

        plt.subplot(122)
        ax2 = sns.boxplot(data=self.data.drop([self.active_col], axis =1), orient = 'h')
        ax2.set_xlabel("Docking score", fontsize = 16, weight = 'semibold')
        ax2.set_title("Boxplot", fontsize = 30, weight = 'semibold')
        if self.savefig == True:
            #plt.savefig(f'{model}_compare.png', dpi = 600)
            plt.savefig(f'{self.path_raw}/{self.name}_score_distribution.png', dpi = 600)
    
    
    def internal_process(self):
        self.table = pd.DataFrame()
        self.df_table = pd.DataFrame()
        for i in self.data.drop([self.active_col], axis = 1).columns:
            data = self.data[[i,self.active_col]]
            df_table = pd.concat([self.df_table, data], axis =1).reset_index(drop = True)
            dock = validation_process(data = data, active_col = self.active_col, score_type= f"{i}", rescore_method =self.rescore_method, verbose = self.verbose)
            dock.validation()
            self.table = pd.concat([self.table, dock.table], axis =0).reset_index(drop = True)
        if self.savefig == True:
            #plt.savefig(f'{model}_compare.png', dpi = 600)
            plt.savefig(f'{self.path_raw}/{self.name}_internal_compare.png', dpi = 600)
        plt.show()
            
    def fit(self):
        
        self.internal_process()
        self.visualize_distribution()
    
        
        

    
 
