import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
from validation_process import validation_process
from internal_valid import internal_valid
sns.set()
class validation_compare:
    
    def __init__(self, path, active_col, rescore_method, clean = None,verbose = None, figsize = None, savefig=False):
       
        self.path = path
        self.active_col = active_col
        self.rescore_method = rescore_method 
        self.verbose = verbose
        self.clean = clean
       
        self.savefig = savefig
        
        img_directory = 'Img'
        self.path_raw = os.path.join(str(os.getcwd()), img_directory)
        if os.path.isdir(self.path_raw) == False:
            os.mkdir(self.path_raw)
        else:
            pass
        
        self.figsize = figsize
        if self.figsize == None:
            pass
        else:
            fig = plt.figure(figsize = self.figsize)
            background_color = "#F0F6FC"
            fig.patch.set_facecolor(background_color)
        #sns.set()
      
    def get_data_name(self):
        #os.chdir(self.path)
        #self.data_dir = str(os.getcwd())
        self.data_name = []
        for i in sorted(glob.glob(f"{self.path}/*.csv")):
            self.data_name.append(i[len(self.path)+1:-4])
        print(self.data_name)
        
    def external_compare(self):

        plt.figure(figsize = (12,8))
        self.table = pd.DataFrame()
        for i in self.data_name:
            data = pd.read_csv(f"{self.path}/{i}.csv")
            
            dock = validation_process(data = data, active_col = self.active_col, score_type= f"{i}_median", rescore_method =self.rescore_method, verbose = self.verbose)
            dock.validation()
            self.table = pd.concat([self.table, dock.table], axis =0).reset_index(drop = True)
            #display(post.data.head(2))
        if self.savefig == True:
            #plt.savefig(f'{model}_compare.png', dpi = 600)
            plt.savefig(f'{self.path_raw}/Compare_external.png', dpi = 600)
        #plt.savefig('Compare_external.png', dpi = 600)
      
    def fit(self):
        self.get_data_name()
        self.external_compare()
    
    
    def internal_fit(self):
        for i in self.data_name:
            print(f"{i} processing...............................")
            df = pd.read_csv(f"{self.path}/{i}.csv")
            df.rename(columns = {(df.columns[0]):('Col')}, inplace=True)
            df.drop(['Col'], axis =1, inplace = True)
            
            inter = internal_valid(df, active_col=self.active_col,name =i, rescore_method='minmax', figsize = (12,8), savefig=True, clean = self.clean, verbose = self.verbose)
            inter.fit()    
            print(f"Finsish...,,,,,..............................")
    
    
