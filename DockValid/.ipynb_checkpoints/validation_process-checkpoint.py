import os
import glob
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, recall_score, precision_score,  roc_curve, auc
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
class validation_process:
    
    def __init__(self, data, active_col, score_type, rescore_method ='minmax', figsize = None, verbose = None):
        self.data = data
        self.active = active_col
        self.score_type = score_type
        #self.model = model
        self.rescore_method  = rescore_method 
        self.figsize = figsize
        if self.figsize == None:
            pass
        else:
            fig = plt.figure(figsize = self.figsize)
            background_color = "#F0F6FC"
            fig.patch.set_facecolor(background_color)
        sns.set()
        self.verbose = verbose
        self.Rescore()
        
    
    def Rescore(self):
        if self.rescore_method == 0:
            self.data['rescore'] = -self.data[self.score_type]
        elif self.rescore_method == 'minmax':            
            self.scl = MinMaxScaler().fit(-self.data[self.score_type].values.reshape(-1, 1))
            self.data['rescore'] = self.scl.transform(-self.data[self.score_type].values.reshape(-1, 1))
        elif self.rescore_method  == 'standard':            
            self.scl = StandardScaler().fit(-self.data[self.score_type].values.reshape(-1, 1))
            self.data['rescore'] = self.scl.transform(-self.data[self.score_type].values.reshape(-1, 1))
        elif self.rescore_method  == 'quantile':            
            self.scl = RobustScaler().fit(-self.data[self.score_type].values.reshape(-1, 1))
            self.data['rescore'] = self.scl.transform(-self.data[self.score_type].values.reshape(-1, 1))
    
    def metrics(self):
       
        self.fpr, self.tpr, self.thresholds = roc_curve(self.data[self.active], self.data['rescore'])
        
        self.roc_auc = round(auc(self.fpr, self.tpr),3)
        
        self.log_roc_auc = round(self.roc_log_auc(self.data[self.active], self.data['rescore'], ascending_score = False),3)
        
        self.bedroc = round(self.bedroc(self.data[self.active], self.data['rescore']),3)
        
        self.rie = round(self.rie(self.data[self.active], self.data['rescore']),3)
        
        self.ef1 = round(self.EF(self.data[self.active], self.data['rescore'], 0.01),3)
        
    def EF(self, actives_list, score_list, n_percent):
        """ Calculates enrichment factor.
        Parameters:
        actives_list - binary array of active/decoy status.
        score_list - array of experimental scores.
        n_percent - a decimal percentage.
        """
        total_actives = len(actives_list[actives_list == 1])
        total_compounds = len(actives_list)
        # Sort scores, while keeping track of active/decoy status
        # NOTE: This will be inefficient for large arrays
        labeled_hits = sorted(zip(score_list, actives_list), reverse=True)
        # Get top n percent of hits
        num_top = int(total_compounds * n_percent)
        top_hits = labeled_hits[0:num_top]    
        num_actives_top = len([value for score, value in top_hits if value == 1])
        # Calculate enrichment factor
        return num_actives_top / (total_actives * n_percent)
    
    def rie(self, y_true, y_score, alpha=20, pos_label=None):
        """Computes Robust Initial Enhancement [1]_. This function assumes that results
        are already sorted and samples with best predictions are first.
        Parameters
        ----------
        y_true : array, shape=[n_samples]
            True binary labels, in range {0,1} or {-1,1}. If positive label is
            different than 1, it must be explicitly defined.
        y_score : array, shape=[n_samples]
            Scores for tested series of samples
        alpha: float
            Alpha. 1/Alpha should be proportional to the percentage in EF.
        pos_label: int
            Positive label of samples (if other than 1)
        Returns
        -------
        rie_score : float
             Robust Initial Enhancement
        References
        ----------
        .. [1] Sheridan, R. P.; Singh, S. B.; Fluder, E. M.; Kearsley, S. K.
               Protocols for bridging the peptide to nonpeptide gap in topological
               similarity searches. J. Chem. Inf. Comput. Sci. 2001, 41, 1395-1406.
               DOI: 10.1021/ci0100144
        """
        if pos_label is None:
            pos_label = 1
        labels = y_true == pos_label
        N = len(labels)
        ra = labels.sum() / N
        ranks = np.argwhere(labels.values).astype(float) + 1  # need 1-based ranking
        observed = np.exp(-alpha * ranks / N).sum()
        expected = (ra * (1 - np.exp(-alpha))
                    / (np.exp(alpha / N) - 1))
        rie_score = observed / expected
        return rie_score

    def bedroc(self,y_true, y_score, alpha=20., pos_label=None):
        """Computes Boltzmann-Enhanced Discrimination of Receiver Operating
        Characteristic [1]_.  This function assumes that results are already sorted
        and samples with best predictions are first.
        Parameters
        ----------
        y_true : array, shape=[n_samples]
            True binary labels, in range {0,1} or {-1,1}. If positive label is
            different than 1, it must be explicitly defined.
        y_score : array, shape=[n_samples]
            Scores for tested series of samples
        alpha: float
            Alpha. 1/Alpha should be proportional to the percentage in EF.
        pos_label: int
            Positive label of samples (if other than 1)
        Returns
        -------
        bedroc_score : float
            Boltzmann-Enhanced Discrimination of Receiver Operating Characteristic
        References
        ----------
        .. [1] Truchon J-F, Bayly CI. Evaluating virtual screening methods: good
               and bad metrics for the "early recognition" problem.
               J Chem Inf Model. 2007;47: 488-508.
               DOI: 10.1021/ci600426e
        """
        if pos_label is None:
            pos_label = 1
        labels = y_true == pos_label
        ra = labels.sum() / len(labels)
        ri = 1 - ra
        rie_score = self.rie(y_true, y_score, alpha=alpha, pos_label=pos_label)
        bedroc_score = (rie_score * ra * np.sinh(alpha / 2) /
                        (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * ra))
                        + 1 / (1 - np.exp(alpha * ri)))
        return bedroc_score

    def roc_log_auc(self, y_true, y_score, pos_label=None, ascending_score=True,
                    log_min=0.001, log_max=1.):
        """Computes area under semi-log ROC.
        Parameters
        ----------
        y_true : array, shape=[n_samples]
            True binary labels, in range {0,1} or {-1,1}. If positive label is
            different than 1, it must be explicitly defined.
        y_score : array, shape=[n_samples]
            Scores for tested series of samples
        pos_label: int
            Positive label of samples (if other than 1)
        ascending_score: bool (default=True)
            Indicates if your score is ascendig. Ascending score icreases with
            deacreasing activity. In other words it ascends on ranking list
            (where actives are on top).
        log_min : float (default=0.001)
            Minimum value for estimating AUC. Lower values will be clipped for
            numerical stability.
        log_max : float (default=1.)
            Maximum value for estimating AUC. Higher values will be ignored.
        Returns
        -------
        auc : float
            semi-log ROC AUC
        """
        if ascending_score:
            y_score = -y_score
        fpr, tpr, t = roc_curve(y_true, y_score)
        fpr = fpr.clip(log_min)
        idx = (fpr <= log_max)
        log_fpr = 1 - np.log10(fpr[idx]) / np.log10(log_min)
        return auc(log_fpr, tpr[idx])
    
    def plot_roc (self):
        """ Calculates and plots and ROC and AUC.
        Parameters:
        actives_list - binary array of active/decoy status.
        score_list - array of experimental scores.
        tool- a string name of the tool used.
        receptor - a string name of the protein used.
        """
     
        
        self.gmeans = np.sqrt(self.tpr * (1-self.fpr))
        # locate the index of the largest g-mean
        ix = np.argmax(self.gmeans)
        cutoff = -self.scl.inverse_transform(self.thresholds[ix].reshape(-1, 1))
        if self.verbose != None:
            print('Best Threshold=%f,tpr=%.3f,fpr=%.3f, G-mean=%.3f' % (self.thresholds[ix],
                                                              self.tpr[ix],
                                                              self.fpr[ix],
                                                              self.gmeans[ix]))
            print('Cutoff=%.3f kcal/mol' % cutoff)
        lw = 2
        
        plt.plot(self.fpr, self.tpr, 
                 lw=lw, label=f'{self.score_type} (AUC = %0.3f), Cutoff = %.2f kcal/mol' % (self.roc_auc, cutoff))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.scatter(self.fpr[ix], self.tpr[ix], marker='o')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize = 16)
        plt.ylabel('True Positive Rate', fontsize = 16)
        plt.title('Receiver operating characteristic', fontsize = 24, weight = 'semibold')
        plt.legend(loc="lower right")
        
        
    def validation(self):
        self.metrics()
        index = ['Model', "AUCROC", "logAUCROC", "BedROC", "EF1%","RIE"]
        data =[self.score_type, self.roc_auc,self.log_roc_auc, 
               self.bedroc, self.ef1, self.rie]
        self.table = pd.DataFrame(data = data, index = index).T
        if self.verbose != None:
            display(self.table)
        self.plot_roc()
