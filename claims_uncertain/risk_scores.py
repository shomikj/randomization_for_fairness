import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm, ensemble, tree
import random
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')


class RiskScores:

    def __init__(self, df, features, label, attributes, random_seed, method):
        self.df = df
        self.features = features
        self.label = label
        self.attributes = attributes
        self.random_seed = random_seed
        self.method = method

        df_train, df_test = train_test_split(self.df, test_size=0.2, random_state=self.random_seed)
        
        X_train, X_test = df_train[self.features].to_numpy(), df_test[self.features].to_numpy()
        y_train, y_test = df_train[self.label].to_numpy(),  df_test[self.label].to_numpy()
        z_train, z_test = df_train[self.attributes].to_numpy(), df_test[self.attributes].to_numpy()

        self.n_train = len(X_train)
        self.n_test = len(X_test)
        self.partition = {'X_tr': X_train,'X_test': X_test,
                          'y_tr': y_train,'y_test': y_test,
                          'z_tr': z_train,'z_test': z_test}

        self.results = pd.DataFrame()

    def train_model(self, X_train, y_train):
        if self.method == "log":
            clf = LogisticRegression(random_state=self.random_seed)
        elif self.method == "rf":
            clf = ensemble.RandomForestClassifier(n_estimators=25, max_depth=10, random_state=self.random_seed)
        elif self.method == "dt":
            clf = tree.DecisionTreeClassifier(max_depth=10, random_state=self.random_seed)

        clf = CalibratedClassifierCV(clf)
        clf.fit(X_train, y_train)
        
        scores = [i[1] for i in clf.predict_proba(self.partition["X_test"])]
        return scores
    
    # For randomization using variance: calculate risk scores from bootstrapped models trained on sub-samples of the training data
    def get_bootstrap_scores(self, niter=11, bootstrap_size=0.5):
        for i in tqdm(range(niter)):
            np.random.seed(i)
            idx = np.random.permutation(self.n_train)[:int(self.n_train*bootstrap_size)]
            self.results["risk_b"+str(i)] = self.train_model(self.partition["X_tr"][idx], self.partition["y_tr"][idx])

    # For randomization using outliers: calculate conformal prediction p-value associated with outlier detection scores
    # We use distance to training data as our outlier detection score 
    def get_outlier_score(self, niter=5, validation_size=0.5):
        all_p_vals = []
        for i in tqdm(range(niter)):
            np.random.seed(i)
            idx = np.random.permutation(self.n_train)[:int(self.n_train*validation_size)]
            
            X_tr = self.partition["X_tr"][idx]
            X_cal = np.delete(self.partition["X_tr"], idx, axis=0)
            
            center = np.mean(X_tr, axis=0)
            nc_cal = np.linalg.norm(X_cal-center, axis=1)
            nc_test = np.linalg.norm(self.partition["X_test"]-center, axis=1)
            
            n_cal = len(nc_cal)
            p_vals = []
            for nc in nc_test:
                p_vals.append((nc_cal>nc).sum() / (n_cal+1))
            all_p_vals.append(np.array(p_vals))
        
        self.results["outlier_pval"] = np.mean(np.array(all_p_vals), axis=0)
    
    def get_risk_scores(self):
        self.results["y"] = self.partition["y_test"]

        for i,a in enumerate(self.attributes):
            self.results[a] = self.partition["z_test"][:, i]
            
        self.results["risk"] = self.train_model(self.partition["X_tr"], self.partition["y_tr"])
    
        self.get_bootstrap_scores()
        self.get_outlier_score()

        return self.results