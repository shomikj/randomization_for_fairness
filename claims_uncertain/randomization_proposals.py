import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class experiment():
    def __init__(self, df):
        self.df = df
        self.n = len(df)
        self.df = self.df.sort_values(by=["risk"], ascending=False).reset_index(drop=True)

    def weighted_lottery(self, claims, k, weights=[]):
        if len(weights)>0:
            norm_weights = weights / np.sum(weights)
        else:
            norm_weights = claims / np.sum(claims)
        selected = np.random.choice(claims, size=k, replace=False, p=norm_weights)
        return selected

    def get_partial_bf_selections(self, idx, claims, perc_random_n, perc_random_k, selection_rate):
        k = int(self.n * selection_rate)
        random_k = int(perc_random_k * k)
        not_random_k = k - random_k
        random_n = int(perc_random_n * self.n)

        if not_random_k == 0 or random_n > self.n - not_random_k: #all resources randomized
            determ = np.array([])
            random = self.weighted_lottery(idx[:random_n], k, claims[:random_n])
        elif random_k == 0 or random_n < random_k: #no resources randomized
            determ = idx[:k]
            random = np.array([])
        else:
            determ = idx[:not_random_k]
            random = self.weighted_lottery(idx[not_random_k:not_random_k+random_n], random_k,
                                           claims[not_random_k:not_random_k+random_n])
        return np.concatenate([determ, random])

    def get_variance_selections(self, idx, decision_boundary, selection_rate):
        k = int(self.n * selection_rate)

        n_bootstrap = 11
        b1 = np.zeros(self.n)
        for j in range(n_bootstrap):
            b1 = b1 + (self.df["risk_b"+str(j)]>=decision_boundary).astype(int)
        
        self.df["maj_vote"] = b1 / n_bootstrap

        confident_selections = self.df.loc[self.df["maj_vote"]==1].index.values
        k_remaining = k - len(confident_selections)

        if k_remaining > 0:
            unconfident_selections = self.df.loc[(self.df["maj_vote"]>0)&(self.df["maj_vote"]<1)].index.values
            if len(unconfident_selections)==0:
                confident_selections = idx[:k]
                random_selections = np.array([])
                k_remaining = 0
            else:
                unconfident_weights = self.df.loc[(self.df["maj_vote"]>0)&(self.df["maj_vote"]<1), "maj_vote"].values
                random_selections = self.weighted_lottery(unconfident_selections, k_remaining, unconfident_weights)
        else:
            unconfident_selections = np.array([])
            random_selections = np.array([])

        return np.concatenate([confident_selections, random_selections]), k_remaining / k, len(unconfident_selections) / self.n

    def get_outlier_selections(self, idx, alpha, selection_rate):
        k = int(self.n * selection_rate)

        top_k_idx = idx[:k]
        confident_selections = self.df.loc[(self.df.index.isin(idx[:k]))&(self.df["outlier_pval"]>alpha)].index.values
        k_remaining = k - len(confident_selections)

        if k_remaining > 0:
            unconfident_selections = self.df.loc[(self.df["outlier_pval"]<=alpha)].index.values
            random_selections = np.random.choice(unconfident_selections, k_remaining, replace=False)
        else:
            unconfident_selections = np.array([])
            random_selections = np.array([])

        return np.concatenate([confident_selections, random_selections]), k_remaining / k, len(unconfident_selections) / self.n

    def get_partial_bf_utility(self, perc_random_n, perc_random_k, selection_rate, iterations=10):
        claims = self.df["risk"].to_numpy()
        idx = self.df.index.to_numpy()
        
        random_util = []
        for i in range(iterations):
            random_idx = self.get_partial_bf_selections(idx, claims, perc_random_n, perc_random_k, selection_rate)
            random_util.append(np.mean(self.df.loc[random_idx, "y"].values))
        return np.mean(random_util)

    def exp_randomization_using_variance(self, selection_rate, iterations=100):
        k = int(self.n * selection_rate)
        decision_boundary = self.df.loc[k-1, "risk"]
        idx = self.df.index.to_numpy()

        results = []
        for i in range(iterations):
            random_idx, perc_random_k, perc_random_n = self.get_variance_selections(idx, decision_boundary, selection_rate)
            random_util = np.mean(self.df.loc[random_idx, "y"].values)
            results.append({"random_util": random_util, "perc_random_k": perc_random_k, "perc_random_n": perc_random_n,
                           "partial_bf_util": self.get_partial_bf_utility(perc_random_n, perc_random_k, selection_rate),
                           "determ_util": np.mean(self.df.loc[:k, "y"])})

        results = pd.DataFrame(results)
        return results

    def exp_randomization_using_outliers(self, alpha, selection_rate, iterations=100):
        claims = self.df["risk"].to_numpy()
        idx = self.df.index.to_numpy()
        k = int(self.n * selection_rate)

        results = []
        for i in range(iterations):
            random_idx, perc_random_k, perc_random_n = self.get_outlier_selections(idx, alpha, selection_rate)
            random_util = np.mean(self.df.loc[random_idx, "y"].values)
            results.append({"random_util": random_util, "perc_random_k": perc_random_k, "perc_random_n": perc_random_n,
                           "partial_bf_util": self.get_partial_bf_utility(perc_random_n, perc_random_k, selection_rate),
                           "determ_util": np.mean(self.df.loc[:k, "y"])})
        
        results = pd.DataFrame(results)
        return results

    def exp_partial_bf_randomization_rate_utility(self, selection_rate, iterations=100, linspace=11):
        results = []
        for perc_random_k in np.linspace(0, 1, linspace):
            for perc_random_n in np.linspace(0, 1, linspace):
                random_n = int(perc_random_n * self.n)
                random_k = int(perc_random_k * self.n * selection_rate)
                not_random_k = int(self.n * selection_rate) - random_k
                if random_n < random_k or random_n > self.n - not_random_k:
                    continue
                
                random_util = self.get_partial_bf_utility(perc_random_n, perc_random_k, selection_rate, iterations)
                results.append({"perc_random_k": perc_random_k, "perc_random_n": perc_random_n, "random_util": random_util})

        results = pd.DataFrame(results)
        return results

    def exp_systemic_exclusion(self, selection_rate=0.1, noise_std=0, num_decision_makers=4, iterations=100,
                               randomization_method="variance", alpha=None, perc_random_n=None, perc_random_k=None):
        
        k = int(selection_rate*self.n)

        results = []
        for i in range(iterations):
            claims = self.df["risk"].to_numpy()
            people = self.df.index.values

            noisy_claims = {}
            sorted_noisy_people = {}
            sorted_noisy_claims = {}
            for j in range(num_decision_makers):
                if noise_std>0:
                    noisy_claims[j] = np.clip(np.add(claims, np.random.normal(0, noise_std, self.n)), 0, 1)
                else:
                    noisy_claims[j] = claims
                sorted_noisy_people[j] = np.array([x for _,x in sorted(zip(noisy_claims[j], people), reverse=True)])
                sorted_noisy_claims[j] = sorted(noisy_claims[j], reverse=True)
            
            random_chosen = {}
            for j in range(num_decision_makers):
                if randomization_method == "outliers":
                    random_chosen[j], _, _ = self.get_outlier_selections(sorted_noisy_people[j], alpha, selection_rate) 
                elif randomization_method == "variance":
                    decision_boundary = sorted_noisy_claims[j][k-1]
                    random_chosen[j], _, _ = self.get_variance_selections(sorted_noisy_people[j], decision_boundary, selection_rate)
                elif randomization_method == "partial_bf":
                    random_chosen[j] = self.get_partial_bf_selections(sorted_noisy_people[j], sorted_noisy_claims[j], perc_random_n, perc_random_k, selection_rate)
                else:
                    random_chosen[j] = sorted_noisy_people[j][:k]

            random_chosen_once = random_chosen[0]
            for j in range(1,num_decision_makers):
                random_chosen_once=np.union1d(random_chosen_once, random_chosen[j])
                results.append({"m": j+1, "random_ser": 1-(len(random_chosen_once)/self.n)})
                
        results = pd.DataFrame(results)
        results = results.groupby("m").mean().reset_index()
        return results

    def exp_partial_bf_randomization_rate_ser(self, selection_rate=0.25, noise_std=0, num_decision_makers=4, iterations=100, linspace=11):
        results = []
        for perc_random_k in np.linspace(0, 1, linspace):
            for perc_random_n in np.linspace(0, 1, linspace):
                random_n = int(perc_random_n * self.n)
                random_k = int(perc_random_k * self.n * selection_rate)
                not_random_k = int(self.n * selection_rate) - random_k
                if random_n < random_k or random_n > self.n - not_random_k:
                    continue

                random_ser = self.exp_systemic_exclusion(selection_rate, noise_std, num_decision_makers, iterations,
                                                        "partial_bf", None, perc_random_n, perc_random_k)
                random_ser["perc_random_k"] = perc_random_k
                random_ser["perc_random_n"] = perc_random_n
                results.append(random_ser)

        results = pd.concat(results)
        return results