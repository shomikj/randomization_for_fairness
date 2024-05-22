import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sympy.stats import QuadraticU, density
import math
import matplotlib.lines as mlines
from inv_normal import inv_normal


class experiment():
    # Initialize distribution with 2 parameters
    def __init__(self, dist="uniform", p1=0, p2=0, n=1000):
        self.dist = dist
        self.p1 = p1
        self.p2 = p2
        self.n = n

    # Get claims based on distribution
    def get_claims(self):
        if self.dist=="uniform":
            return np.random.uniform(0,1,self.n)
        elif self.dist=="normal":
            loc, scale = self.p1, self.p2
            a, b = (0 - loc) / scale, (1 - loc) / scale
            return stats.truncnorm(a, b, loc=loc, scale=scale).rvs(self.n)
        elif self.dist=="inv_normal":
            return inv_normal(0, 1, self.p1, self.p2, size=self.n)
        elif self.dist=="pareto":
            loc,scale = -1,self.p2
            b = self.p1
            c = (1 - loc) / scale
            return stats.truncpareto(b, c, loc=loc, scale=scale).rvs(self.n)
        elif self.dist=="inv_pareto":
            loc,scale = -1,self.p2
            b = self.p1
            c = (1 - loc) / scale
            return 1-stats.truncpareto(b, c, loc=loc, scale=scale).rvs(self.n)

    # Get pdf of claims based on distribution
    def get_claims_pdf(self):
        x = np.linspace(0,1,1000)
        if self.dist=="uniform":
            y = np.ones(1000)
        elif self.dist=="normal":
            loc, scale = self.p1, self.p2
            a, b = (0 - loc) / scale, (1 - loc) / scale
            y = stats.truncnorm(a, b, loc=loc, scale=scale).pdf(x)
        elif self.dist=="inv_normal":
            loc, scale = self.p1, self.p2
            a, b = (0 - loc) / scale, (1 - loc) / scale
            y = stats.truncnorm(a, b, loc=loc, scale=scale).pdf(x)
            y = np.max(y)-y
        elif self.dist=="pareto":
            loc,scale = -1,self.p2
            b = self.p1
            c = (1 - loc) / scale
            y = stats.truncpareto(b, c, loc=loc, scale=scale).pdf(x)
        elif self.dist=="inv_pareto":
            loc,scale = -1,self.p2
            b = self.p1
            c = (1 - loc) / scale
            y = stats.truncpareto(b, c, loc=loc, scale=scale).pdf(x)
            y = np.max(y)-y
        return x,y

    def weighted_lottery(self, claims, k, weights=[]):
        if len(weights)>0:
            norm_weights = weights / np.sum(weights)
        else:
            norm_weights = claims / np.sum(claims)
        selected = np.random.choice(claims, size=k, replace=False, p=norm_weights)
        return selected

    # Full Broome Fair Lottery: Expected Utility vs Selection Rate
    def full_bf_utility_vs_selection(self, iterations=1000):
        results = []
        for i in tqdm(range(iterations)):
            claims = self.get_claims()
            sorted_claims = np.sort(claims)
            for k in range(1, int(self.n/2)):
                determ_util = np.mean(sorted_claims[-k:])
                random_util = np.mean(self.weighted_lottery(sorted_claims, k))
                results.append({"k": k, "random_util": random_util, "determ_util": determ_util})
        
        results = pd.DataFrame(results)
        results = results.groupby("k").mean().reset_index()
        results["selection_rate"] = results["k"]/self.n
        results["util_diff"] = results["determ_util"] - results["random_util"]
        
        return results

    # Partial Broome Fair Lottery: Expected Utility vs Selection Rate
    # Set U = partial amount * K and L = N - 2K + U
    # Directly select top U claims and reject bottom L claims
    # Conduct weighted lottery over the remaining claims
    def partial_bf_utility_vs_selection(self, partial=0.25, iterations=1000):
        results = []
        for i in tqdm(range(1000)):
            claims = self.get_claims()
            sorted_claims = np.sort(claims)
            for k in range(1, int(self.n/2)):
                determ_util = np.mean(sorted_claims[-k:])
                
                u = round(partial*k)
                l = round(self.n-(2-partial)*k)
                if u==0:
                    determ = np.array([])
                    random = self.weighted_lottery(sorted_claims[-2*k:], k)
                else:
                    determ = sorted_claims[-u:]
                    if ((k-u)>0):
                        random = self.weighted_lottery(sorted_claims[l:-u], k-u)
                    else:
                        random = np.array([])
                random_util = np.mean(np.concatenate([determ, random]))
                results.append({"k": k, "random_util": random_util, "determ_util": determ_util})

        results = pd.DataFrame(results)
        results = results.groupby("k").mean().reset_index()
        results["selection_rate"] = results["k"]/self.n
        results["util_diff"] = results["determ_util"] - results["random_util"]
        
        return results

    # Test effect of randomization on systemic exclusion rate
    # partial: amount of randomization (see partial BF) 
    # Fixed selection rate, number of decision makers, and gaussian noise added to each decision maker's estimation of claims
    def systemic_exclusion(self, partial=0.5, selection_rate=0.5, noise_std=0, num_decision_makers=10, iterations=1000):
        results = []
        for i in tqdm(range(iterations)):
            claims = self.get_claims()
            claims = np.sort(claims)
            people = np.arange(self.n)
            
            noisy_claims = {}
            noisy_people = {}
            for j in range(num_decision_makers):
                if noise_std>0:
                    noisy_claims[j] = np.clip(np.add(claims, np.random.normal(0, noise_std, self.n)), 0, 1)
                    noisy_people[j] = np.array([x for _,x in sorted(zip(noisy_claims[j], people))])
                    noisy_claims[j] = np.sort(noisy_claims[j])
                else:
                    noisy_claims[j] = claims
                    noisy_people[j] = people     
            
            k = int(selection_rate*self.n)
            determ_chosen = {}
            random_chosen = {}
            for j in range(num_decision_makers):
                determ_chosen[j] = noisy_people[j][-k:]

                if partial==0:
                    random_chosen[j] = self.weighted_lottery(noisy_people[j], k, noisy_claims[j])
                else:
                    u = round(partial*k)
                    l = round(self.n-(2-partial)*k)
                    if u==0:
                        determ = np.array([])
                        random = self.weighted_lottery(noisy_people[j][-2*k:], k, noisy_claims[j][-2*k:])
                    else:
                        determ = noisy_people[j][-u:]
                        if ((k-u)>0):
                            random = self.weighted_lottery(noisy_people[j][l:-u], k-u, noisy_claims[j][l:-u])
                        else:
                            random = np.array([])
                    random_chosen[j] = np.concatenate([determ, random])
            

            determ_chosen_once = determ_chosen[0]
            random_chosen_once = random_chosen[0]
            for j in range(1,num_decision_makers):
                determ_chosen_once=np.union1d(determ_chosen_once, determ_chosen[j])
                random_chosen_once=np.union1d(random_chosen_once, random_chosen[j])
                
                results.append({"m": j+1, "determ": 1-(len(determ_chosen_once)/self.n), "random": 1-(len(random_chosen_once)/self.n)})

        results = pd.DataFrame(results)
        results = results.groupby("m").mean().reset_index()
        results["diff"] = results["determ"] - results["random"]
        
        return results