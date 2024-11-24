## Classes of bandit environments used in the paper

import time
import os
import sys
import numpy as np
import scipy.sparse
import scipy.linalg
import sklearn.model_selection
import sklearn.tree
import sklearn.ensemble
import sklearn.linear_model
import Datasets

def synth_env(n, M, K, d):
    """Synthetic bandit environment similar to Vaswani et al, 2020."""
    basis = np.eye(d)
    basis[:, -1] = 1
    dataset = np.zeros([n,M,K,d])
    np.random.seed(2)
    for t in range(n):
        for i in range(M):
            # arm features in a unit (d - 2)-sphere
            x = np.random.randn(K, d - 1)
            x /= np.sqrt(np.square(x).sum(axis=1))[:, np.newaxis]
            x = np.hstack((x, np.ones((K, 1))))  # bias term
            x[: basis.shape[0], :] = basis
            dataset[t,i,:,:] = x    
        # parameter vector in a (d - 2)-sphere with radius 0.5
    theta = np.random.randn(d - 1)
    theta *= 0.5 / np.sqrt(np.square(theta).sum())
    theta = np.append(theta, [0.5])       
    return dataset, theta

def semi_synth_env(featureset):  
    """Semi-synthetic bandit environment generated from MSLR-WEB10K Dataset."""         
    dataset = Datasets.Datasets() 
    dataset.loadNpz('mslr') 
    allFeatures=scipy.sparse.vstack(dataset.features, format='csc')
    allTargets=np.hstack(dataset.relevances)
    if featureset is not None:
        allFeatures = allFeatures[:, featureset]
    lasso = sklearn.linear_model.Lasso(fit_intercept=False,
                    precompute=False, copy_X=False, 
                    max_iter=10000, tol=1e-4, warm_start=False, positive=False,
                    random_state=None, selection='random')
    lasso.fit(allFeatures, allTargets)
    theta = lasso.coef_ 
    return dataset, theta   

class FedLinBandit:
    """Federated Linear bandit."""

    def __init__(self, M, d, dataset, featureset, theta, dist = "normal", sigma=0.5):
        
        self.M = M
        self.d = d
        self.dataset = dataset
        self.featureset = featureset
        self.theta = theta
        self.dist = dist
        if self.dist == "normal":
            self.sigma = sigma
         
    def reset_random(self, seed):
        self.random = np.random.RandomState(seed)      

    def randomize(self, t):
        self.best_arm = np.zeros(self.M, dtype = int)
        featureAllAgent = []
        self.mu = []
        self.rt = []  
        for i in range(self.M):
            if self.dist == "real":
                queryPerAgent = int(len(self.dataset.queryMappings)/self.M) 
                currentQueryId = self.random.randint(queryPerAgent*i, queryPerAgent*(i+1))
                #currentQueryNo = self.dataset.queryMappings[currentQueryId]
                #currentTargets = self.dataset.relevances[currentQueryd]
                currentAllArms = self.dataset.docsPerQuery[currentQueryId]  
                if currentAllArms < 1:
                    print("No docs present")
                currentFeatures = self.dataset.features[currentQueryId]
            else:
                currentAllArms = self.dataset.shape[2]
                currentFeatures = self.dataset[t,i,:,:]     
            if self.featureset is not None:
                currentFeatures = currentFeatures[:, self.featureset] 
            currentAllRewards = currentFeatures.dot(self.theta)    
            self.best_arm[i] = np.argmax(currentAllRewards)
            self.mu.append(currentAllRewards)
            if self.dist == "real":
                normalizedFeatures = currentFeatures.toarray() / np.linalg.norm(currentFeatures.toarray(), axis = 1, keepdims = True)
            else:
                normalizedFeatures = currentFeatures / np.linalg.norm(currentFeatures, axis = 1, keepdims = True)
            featureAllAgent.append(normalizedFeatures) 
            
            if self.dist == "normal":
                self.rt.append(currentAllRewards + self.sigma * self.random.randn(currentAllArms))
            elif self.dist == "bernoulli":
                self.rt.append(currentAllRewards + (self.random.rand(currentAllArms) < currentAllRewards).astype(float))
            elif self.dist == "beta":
                self.rt.append(currentAllRewards + self.random.beta(4 * currentAllRewards, 4 * (1 - currentAllRewards)))
            elif self.dist == "real":
                self.rt.append(currentAllRewards)    
        return featureAllAgent
                      
    def reward(self, arm): # instantaneous reward of the arm
        RewardAllAgent = np.zeros(self.M)
        for i in range(self.M):
            RewardAllAgent[i] = self.rt[i] [arm[i]]
        return RewardAllAgent

    def regret(self, arm): # instantaneous regret of the arm
        RegretAllAgent = np.zeros(self.M)
        for i in range(self.M):
            RegretAllAgent[i] = self.rt[i] [self.best_arm[i]] - self.rt[i] [arm[i]]
        return RegretAllAgent
        
    def pregret(self, arm): # expected regret of the arm
        PRegretAllAgent = np.zeros(self.M)
        for i in range(self.M):
            PRegretAllAgent[i] = self.mu[i] [self.best_arm[i]] - self.mu[i] [arm[i]]
        return PRegretAllAgent

    def print(self):
        if self.dist == "normal":
            return "Synth Fed Lin bandit: %d dimensions, %d agents" % (self.d, self.M)
        else:
            return "Semi-synth Fed lin bandit: %d dimensions, %d agents" % (self.d, self.M)     
