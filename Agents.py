## Classes of Algorithms compared in the paper

import time
import os
import sys
import numpy as np

def binary_first_one(num, cnt):  # index of first one in binary representation
    if num % 2 == 1:
        return cnt
    else:
        return(binary_first_one(num // 2, cnt + 1))  

def binary_all_one(num, N):  # binary representation in reverse order
    RevBinRep = np.zeros(N, dtype=int)
    cnt = 0
    while num > 0:
        if num % 2 == 1:
            RevBinRep[cnt] = 1
        num = num // 2
        cnt += 1 
    return RevBinRep

def R1d(x, g, L, b, p):             # 1-D Randomizer
    x_bar  = np.floor(x*g/L)
    eta1 = np.random.binomial(1, x*g/L-x_bar)
    eta2 = np.random.binomial(b,p)
    return x_bar + eta1 + eta2    

def randomizer(vec, mat, g, L, b, p): # Vector Randomizer
    d = np.shape(vec)[0]
    assert d == np.shape(mat)[0]
    assert d == np.shape(mat)[1]
    privVec =  np.zeros(d)
    privMat =  np.zeros([d,d])
    for i in range(d):
        privVec[i] = R1d(vec[i] + L, g, L, b, p)
        for j in range(i,d):
            privMat[i,j] = R1d(mat[i,j] + L, g, L, b, p)
    return privVec, privMat 

def shuffler(privVecAll, privMatAll):  # Shuffler
    N = np.shape(privVecAll)[0]
    assert N == np.shape(privMatAll)[0]
    d = np.shape(privVecAll)[1]
    assert d == np.shape(privMatAll)[1]
    assert d == np.shape(privMatAll)[2]
    shuffledVec = np.zeros(d)
    shuffledMat = np.zeros([d, d])
    for n in range(N):
        for i in range(d):
            shuffledVec[i] += privVecAll[n,i]
            for j in range(i,d):
                shuffledMat[i,j] += privMatAll[n,i,j]
    return shuffledVec, shuffledMat   

def A1d(y, g, L, b, p, N):         # 1-D Analyzer
    return L/g * (y - p * b * N)         

def analyzer(shuffledVec, shuffledMat, g, L, b, p, N):   # Vector Analyzer
    d = np.shape(shuffledVec)[0]
    assert d == np.shape(shuffledMat)[0]
    assert d == np.shape(shuffledMat)[1]
    outVec = np.zeros(d)
    outMat = np.zeros([d, d])
    for i in range(d):
        outVec[i] = A1d(shuffledVec[i], g, L, b, p, N) - N * L
        for j in range(i,d):
            outMat[i,j] = A1d(shuffledMat[i,j], g, L, b, p, N) - N * L 
            outMat[j,i] = outMat[i,j]
    return outVec, outMat        
  

class FedLinBanditAlg:
    def __init__(self, env, n, params):
        self.M = env.M
        self.d = env.d
        self.n = n
        for attr, val in params.items():
            setattr(self, attr, val)
        self.Gram = np.zeros([self.M, self.d, self.d]) 
        self.Bias = np.zeros([self.M, self.d])
        self.GramSync = np.zeros([self.d, self.d])
        self.BiasSync = np.zeros(self.d)
        self.BatchNo = 1

        self.L = np.ones(self.M) # feature norm bound
        self.S = np.sqrt(self.d) # parameter norm bound
        self.N = int(np.floor(np.log2(self.n // self.BatchSize))) + 1

class FedLinUCB(FedLinBanditAlg):    # non-private Federated LCB (based on Abbasi-Yadkori et al, 2011)
    def __init__(self, env, n, params):
        super(FedLinUCB, self).__init__(env, n, params)
        
        self.Lambda = np.sqrt(self.M)
    
    def confidence_ellipsoid_width(self, t):
        width = np.sqrt(self.Lambda)*self.S+\
                self.R*np.sqrt(self.d*np.log(1+t*self.M*np.square(self.L)/(self.d*self.Lambda))+np.log(1/self.alpha))  
        return width
    
    def get_arm(self, t, X):
        arm = np.zeros(self.M, dtype=int)
        cew = self.crs * self.confidence_ellipsoid_width(t)
        for i in range(self.M):
            Gram_inv_i = np.linalg.inv(self.Gram[i,:,:] + self.GramSync + self.Lambda * np.eye(self.d))   
            theta_hat_i = Gram_inv_i.dot(self.Bias[i,:] + self.BiasSync)
            Xi = X[i][:,:]
            mu_i = Xi.dot(theta_hat_i) + cew[i] * np.sqrt((Xi.dot(Gram_inv_i) * Xi).sum(axis = 1)) #UCBs
            arm[i] = np.argmax(mu_i)
        return arm  
        
    def update(self, t, arm, r, X):
        for i in range(self.M):
            xi = X[i][arm[i],:]
            self.Gram[i,:,:] += np.outer(xi, xi) 
            self.Bias[i,:] += xi * r[i] 
        
        if t == (self.BatchNo * self.BatchSize):
            self.GramSync += np.sum(self.Gram, axis = 0)
            self.BiasSync += np.sum(self.Bias, axis = 0) 
            self.Gram = np.zeros([self.M, self.d, self.d]) 
            self.Bias = np.zeros([self.M, self.d])
            self.BatchNo += 1    
            
    
    @staticmethod
    def print():
        return "FedLinUCB" 

class SDPFedLinUCBAmp(FedLinUCB):  # Shuffle model with privacy amplification (Thm 5.3)
    def __init__(self, env, n, params):
        super(SDPFedLinUCBAmp, self).__init__(env, n, params) 
        
        self.noise = np.max(self.L)*np.sqrt(self.N*np.log(1/self.delta)*np.log(1+self.N/(self.n*self.delta))\
            *np.log(self.M*self.N/self.delta))/(self.epsilon*np.sqrt(self.M))
        self.Lambda = np.sqrt(self.M*self.N)*self.noise\
            *(np.sqrt(self.N + np.log(1/self.alpha)) + np.sqrt(self.d))
            
        self.AlphaGramAll = np.zeros([self.M,self.N,self.d,self.d])
        self.AlphaBiasAll = np.zeros([self.M,self.N,self.d])
        self.PrivAlphaGramSyncAll = np.zeros([self.N,self.d,self.d])
        self.PrivAlphaBiasSyncAll = np.zeros([self.N,self.d])                   
    
    def update(self, t, arm, r, X): 

        for i in range(self.M):
            xi = X[i][arm[i],:]
            self.Gram[i,:,:] += np.outer(xi, xi) 
            self.Bias[i,:] += xi * r[i] 
            
        if t == (self.BatchNo * self.BatchSize):
            FirstOneIdx = binary_first_one(self.BatchNo, 0) 
            PrivAlphaGramSync = np.zeros([self.d, self.d])
            PrivAlphaBiasSync = np.zeros(self.d)
            for i in range(self.M): 
                
                self.AlphaGramAll[i,FirstOneIdx,:,:] = np.sum(self.AlphaGramAll[i,0:FirstOneIdx,:,:], axis=0) + self.Gram[i,:,:]
                self.AlphaBiasAll[i,FirstOneIdx,:] = np.sum(self.AlphaBiasAll[i,0:FirstOneIdx,:], axis=0) + self.Bias[i,:]
                
                NoiseGram_i = np.random.normal(0,self.noise,[self.d,self.d])  
                NoiseGram_i = (NoiseGram_i + np.transpose(NoiseGram_i)) / 2.0
                NoiseBias_i = np.random.normal(0,self.noise,self.d)
                
                PrivAlphaGram_i = self.AlphaGramAll[i,FirstOneIdx,:,:] + NoiseGram_i
                PrivAlphaBias_i = self.AlphaBiasAll[i,FirstOneIdx,:] + NoiseBias_i
            
                PrivAlphaGramSync += PrivAlphaGram_i
                PrivAlphaBiasSync += PrivAlphaBias_i

            self.PrivAlphaGramSyncAll[FirstOneIdx,:,:] = PrivAlphaGramSync
            self.PrivAlphaBiasSyncAll[FirstOneIdx,:] = PrivAlphaBiasSync
            
            AllOneIdx = binary_all_one(self.BatchNo, self.N)
            self.GramSync = np.sum(self.PrivAlphaGramSyncAll[AllOneIdx==1,:,:], axis=0)
            self.BiasSync = np.sum(self.PrivAlphaBiasSyncAll[AllOneIdx==1,:], axis=0)

            self.Gram = np.zeros([self.M, self.d, self.d]) 
            self.Bias = np.zeros([self.M, self.d])
            self.BatchNo += 1
      
    @staticmethod
    def print():
        return "SDPFedLinUCB"

class LDPFedLinUCB(SDPFedLinUCBAmp):    # silo-level local model (Thm. 5.1)
    def __init__(self, env, n, params):
        super(LDPFedLinUCB, self).__init__(env, n, params) 

        self.noise = np.max(self.L)*np.sqrt(self.N*(np.log(1/self.delta)+self.epsilon))/self.epsilon
        self.Lambda = np.sqrt(self.M*self.N)*self.noise\
             *(np.sqrt(self.N + np.log(1/self.alpha)) + np.sqrt(self.d))
            
    @staticmethod
    def print():
        return "LDPFedLinUCB"   

class SDPFedLinUCBVecSum(SDPFedLinUCBAmp):  # Shuffle model with vector sum (Thm. 5.5)
    def __init__(self, env, n, params):
        super(SDPFedLinUCBVecSum, self).__init__(env, n, params)  
        
        self.noise = np.max(self.L)*np.sqrt(self.N*np.log(1/self.delta))\
                *np.log(self.N*self.d**2/self.delta)/self.epsilon     
        self.Lambda = np.sqrt(self.N)*self.noise*(np.sqrt(self.N + np.log(1/self.alpha)) + np.sqrt(self.d)) 

        self.GramAll = np.zeros([self.M,self.n,self.d,self.d])
        self.BiasAll = np.zeros([self.M,self.n,self.d])

        self.PrivGramSyncAll = np.zeros([self.N,self.d,self.d])
        self.PrivBiasSyncAll = np.zeros([self.N,self.d])             

    def update(self, t, arm, r, X): 

        for i in range(self.M):
            xi = X[i][arm[i],:]
            currentBias = xi * r[i]
            currentGram = np.outer(xi, xi)
            self.Gram[i,:,:] += currentGram 
            self.Bias[i,:] += currentBias 

            self.GramAll[i,t-1,:,:] = currentGram
            self.BiasAll[i,t-1,:] = currentBias
                        
        if t == (self.BatchNo * self.BatchSize):
             
            FirstOneIdx = binary_first_one(self.BatchNo, 0)

            t_end = self.BatchNo * self.BatchSize
            t_start = t_end - 2**FirstOneIdx * self.BatchSize 
            dataSize = (t_end - t_start) * self.M  
            
            g = np.maximum(np.maximum(2*np.sqrt(dataSize), self.d), 4)
            b = self.scale* g**2 *(np.log(self.d**2/self.delta))**2/(self.epsilon**2 * dataSize)
            p = 0.25 
            
            currentPrivBiasAll = np.zeros([dataSize,self.d])
            currentPrivGramAll = np.zeros([dataSize,self.d,self.d]) 
            dataCnt = 0
            
            for i in range(self.M):
                for j in range(t_start, t_end): 
                    Bias_ij = self.BiasAll[i,j,:]
                    Gram_ij = self.GramAll[i,j,:,:]
                    currentPrivBiasAll[dataCnt,:], currentPrivGramAll[dataCnt,:,:]\
                        = randomizer(Bias_ij, Gram_ij, g, np.max(self.L), b, p)
                    dataCnt += 1
            assert dataCnt == dataSize        

            currentShuffledBias, currentShuffledGram\
                 = shuffler(currentPrivBiasAll, currentPrivGramAll)       
            currentOutBias, currentOutGram\
                 = analyzer(currentShuffledBias, currentShuffledGram, g, np.max(self.L), b, p, dataSize)

            self.PrivGramSyncAll[FirstOneIdx,:,:] = currentOutGram 
            self.PrivBiasSyncAll[FirstOneIdx,:] = currentOutBias
            
            AllOneIdx = binary_all_one(self.BatchNo, self.N)
            self.GramSync = np.sum(self.PrivGramSyncAll[AllOneIdx==1,:,:], axis=0)
            self.BiasSync = np.sum(self.PrivBiasSyncAll[AllOneIdx==1,:], axis=0)

            self.Gram = np.zeros([self.M, self.d, self.d]) 
            self.Bias = np.zeros([self.M, self.d])
            self.BatchNo += 1
   
      
    @staticmethod
    def print():
        return "SDPFedLinUCBVecSum"                                      