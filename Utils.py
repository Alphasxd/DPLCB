## Utility functions used in the experiments

import itertools
import scipy
import time
import os
import sys
import numpy as np

def get_feature_sets(datasetName):
    """Extract title and body features of query-document pairs in MSLR-WEB10K datase"""
    anchorURL = [0]
    bodyDoc = [0]
    
    if datasetName.startswith('MSLR'):
        for i in range(25):
            anchorURL.extend([5*i+2, 5*i+4])
            bodyDoc.extend([5*i+1, 5*i+3, 5*i+5])
        anchorURL.extend([126, 127, 128, 129, 130, 131])
        bodyDoc.extend([132,133])    
    else:
        print("[ERR] Unknown dataset. Use MSLR", flush=True)
        sys.exit(0)
        
    return anchorURL, bodyDoc

def evaluate_one(Alg, params, env, n, period_size):
    """One run of a bandit algorithm."""
    alg = Alg(env, n, params)
    regret = np.zeros(n // period_size)
    reward = np.zeros(n // period_size)
    for t in range(n):   
        features = env.randomize(t) # generate state
        arm = alg.get_arm(t + 1, features) # take action
        alg.update(t + 1, arm, env.reward(arm), features) # update model and regret
        regret_at_t = np.sum(env.regret(arm))  
        reward_at_t =  np.sum(env.reward(arm))     
        regret[t // period_size] += regret_at_t 
        reward[t // period_size] += reward_at_t      
    return regret, reward, alg

def evaluate(Alg, params, env, seeds, n, period_size, printout=True):
    """Multiple runs of a bandit algorithm."""
    if printout:
        print("Evaluating %s" % Alg.print(), end="")
    start = time.time()
    num_exps = len(seeds)
    regret = np.zeros((n // period_size, num_exps))
    reward = np.zeros((n // period_size, num_exps))
    alg = num_exps * [None]
    dots = np.linspace(0, num_exps - 1, 100).astype(int)
    for i in range(num_exps):
        print('Env number:', i)
        env.reset_random(seeds[i])        
        output = evaluate_one(Alg, params, env, n, period_size)
        regret[:, i] = output[0]
        reward[:, i] = output[1]
        alg[i] = output[2]
        if i in dots and printout:
            print(".", end="")
    if printout:
        print(" %.1f seconds" % (time.time() - start))        
        total_regret = regret.sum(axis=0)
        print("Regret: %.2f +/- %.2f (median: %.2f, max: %.2f)" %
            (total_regret.mean(), total_regret.std() / np.sqrt(num_exps),
            np.median(total_regret), total_regret.max()))
    return regret, reward, alg
