## Main function used in the paper

import itertools
import scipy
import time
import os
import sys
import scipy.stats
from scipy.stats import norm
import numpy as np
import scipy.sparse
import scipy.linalg
import Agents
import Environments
from Environments import FedLinBandit
import Utils


if __name__ == "__main__":
    base_dir = os.path.join(".", "Results", "FedLin")  # Directory to save results
    
    # Environments 
    environments = [

        # Synthetic bandit instance with 100 agents (for Figure 2.a, 2.b)
        (FedLinBandit, {"dist": "normal", "sigma": 0.5}, 100, "Normal(M=100)"),

        # Real-data bandit instance with 10 agents (for Figure 2.c)
        (FedLinBandit, {"dist": "real"}, 10, "Real(M=10)")
    ]

    for env_def in environments:
        env_class, env_params, M, env_name = env_def[0], env_def[1], env_def[2], env_def[-1]
        print("================== running environment", env_name, "==================")
        
        if env_params["dist"] == "real":
            num_runs = 25 # number of parallel runs
            n = 25000 # number of rounds 
            period_size = 25
            time_idx = period_size * np.array([np.arange(n//period_size)+1,]*num_runs).transpose() 
            rVar = 1
            cScale = 0.1
            anchorURLFeatures, bodyTitleDocFeatures = Utils.get_feature_sets("MSLR")
            # bandit parameter trained on title features 
            # (change to bodyTitleDocFeatures for training on body features)
            featureset = anchorURLFeatures 
            if featureset is not None:
                d = len(featureset)
            else:
                d = len(anchorURLFeatures) + len(bodyTitleDocFeatures)
            dataset, theta = Environments.semi_synth_env(featureset)    
        else: 
            num_runs = 25 # number of parallel runs
            n = 10000 # number of rounds
            period_size = 10
            time_idx = period_size * np.array([np.arange(n//period_size)+1,]*num_runs).transpose()  
            rVar = env_params["sigma"]
            cScale = 1
            K = 100
            d = 10
            featureset = None
            dataset, theta = Environments.synth_env(n, M, K, d)

        # Create environment
        env = env_class(M, d, dataset, featureset, theta, **env_params)
        res_dir = os.path.join(base_dir, env_name)
        os.makedirs(res_dir, exist_ok=True)
        
        # Algorithms to compare
        algorithms = [

                # Comparison for Figure 2.a
                (Agents.SDPFedLinUCBAmp, {"epsilon": 0.0001,"delta": 0.0001,"BatchSize": 25,"alpha": 0.01,"crs": cScale,"R": rVar},\
                     "SDPFedLinUCBAmp(eps=0.0001,delta=0.0001,B=25)"),
                (Agents.SDPFedLinUCBAmp, {"epsilon": 0.001,"delta": 0.0001,"BatchSize": 25,"alpha": 0.01,"crs": cScale,"R": rVar},\
                     "SDPFedLinUCBAmp(eps=0.001,delta=0.0001,B=25)"),
                (Agents.LDPFedLinUCB, {"epsilon": 0.0001,"delta": 0.0001,"BatchSize": 25,"alpha": 0.01,"crs": cScale,"R": rVar},\
                     "SDPFedLinUCBAmp(eps=0.0001,delta=0.0001,B=25)"),
                (Agents.LDPFedLinUCB, {"epsilon": 0.001,"delta": 0.0001,"BatchSize": 25,"alpha": 0.01,"crs": cScale,"R": rVar},\
                     "SDPFedLinUCBAmp(eps=0.001,delta=0.0001,B=25)"),          

                # Comparison for Figure 2.b
                (Agents.LDPFedLinUCB, {"epsilon": 0.2,"delta": 0.1,"BatchSize": 25,"alpha": 0.01,"crs": cScale,"R": rVar},\
                    "LDPFedLinUCB(eps=0.2,delta=0.1,B=25)"),
                (Agents.SDPFedLinUCBVecSum, {"epsilon": 0.2,"delta": 0.1,"scale": 1,"BatchSize": 25,"alpha": 0.01,"crs": cScale,"R": rVar},\
                    "SDPFedLinUCBVecSum(eps=0.2,delta=0.1,B=25)"),    

                #Comparison for Figure 2.c
                (Agents.FedLinUCB, {"BatchSize": 25,"alpha": 0.01,"crs": cScale,"R": rVar},\
                     "FedLinUCB(B=25)"),
                (Agents.LDPFedLinUCB, {"epsilon": 0.2,"delta": 0.1,"BatchSize": 25,"alpha": 0.01,"crs": cScale,"R": rVar},\
                      "LDPFedLinUCB(eps=0.2,delta=0.1,B=25)"),
                (Agents.LDPFedLinUCB, {"epsilon": 1,"delta": 0.1,"BatchSize": 25,"alpha": 0.01,"crs": cScale,"R": rVar},\
                     "LDPFedLinUCB(eps=1,delta=0.1,B=25)"),    
                (Agents.LDPFedLinUCB, {"epsilon": 5,"delta": 0.1,"BatchSize": 25,"alpha": 0.01,"crs": cScale,"R": rVar},\
                     "LDPFedLinUCB(eps=5,delta=0.1,B=25)")             
            ]      

        seeds = np.arange(num_runs)
        for alg_def in algorithms:
            alg_class, alg_params, alg_name = alg_def[0], alg_def[1], alg_def[-1] 
            fname = os.path.join(res_dir, alg_name)        
            if os.path.exists(fname):
                print('File exists. Will load saved file. Moving on to the next algorithm')
            else:
                regret, _, _ = Utils.evaluate(alg_class, alg_params, env, seeds, n, period_size)                
                cum_regret = regret.cumsum(axis=0)
                avg_regret = cum_regret/time_idx 
                np.savetxt(fname, avg_regret, delimiter=",") 