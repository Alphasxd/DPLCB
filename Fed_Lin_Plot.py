## plot function used in the the paper 
# code is built on https://github.com/vaswanis/randucb


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import os

mpl.rcParams["axes.linewidth"] = 0.75
mpl.rcParams["grid.linewidth"] = 0.75
mpl.rcParams["lines.linewidth"] = 0.75
mpl.rcParams["patch.linewidth"] = 0.75
mpl.rcParams["xtick.major.size"] = 3
mpl.rcParams["ytick.major.size"] = 3

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.size"] = 14
mpl.rcParams["axes.titlesize"] = "large"
mpl.rcParams["legend.fontsize"] = "large"

#mpl.rcParams["text.usetex"] = True
# mpl.rcParams['text.latex.preamble'] = [
#    r'\usepackage{amsmath}',
#    r'\usepackage{amssymb}']

#print("matplotlib %s" % mpl.__version__)

def linestyle2dashes(style):
    if style == 'solid':
        return (10, ())
    elif style == 'dotted':
        return (0, (1, 1))
    elif style == 'loosely dotted':
        return (0, (1, 10))
    elif style == 'densely dotted':
        return (0, (1, 1))
    elif style == 'dashed':
        return (0, (5, 5))
    elif style == 'loosely dashed':
        return (0, (5, 10))
    elif style == 'densely dashed':
        return (0, (5, 1))
    elif style == 'dashdotted':
        return (0, (3, 5, 1, 5))
    elif style == 'loosely dashdotted':
        return (0, (3, 10, 1, 10))
    elif style == 'densely dashdotted':
        return (0, (3, 1, 1, 1))
    elif style == 'dashdotdotted':
        return (0, (3, 5, 1, 5, 1, 5))
    elif style == 'loosely dashdotdotted':
        return (0, (3, 10, 1, 10, 1, 10))
    elif style == 'densely dashdotdotted':
        return (0, (3, 1, 1, 1, 1, 1))


environments = [   

    # Synthetic bandit instance  with 100 agents  (Figure 2.a, 2.b) 
    ("Normal(M=100)", {"dist": "normal"}), 

    # Real-data bandit instance with 10 agents (Figure 2.c)
    ("Real_new(M=10)", {"dist": "real"})  
] 
algorithms = [

    # Comparison for Figure 2.a
    ("SDPFedLinUCB(eps=0.0001,delta=0.0001,B=25)", "red", "solid", "SDP-FedLinUCB($\epsilon=0.0001$)"),
    ("SDPFedLinUCB(eps=0.001,delta=0.0001,B=25)", "blue", "solid", "SDP-FedLinUCB($\epsilon=0.001$)"),
    ("LDPFedLinUCB(eps=0.0001,delta=0.0001,B=25)", "green", "solid", "LDP-FedLinUCB($\epsilon=0.0001$)"),
    ("LDPFedLinUCB(eps=0.001,delta=0.0001,B=25)", "black", "solid", "LDP-FedLinUCB($\epsilon=0.001$)"),
    
    # Comparison for Figure 2.b
    ("LDPFedLinUCB(eps=0.2,delta=0.1,B=25)", "blue", "solid", "LDP-FedLinUCB($\epsilon=0.2$)"),
    ("SDPFedLinUCBVecSum(eps=0.2,delta=0.1,B=25)", "red", "solid", "SDP-FedLinUCB($\epsilon=0.2$)"),

    # Comparison for Figure 2.c
    ("FedLinUCB(B=25)", "black", "solid", "FedLinUCB"), 
    ("LDPFedLinUCB(eps=0.2,delta=0.1,B=25)", "blue", "solid", "LDP-FedLinUCB($\epsilon=0.2$)"),
    ("LDPFedLinUCB(eps=1,delta=0.1,B=25)", "red", "solid", "LDP-FedLinUCB($\epsilon=1$)"),
    ("LDPFedLinUCB(eps=5,delta=0.1,B=25)", "green", "solid", "LDP-FedLinUCB($\epsilon=5$)"),
]
plot_name = "alg-comparison"

for fig_idx, env_def in enumerate(environments):
    env_name, env_params = env_def[0], env_def[1]
    res_dir = os.path.join("Results", "FedLin", env_name) # path to saved file
    if env_params["dist"] == "real":
        n = 25000
        period_size = 25
        step = period_size * np.arange(1, n // period_size + 1)
    else:
        n = 10000
        period_size = 10
        step = period_size * np.arange(1, n // period_size + 1) 
    
    plt.figure(figsize=(5, 3))
    for alg_idx, alg_def in enumerate(algorithms):
        alg_name, alg_color, alg_line, alg_label = alg_def[0], alg_def[1], alg_def[2], alg_def[3]
        
        fname = os.path.join(res_dir, alg_name)
        data = np.loadtxt(fname, delimiter=",")
        dataStd = data.std(axis=1) / np.sqrt(data.shape[1]) 

        plt.plot(step, data.mean(axis=1), alg_color, linestyle=linestyle2dashes(alg_line), label=alg_label)
        plt.fill_between(step,
                         data.mean(axis=1) - dataStd,
                         data.mean(axis=1) + dataStd,
                         color=alg_color, alpha=0.2, linewidth=0)
    
    plt.xlabel("Round")
    plt.ylabel("Time-average Regret")    
    plt.legend(loc= 'best', fontsize=12)
    plt.grid(True)       
        
    plot_dir = os.path.join(".", "Plots", "FedLin") # Directory to save plots
    os.makedirs(plot_dir, exist_ok=True)
    fig_name = env_name + plot_name + ".pdf"
    fname = os.path.join(plot_dir, fig_name)
    plt.savefig(fname, format = "pdf", dpi = 1200, bbox_inches="tight")
    
    plt.show(block=False)
    plt.pause(3) 
    plt.close("all")