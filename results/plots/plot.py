#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 13:06:38 2021

@author: niravshah
"""
from plotly.offline import plot
import plotly.graph_objs as go

def plot_result_box(x, f1_linear, f1_knn, f1_naive, title):
    
    
    
    data=[]
    fig = go.Figure()
    data.append(go.Box(x=x, y=f1_linear, name="Linear Regression"))
    data.append(go.Box(x=x, y=f1_knn, name="K Nearest Neighbour"))
    data.append(go.Box(x=x, y=f1_naive, name="Naive Bayes"))
    
    
    layout = go.Layout(title=title, xaxis=dict(tickvals=[1,2,3],ticktext=[1,2,4]), xaxis_title="Sampling Rate", yaxis_title="F1 Score",)
    fig = go.Figure(data=data, layout=layout)
    plot(fig,filename=str(title) + ".html",auto_open=False)
    

x = [1, 1, 2, 2, 3, 3]
LR_under_sampling   = [0.305, 0.432, 0.300, 0.442, 0.326, 0.467] 
LR_hybrid_sampling  = [0.303, 0.432 ,0.298, 0.446, 0.325, 0.466]
LR_over_sampling    = [0.302, 0.437, 0.299, 0.448, 0.326, 0.471]

NN_under_sampling   = [0.774, 0.531, 0.819, 0.561, 0.870, 0.619] 
NN_hybrid_sampling  = [0.849, 0.516, 0.878, 0.564, 0.910, 0.633]
NN_over_sampling    = [0.958, 0.599, 0.966, 0.639, 0.975, 0.698]

NB_under_sampling   = [0.320, 0.219, 0.318, 0.200, 0.337, 0.196] 
NB_hybrid_sampling  = [0.315, 0.206, 0.316, 0.192, 0.333, 0.191]
NB_over_sampling    = [0.314, 0.200, 0.319, 0.197, 0.335, 0.194]


plot_result_box(x, LR_under_sampling, NN_under_sampling, NB_under_sampling, "Random Undersampling Results")
plot_result_box(x, LR_hybrid_sampling, NN_hybrid_sampling, NB_hybrid_sampling, "Random Hybridsampling Results")
plot_result_box(x, LR_over_sampling, NN_over_sampling, NB_over_sampling, "Random Oversampling Results")

# plot_result_box([(0.305,0.432), (0.300, 0.442), (0.326, 0.467)],
#                 [(0.305,0.432), (0.300, 0.442), (0.326, 0.467)],
#                 [(0.305,0.432), (0.300, 0.442), (0.326, 0.467)], "Random Hybrid Sampling Results")
# plot_result_box([(0.305,0.432), (0.300, 0.442), (0.326, 0.467)],
#                 [(0.305,0.432), (0.300, 0.442), (0.326, 0.467)],
#                 [(0.305,0.432), (0.300, 0.442), (0.326, 0.467)], "Random Over Sampling Results")