%% clear everything
clc
clear
close all


%% read data set
D=xlsread('dataset.xlsx');

%% constants
alpha = 0.1; % learning rate
layers = [size(D, 1),10,1]; % number of nodes in each layer








