%% clear everything
clc
clear
close all


%% constants
alpha = 0.1; % learning rate



%% read data set
D=xlsread('dataset.xlsx');
maxn = 100;
features = zeros(maxn,maxn);

for i=1:size(D, 1)
    v = D(i,1);
    u = D(i,1);
    features(v,u) = 1;
    features(u,v) = 1;
    
end

features





