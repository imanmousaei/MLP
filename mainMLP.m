%% clear everything
clc
clear
close all


%% read data set
D=xlsread('dataset.xlsx');


%% constants
alpha = 0.1; % learning rate
layers = [size(D,2)-1, 10, 1]; % number of nodes in each layer
Nlayers = numel(layers);


%% evaluate activations
for L=2:Nlayers
   for i=1:layers(L)
       tmp = b(L,i);
       for j=1:layers(L)
           tmp = tmp + w(L,i,j)*a(L-1,j);
       end
       a(L,i) = actFcn(tmp,L,Nlayers);
   end
end







