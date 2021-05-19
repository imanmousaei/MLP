%% clear everything
clc
clear
close all


%% read data set
D=xlsread('dataset.xlsx');


%% vars
alpha = 0.1; % learning rate
layers = [size(D,2)-1, 10, 1]; % number of nodes in each layer
epochs = 20;
Nlayers = numel(layers);
maxL = max(size(D,2)-1,10);
N = size(D,1);
R = 0.8;
Ntrain = round(R*N);

a = zeros(Nlayers, maxL);
z = zeros(Nlayers, maxL);
b = zeros(Nlayers, maxL);
w = zeros(Nlayers, maxL, maxL);


%% init w
for L=2:Nlayers
   for i=1:layers(L)
       for j=1:layers(L-1)
           w(L,i,j) = rand * 10;
       end
   end
end


%% train
for ep=1:epochs
    for row=1:Ntrain
        %% fill first layer with training data
        a(1,1:size(D,2)-1) = D(row,1:end-1);
        y = D(row,end) - 1;
        

        %% evaluate activations
        for L=2:Nlayers
           for i=1:layers(L)
               tmp = b(L,i);
               for j=1:layers(L-1)
                   tmp = tmp + w(L,i,j)*a(L-1,j);
               end
               z(L,i) = tmp;
               a(L,i) = actFcn(tmp,L,Nlayers);
           end
        end

        %% back propagation
        for L=Nlayers:2
            for i=1:layers(L)
                for j=1:layers(L-1)
                   wGrad = wGradient(i,j,L,a,z,Nlayers,y);
                   w(L,i,j) = w(L,i,j) - alpha * wGrad;
                end

                bGrad = bGradient(i,L,a,z,Nlayers,y);
                b(L,i) = b(L,i) - alpha * bGrad;
            end
        end
    end
end

%% test
for row=Ntrain+1:N
    %% fill first layer with training data
    a(1,1:size(D,2)-1) = D(row,1:end-1);
    y = D(row,end) - 1;


    %% evaluate activations
    for L=2:Nlayers
       for i=1:layers(L)
           tmp = b(L,i);
           for j=1:layers(L-1)
               tmp = tmp + w(L,i,j)*a(L-1,j);
           end
           z(L,i) = tmp;
           a(L,i) = actFcn(tmp,L,Nlayers);
       end
    end

    yHat(row) = a(3,1);
    yReal(row) = y;


end


plotconfusion(categorical(yReal),categorical(yHat));



