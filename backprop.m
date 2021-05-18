function [w,b] = backprop(a,z,y,layers,w,b,alpha)
    % return gradient of MSE loss function to w(L,i,j) ; 
    % w(L,i,j) : weight of a(L,i) to a(L-1,j)
    % z : w(L,i).a(L-1,j) + b(L,i)
    % layers : number of nodes in each layer
    % alpha : learning rate

    Nlayers = numel(layers);
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


