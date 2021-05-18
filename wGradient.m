function grad = wGradient(i,j,L,a,z,Nlayers,y)
    % return gradient of MSE loss function to w(L,i,j) ; 
    % w(L,i,j) : weight of a(L,i) to a(L-1,j)
    % z : w(L,i).a(L-1,j) + b(L,i)
    % Nlayers : number of layers

    if L==Nlayers
        grad = exp(-z(L,i)) / (1 - exp(-z(L,i)))^2; % gradient of sigmoid
    else
        grad = 1-tanh(z(L,i))^2; % gradient of tanh
    end

    grad = grad * a(L-1,j) * 2*(a(L,i)-y);

end


