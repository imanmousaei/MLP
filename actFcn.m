function f = actFcn(x,L,Nlayers)
    if L == Nlayers
        f = 1/(1+exp(-x)); % sigmoid
    else
        f = tanh(x); 
    end

end


