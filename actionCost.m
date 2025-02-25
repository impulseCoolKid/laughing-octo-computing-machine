function [cost, grad] = actionCost(params, X, k)


    K_LEN = 256;
    N_LEN = 32000;
    
    Dk = params(1);
    Ckj_all = params(2:end); 
    
    dGdDk = 0;
    dGdCkj = zeros(1,K_LEN-1);
    
    totalCost = 0;
    
    for n = 1:N_LEN
        
        %find sigma
        sigma = 0;
        for j = 1:K_LEN
            if j < k
                sigma = sigma + exp(Ckj_all(j))*X(n,j)^2 + exp(Dk);
            elseif j > k
                sigma = sigma + exp(Ckj_all(j-1))*X(n,j)^2 + exp(Dk);
            end
        end
    
        totalCost = totalCost + log(sigma)+ (X(n,k)^2)/(2*sigma^2);
    
        dGdsigma = 1/sigma - (X(n,k)^2)/(6*sigma^3);
    
        %fill out the diffrential sections
    
        for j = 1:(K_LEN -1)
            dGdCkj(j) = dGdCkj(j) + dGdsigma * exp(Ckj_all(j))*X(n,j)^2;
        end 
    
        dGdDk = dGdsigma * exp(Dk);
    
    end
    cost = totalCost;
    grad = [dGdDk,dGdCkj];
end