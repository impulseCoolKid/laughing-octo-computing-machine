%func to optamise

function [f,df] = optamizationFunc(baArr,x,xj)

    K_length = 256;
    N_length = 32000;
    totalSum = double(0);

    for n = 0:N_length
        for j = 0:K_length
            if j ~= k
                sigma = a(k,j)*X(n,j)^2 + b(k); 
            end
        end
        totalSum = totalSum - log(sigma) - (X(n,k)^2)/(2*sigma^2);
    end




    K_length = 256;
    N_length = 32000;
    totalSum = double(0);

    for n = 0:N_length
        
        sigma  = Double(0);
        for j = 0:K_length
            if j ~= k
                sigma = a(k,j)*X(n,j)^2 + b(k); 
            end
            
        %+ log(1) - (1/2)*log(2*pi) %this doesnt seem important since
        %doesnt change
        totalSum = totalSum - log(sigma) - (X(n,k)^2)/(2*sigma^2);
            
            
        end
    end
end


% Define the function that computes the value and gradient using parameters P1 and P2
function [f, grad] = poly_log_fun_params(X, P1, P2)
    % Extract variables: x and y (with y > 0)
    x = X(1);
    y = X(2);

    % Compute the function value
    f = P1 * (x^4 - 3*x^3 + 2) + P2 * (y*log(y) + y^2);
    
    % Compute the gradients
    grad_x = P1 * (4*x^3 - 9*x^2);         % derivative with respect to x
    grad_y = P2 * (log(y) + 1 + 2*y);        % derivative with respect to y
    
    % Combine gradients into a vector
    grad = [grad_x; grad_y];
end

% Main script to perform the minimization using additional parameters

% Initial guess for the variables [x; y] (ensure y > 0 for the log term)
X0 = [2; 1];

% Set the maximum number of iterations (line searches or function evaluations)
max_iter = 100;

% Define parameters P1 and P2
P1 = 1.5;
P2 = 2.0;

% Call the minimize function using conjugate gradients.
% The extra parameters P1 and P2 are passed to the objective function.
[X, fX, iter] = minimize(X0, @poly_log_fun_params, max_iter, P1, P2);

% Display the results
fprintf('Minimum found at: x = %f, y = %f\n', X(1), X(2));
fprintf('Function value at minimum: f(x,y) = %f\n', fX(end));
fprintf('Number of iterations: %d\n', iter);