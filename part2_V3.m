clear; close all;
load('representational.mat'); % Loads Y, R, W
import checkgrad.*;

% this page is about writing test functions 
function x = computeX()
 
    % Retrieve y and r from the base workspace
    Y = evalin('base', 'Y');
    R = evalin('base', 'R');
    
    % Perform the matrix multiplication
    x = Y * R;
end

function test_checkgrad()

    % Initial point for testing
    c =  ones(255, 1);  
    d = 0;              
    params = [b; a]; 
    x = computeX();
    k = 50;
    
    % Perturbation epsilon for finite differences
    eps = 1e-6;
    
    % Call checkgrad with the test function handle.
    % The checkgrad function is expected to print the partial derivatives,
    % the finite difference approximations, and return the relative error.
    rel_error = checkgradStudent(params, eps,x,k);
    %checkgradStudent(params, e,x,k);
    % Display the result
    fprintf('Relative error from checkgrad: %e\n', rel_error);
end

function test_minimize()
    a =  zeros(255, 1);  
    b = 0;              
    params = [b; a]; 
    x = computeX();
    k = 50;
     
    len = 100;
    
    [X_opt, fX, iter] = minimize(params, @actionCost, len,x,k);
    
    % Display the results.
    fprintf('Optimal solution (X):\n');
    disp(X_opt);
    
    %fprintf('Function value progression (fX):\n');
    %disp(fX);
    
    fprintf('Number of iterations (line searches) used: %d\n', iter);
end

%test_checkgrad(); 
%rel error is 1, these seems to be more likely a code error than the real 
% rel error

%%the big cheese
function [expa, expb] = run_minimization_for_ks()
    % Preallocate arrays for storing optimized values.
    K_LEN = 256;
    num_a = K_LEN-1;
    a_results = zeros(K_LEN, num_a);  % Each row for the 'a' vector from one run.
    b_results = zeros(K_LEN, 1);        % Vector for the corresponding 'b' values.
    
    x = computeX();
    % Loop over each k value.
    for k = 1:K_LEN
        % Set up initial parameters as in the provided code.
        a = zeros(num_a, 1);  
        b = 0;              
        params = [b; a];
        len = 100;
        
        % Call the minimization function.
        % It is assumed that 'minimize' and 'actionCost' are defined.
        [X_opt, ~, iter] = minimize(params, @actionCost, len, x, k);
        
        % Display the number of iterations used for this run.
        fprintf('For k = %d, iterations used: %d\n', k, iter);
        
        % Separate b and a from X_opt.
        % Assume X_opt(1) is b and X_opt(2:end) is the 255x1 vector a.
        b_results(k) = X_opt(1);
        a_results(k, :) = X_opt(2:end)';  % Transpose to store as a row.
    end
    
    % Apply the exponential function to each element.
    expa = exp(a_results);
    expb = exp(b_results);
end

[expa, expb] = run_minimization_for_ks();

save('trainedModel.mat', 'expa', 'expb'); % Saves A and B to .mat file