clear; close all;
load('representational.mat');
load('trainedModel.mat')

function x = computeX()
 
    % Retrieve y and r from the base workspace
    Y = evalin('base', 'Y');
    R = evalin('base', 'R');
    
    % Perform the matrix multiplication
    x = Y * R;
end

function sigmak = genSigmaK(x, a, b, K_LEN)
    % Preallocate the output vector
    sigmak = zeros(K_LEN, 1);
    
    % Loop over each index k to compute sigma_k
    for k = 1:K_LEN
        temp_sum = 0;
        % Loop over each coefficient in row k of matrix a
        for j = 1:(K_LEN-1)
            % Compensate for the missing diagonal coefficient by adjusting the index for x.
            if j < k
                xi = x(j);
            else
                xi = x(j+1);
            end
            temp_sum = temp_sum + a(k, j) * (xi^2);
        end
        % Add the offset b(k) to complete sigma_k calculation.
        sigmak(k) = temp_sum + b(k);
    end
end

function C = genCMatrix(X, a, b)


    [n, K] = size(X);
    C = zeros(n, K);  % Preallocate the output matrix

    % Loop over each observation (each row of X)
    for i = 1:n
        % For each sigma index k (each variable)
        for k = 1:K
            sigma_k = 0;
            % Loop over the coefficients for sigma_k (length K-1)
            for j = 1:(K-1)
                if j < k
                    xi = X(i, j);
                else
                    xi = X(i, j+1);
                end
                sigma_k = sigma_k + a(k, j) * (xi^2);
            end
            sigma_k = sigma_k + b(k);  % Add the offset b(k)
            % Compute c_ik as defined
            C(i, k) = X(i, k) / sigma_k;
        end
    end
end

x = computeX();
C = genCMatrix(x, expa, expb);
k =90;

function histoGenerato(C,k)
   % Extract the data
data = C(:,k);

% Plot the histogram with probability density normalization
histogram(data, 'Normalization', 'pdf');
hold on;

% Define your desired normal distribution parameters
mu = 0;        % Mean value (adjust as needed)
sigma = 1;     % Standard deviation (adjust as needed)

% Generate x-values for plotting the normal distribution
x_values = linspace(min(data), max(data), 100);

% Calculate the normal PDF values
y_pdf = normpdf(x_values, mu, sigma);

% Plot the normal distribution curve
plot(x_values, y_pdf, 'r-', 'LineWidth', 2);

% Add title and labels
title(sprintf('Histogram of c_k, the latent variable at k = %d', k));
xlabel('Data values');
ylabel('Probability density');

hold off;
end
%histoGenerato(C,10);

function [excessKurtosisX, excessKurtosisC] = computeExcessKurtosis(X, C, k)
    excessKurtosisX = kurtosis(X(:, k)) - 3;
    excessKurtosisC = kurtosis(C(:, k)) - 3;
end

[excessKurtosisX, excessKurtosisC] = computeExcessKurtosis(x, C, k);
disp(excessKurtosisX);
disp(excessKurtosisC)

function plotConditionalDistributions(X, C, k1, k2, numBins)

    if nargin < 5
        numBins = 30;  % Default number of bins if not provided.
    end

    %% Process data for X
    datak1 = X(:, k1);
    datak2 = X(:, k2);
    
    % Define bin edges for datak1
    if min(datak1) == max(datak1)
        edges1 = [datak1(1)-1, datak1(1)+1];
    else
        edges1 = linspace(min(datak1), max(datak1), numBins+1);
    end
    
    % Define bin edges for datak2
    if min(datak2) == max(datak2)
        edges2 = [datak2(1)-1, datak2(1)+1];
    else
        edges2 = linspace(min(datak2), max(datak2), numBins+1);
    end
    
    % Compute the 2D histogram for X data
    [countsX, edges1, edges2] = histcounts2(datak1, datak2, edges1, edges2);
    
    % Normalize each row of the histogram to get the conditional density
    p_cond_x = zeros(size(countsX));
    for i = 1:size(countsX,1)
        rowSum = sum(countsX(i, :));
        if rowSum > 0
            p_cond_x(i, :) = countsX(i, :) / rowSum;
        end
    end
    
    % Compute bin centers for X data plotting
    centers1 = (edges1(1:end-1) + edges1(2:end)) / 2;
    centers2 = (edges2(1:end-1) + edges2(2:end)) / 2;
    [Xgrid, Ygrid] = meshgrid(centers1, centers2);
    
    %% Process data for C
    datac1 = C(:, k1);
    datac2 = C(:, k2);
    
    % Define bin edges for datac1
    if min(datac1) == max(datac1)
        edgesC1 = [datac1(1)-1, datac1(1)+1];
    else
        edgesC1 = linspace(min(datac1), max(datac1), numBins+1);
    end
    
    % Define bin edges for datac2
    if min(datac2) == max(datac2)
        edgesC2 = [datac2(1)-1, datac2(1)+1];
    else
        edgesC2 = linspace(min(datac2), max(datac2), numBins+1);
    end
    
    % Compute the 2D histogram for C data
    [countsC, edgesC1, edgesC2] = histcounts2(datac1, datac2, edgesC1, edgesC2);
    
    % Normalize each row to get the conditional density for C
    p_cond_c = zeros(size(countsC));
    for i = 1:size(countsC,1)
        rowSum = sum(countsC(i, :));
        if rowSum > 0
            p_cond_c(i, :) = countsC(i, :) / rowSum;
        end
    end
    
    % Compute bin centers for C data plotting
    centersC1 = (edgesC1(1:end-1) + edgesC1(2:end)) / 2;
    centersC2 = (edgesC2(1:end-1) + edgesC2(2:end)) / 2;
    [XgridC, YgridC] = meshgrid(centersC1, centersC2);
    
    %% Plotting
    figure;
    
    % Plot p(x_{k2}|x_{k1})
    subplot(1,2,1);
    imagesc(centers1, centers2, p_cond_x');
    set(gca, 'YDir', 'normal');
    xlabel(sprintf('x_{%d}', k1));
    ylabel(sprintf('x_{%d}', k2));
    colorbar;
    title(sprintf('p(x_{%d}|x_{%d})', k2, k1));
    hold on;
    contour(Xgrid, Ygrid, p_cond_x', 'LineColor', 'k');
    hold off;
    
    % Plot p(c_{k2}|c_{k1})
    subplot(1,2,2);
    imagesc(centersC1, centersC2, p_cond_c');
    set(gca, 'YDir', 'normal');
    xlabel(sprintf('c_{%d}', k1));
    ylabel(sprintf('c_{%d}', k2));
    colorbar;
    title(sprintf('p(c_{%d}|c_{%d})', k2, k1));
    hold on;
    contour(XgridC, YgridC, p_cond_c', 'LineColor', 'k');
    hold off;
end

%%
%plotConditionalDistributions(x, C, 47, 90, 30);
plotConditionalDistributions(x, C, 6, 90, 30);
