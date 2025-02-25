%clear
%% estimate marginal posteririor 1i
% x(n,k)= sum(d) y(n,d) r (d,k)

function x = computeX()
 
    % Retrieve y and r from the base workspace
    Y = evalin('base', 'Y');
    R = evalin('base', 'R');
    
    % Perform the matrix multiplication
    x = Y * R;
end

function plotHistogramX(k)
    
    x = computeX();
    figure;
    % Plot the histogram of all elements in x (flatten x into a vector)
    histogram(x(:,k));  
    title(sprintf('Histogram of x the latent variable at k = %d',k));
    xlabel('n values');
    ylabel('Frequency');
    
    
end
%plotHistogramX(10);

function plotNormal()
    % Compute x using your custom function (assuming it exists)
    x = computeX();  
    k = 50;
    % Extract the data vector from the selected column
    data = x(:, k);
    
    % Create a new figure and plot a normalized histogram
    figure;
    histogram(data, 'Normalization', 'pdf');  
    hold on;
    
    % Define your desired normal distribution parameters
    mu = 0;        % Mean value (change as needed)
    sigma = 1;     % Standard deviation (change as needed)
    scale = 1;     % Scaling factor (adjust magnitude)
    
    % Generate x values for plotting the normal distribution
    x_values = linspace(min(data), max(data), 100);
    
    % Calculate the normal pdf values using your parameters
    y_pdf = scale * normpdf(x_values, mu, sigma);
    
    % Plot the normal distribution curve
    plot(x_values, y_pdf, 'r-', 'LineWidth', 2);
    
    % Add title and labels
    title(sprintf('Histogram of x at k = %d, with gaussian fit of var = 1,and scale = 1', k));
    xlabel('N');
    ylabel('Probability Density');
    
    hold off;
end


function plotPairwise(k1,k2)
    x = computeX();
    datak1 = x(:, k1);
    datak2 = x(:, k2);
    
    figure;
    histogram2(datak1,datak2)
    hold on;

    title(sprintf('2D Histogram of k = %d & %d', k1,k2));
    xlabel('k1');
    ylabel('k2');
    hold off;
end

function plotNormalised(k1,k2)
    X = computeX();
    datak1 = X(:, k1);
    datak2 = X(:, k2);
    
    % Choose the number of bins (adjust as needed)
    numBins =30;   
    [counts, edges1, edges2] = histcounts2(datak1, datak2, numBins);
    
    % Normalize each row of the histogram to estimate p(xk2 | xk1)
    p_cond = zeros(size(counts));  

    for i = 1:size(counts,1)
        rowSum = sum(counts(i, :));
        if rowSum > 0
            p_cond(i, :) = counts(i, :) / rowSum;
        end
    end
    
    % Create meshgrid for plotting. Use bin centers for a better representation.
    centers1 = (edges1(1:end-1) + edges1(2:end)) / 2;
    centers2 = (edges2(1:end-1) + edges2(2:end)) / 2;
    [Xgrid, Ygrid] = meshgrid(centers1, centers2);
    
    % Plot the conditional probability using imagesc or pcolor.
    figure;
    % imagesc expects x as columns and y as rows; note the transpose of p_cond.
    imagesc(centers1, centers2, p_cond'); 
    set(gca, 'YDir', 'normal'); % Correct the y-axis direction
    xlabel('x_{k1}');
    ylabel('x_{k2}');
    colorbar;
    title(sprintf('Estimated Conditional Distribution p(x_{k2= %d }|x_{k1= %d })',k2,k1));
    
    % OPTIONAL: Overlay contours for clarity
    hold on;
    contour(Xgrid, Ygrid, p_cond', 'LineColor', 'k');
    hold off;
end

%plotNormalised(40,30);


x = computeX();

plotIm(W);
%plotNormalised(47,90);

