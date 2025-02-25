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

function topComponents =TopComponentsForSelectedK(a, W, selectedK, numTop)

    if nargin < 4
        numTop = 10;
    end

    % Determine the total number of components from W.
    K = size(W,2);
    
    % Extract the row for the selected component from a.
    a_row = a(selectedK, :);  % 1 x (K-1)

    % Sort the a-values in descending order.
    [~, sortedIdx] = sort(a_row, 'descend');
    
    % Map sorted column indices to actual component indices.
    topComponents = zeros(numTop+1, 1);
    topComponents(1) = selectedK;
    for i = 2:(numTop+1)
        j = sortedIdx(i);
        if j < selectedK
            actual_k = j;
        else
            actual_k = j + 1;
        end
        topComponents(i) = actual_k;
    end

end

topcomp = TopComponentsForSelectedK(expa,W,40,10);
disp(topcomp);
plotIm(W(:,topcomp));
