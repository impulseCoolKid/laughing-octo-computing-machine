function d = checkgradStudent(params, e,x,k);

% checkgrad checks the derivatives in a function, by comparing them to finite
% differences approximations. The partial derivatives and the approximation
% are printed and the norm of the diffrence divided by the norm of the sum is
% returned as an indication of accuracy.
%
% usage: checkgrad('f', X, e, P1, P2, ...)
%
% where X is the argument and e is the small perturbation used for the finite
% differences. and the P1, P2, ... are optional additional parameters which
% get passed to f. The function f should be of the type 
%
% [fX, dfX] = f(X, P1, P2, ...)
%
% where fX is the function value and dfX is a vector of partial derivatives.
%
% Carl Edward Rasmussen, 2001-08-01.



[y dy] = actionCost(params, x, k);                         % get the partial derivatives dy

dh = zeros(length(params),1) ;
for j = 1:length(params)
  dx = zeros(length(params),1);
  dx(j) = dx(j) + e;                               % perturb a single dimension
  y2 = actionCost((params + e), x, k);
  dx = -dx ;
  y1 = actionCost((params + e), x, k);
  dh(j) = (y2 - y1)/(2*e);
end
disp(length(dy));
disp(dy);
disp(length(dh));
disp(dh);
%disp([dy dh])                                          % print the two vectors
d = norm(dh-dy)/norm(dh+dy);       % return norm of diff divided by norm of sum
