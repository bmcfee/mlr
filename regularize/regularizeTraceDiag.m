function r = regularizeTraceDiag(W, X, gradient)
%
% r = regularizeTraceDiag(W, X, gradient)
%
%
    d = length(W);

    if gradient
        r = ones(d,1) / d;
    else
        r = sum(W) / d;
    end
end
