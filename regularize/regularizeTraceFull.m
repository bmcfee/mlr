function r = regularizeTraceFull(W, X, gradient)
%
% r = regularizeTraceFull(W, X, gradient)
%
%
    d = length(W);

    if gradient
        r = eye(d) / d;
    else
        r = trace(W) / d;
    end
end
