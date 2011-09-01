function r = regularizeTwoFull(W, X, gradient)
%
% r = regularizeTwoFull(W, X, gradient)
%
%
    d = length(W);

    if gradient
        r = W / d;
    else
        r = sum(sum(W.^2)) / (2 * d);
    end
end
