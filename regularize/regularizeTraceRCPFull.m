function r = regularizeTraceRCPFull(W, X, gradient)
%
% r = regularizeTraceRCPFull(W, X, gradient)
%
%
    d = length(W);

    if gradient
%         r = eye(d) / d;
        r = 0;
    else
        r = trace(W) / d;
    end
end
