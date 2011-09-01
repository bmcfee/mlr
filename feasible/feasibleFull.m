function W = feasibleFull(W)
%
% W = feasibleFull(W)
%
% Projects a single d*d matrix onto the PSD cone
%

    [v,d]   = eig(0.5 * (W + W'));
    W       = v * bsxfun(@times, max(real(diag(d)),0), v');
end

