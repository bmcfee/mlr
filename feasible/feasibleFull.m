function W = feasibleFull(W)
%
% W = feasibleFull(W)
%
% Projects a single d*d matrix onto the PSD cone
%

    global FEASIBLE_COUNT;
    FEASIBLE_COUNT = FEASIBLE_COUNT + 1;

    [v,d]   = eig(0.5 * (W + W'));
    W       = v * bsxfun(@times, max(real(diag(d)),0), v');
end

