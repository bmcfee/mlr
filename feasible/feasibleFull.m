function W = feasibleFull(W)
%
% W = feasibleFull(W)
%
% Projects a single d*d matrix onto the PSD cone
%

    global FEASIBLE_COUNT;
    FEASIBLE_COUNT = FEASIBLE_COUNT + 1;

    W = psd_sparse(W);
end

