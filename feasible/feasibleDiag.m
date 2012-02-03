function W = feasibleDiag(W)
%
% W = feasibleDiag(W)
%
% Projects a single d*1 matrix onto the PSD cone
%
    global FEASIBLE_COUNT;
    FEASIBLE_COUNT = FEASIBLE_COUNT + 1;
    W = max(0,W);
end

