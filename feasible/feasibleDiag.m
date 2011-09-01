function W = feasibleDiag(W)
%
% W = feasibleDiag(W)
%
% Projects a single d*1 matrix onto the PSD cone
%
    W = max(0,W);
end

