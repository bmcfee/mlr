function [Y, Loss] = separationOracleAUC(q, D, pos, neg, k)
%
%   [Y,Loss]  = separationOracleAUC(q, D, pos, neg, k)
%
%   q   = index of the query point
%   D   = the current distance matrix
%   pos = indices of relevant results for q
%   neg = indices of irrelevant results for q
%   k   = length of the list to consider (unused in AUC)
%
%   Y is a permutation 1:n corresponding to the maximally
%   violated constraint
%
%   Loss is the loss for Y, in this case, 1-AUC(Y)


    % First, sort the documents in descending order of W'Phi(q,x)
    % Phi = - (X(q) - X(x)) * (X(q) - X(x))'
    
    % Sort the positive documents
    ScorePos        = - D(pos,q);
    [Vpos, Ipos]    = sort(full(ScorePos'), 'descend');
    Ipos            = pos(Ipos);
    
    % Sort the negative documents
    ScoreNeg        = - D(neg,q);
    [Vneg, Ineg]    = sort(full(ScoreNeg'), 'descend');
    Ineg            = neg(Ineg);


    % How many pos and neg documents are we using here?
    numPos  = length(pos);
    numNeg  = length(neg);
    n       = numPos + numNeg;


    NegsBefore = sum(bsxfun(@lt, Vpos, Vneg' + 0.5),1);

    % Construct Y from NegsBefore
    Y                           = nan * ones(n,1);
    Y((1:numPos) + NegsBefore)  = Ipos;
    Y(isnan(Y))                 = Ineg;

    % Compute AUC loss for this ranking
    Loss = 1 - sum(NegsBefore) / (numPos * numNeg * 2);
end

