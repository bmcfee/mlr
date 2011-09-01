function [Y, Loss] = separationOracleNDCG(q, D, pos, neg, k)
%
%   [Y,Loss]  = separationOracleNDCG(q, D, pos, neg, k)
%
%   q   = index of the query point
%   D   = the current distance matrix
%   pos = indices of relevant results for q
%   neg = indices of irrelevant results for q
%   k   = length of the list to consider 
%
%   Y is a permutation 1:n corresponding to the maximally
%   violated constraint
%
%   Loss is the loss for Y, in this case, 1-NDCG(Y)


    % First, sort the documents in descending order of W'Phi(q,x)
    % Phi = - (X(q) - X(x)) * (X(q) - X(x))'
    
    % Sort the positive documents
    ScorePos        = - D(pos, q);
    [Vpos, Ipos]    = sort(full(ScorePos'), 'descend');
    Ipos            = pos(Ipos);
    
    % Sort the negative documents
    ScoreNeg        = - D(neg, q);
    [Vneg, Ineg]    = sort(full(ScoreNeg'), 'descend');
    Ineg            = neg(Ineg);

    % Now, solve the DP for the interleaving

    numPos  = length(pos);
    numNeg  = length(neg);
    n       = numPos + numNeg;

    % From Chakrabarti (KDD08)
    k       = min(k, numPos);

    cVneg   = cumsum(Vneg);

    Discount = zeros(k, 1);
    Discount(1:2) = 1;
    Discount(3:k) = 1./ log2(3:k);

    DCGstar         = sum(Discount);


    % Pre-compute the loss table
    LossTab     = padarray(  hankel(- Discount / DCGstar), ...
                            max(0, [numNeg numPos] - k), 0, 'post');
    if sum(size(LossTab) > [numNeg, numPos])
        LossTab     = LossTab(1:numNeg, 1:numPos);
    end

    % 2010-01-17 09:13:41 by Brian McFee <bmcfee@cs.ucsd.edu>
    % initialize the score table

    pcVneg = [0 cVneg];
    % Pre-compute cellScore
    cellValue = bsxfun(@times, Vpos / (numPos * numNeg), numNeg - 2 * ((1:numNeg)-1)');
    cellValue = bsxfun(@plus, (2 * pcVneg(1:numNeg) - cVneg(end))' / (numPos * numNeg), cellValue);
    cellValue = cellValue + LossTab;

    S       = zeros(numNeg, numPos);
    P       = zeros(numNeg, numPos);
    
    % Initialize first column
    P(:,1) = 1;
    S(:,1) = cellValue(:,1);

    % Initialize first row
    P(1,:) = 1;
    S(1,:) = cumsum(cellValue(1,:));
    
    % For the rest, use the recurrence

    for g = 2:numPos
        [m, pointer]    = cummax(S(:,g-1));
        P(:,g)          = pointer;
        S(:,g)          = m' + cellValue(:,g);
    end

    % Now reconstruct the permutation from the DP table
    Y           = nan * ones(n,1);
    [m,p]       = max(S(:,numPos));
    
    Loss        = 1 + LossTab(p,numPos);
    
    NegsBefore      = zeros(numPos,1);
    NegsBefore(numPos)  = p-1;

    for a = numPos:-1:2
        p               = P(p,a);
        NegsBefore(a-1) = p-1;
        Loss            = Loss + LossTab(p,a-1);
    end
    Y((1:numPos)' + NegsBefore)     = Ipos;
    Y(isnan(Y))                     = Ineg;

end
