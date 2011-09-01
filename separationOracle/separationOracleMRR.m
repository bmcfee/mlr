function [Y, Loss] = separationOracleMRR(q, D, pos, neg, k)
%
%   [Y,Loss]  = separationOracleMRR(q, D, pos, neg, k)
%
%   q   = index of the query point
%   D   = the current distance matrix
%   pos = indices of relevant results for q
%   neg = indices of irrelevant results for q
%   k   = length of the list to consider (unused in MRR)
%
%   Y is a permutation 1:n corresponding to the maximally
%   violated constraint
%
%   Loss is the loss for Y, in this case, 1-MRR(Y)


    % First, sort the documents in descending order of W'Phi(q,x)
    % Phi = - (X(q) - X(x)) * (X(q) - X(x))'
    
    % Sort the positive documents
    ScorePos        = - D(pos,q);
    [Vpos, Ipos]    = sort(full(ScorePos'), 'descend');
    Ipos            = pos(Ipos);
    
    % Sort the negative documents
    ScoreNeg        = -D(neg,q);
    [Vneg, Ineg]    = sort(full(ScoreNeg'), 'descend');
    Ineg            = neg(Ineg);

    % Now, solve the DP for the interleaving

    numPos  = length(pos);
    numNeg  = length(neg);
    n       = numPos + numNeg;

    cVpos           = cumsum(Vpos);
    cVneg           = cumsum(Vneg);


    % Algorithm:
    %   For each RR score in 1/1, 1/2, ..., 1/(numNeg+1)
    %       Calculate maximum discriminant score for that precision level
    MRR                     = ((1:(numNeg+1)).^-1)';

    
    Discriminant            = zeros(numNeg+1, 1);
    Discriminant(end)       = numPos * cVneg(end) - numNeg * cVpos(end);
    
    % For the rest of the positions, we're interleaving one more negative
    % example into the 2nd-through-last positives
    offsets                 = 1 + binarysearch(Vneg, Vpos(2:end));
    
    % How many of the remaining positives go before Vneg(a)?
    NegsBefore              = -bsxfun(@ge, offsets, (1:length(Vpos))');

    % For the last position, all negatives come before all positives
    NegsBefore(:,numNeg+1)  = numNeg;

    Discriminant(1:numNeg) = -2 * (offsets .* Vneg - cVpos(offsets));
    Discriminant = sum(Discriminant) - cumsum(Discriminant) + Discriminant;


    % Normalize discriminant scores
    Discriminant    = Discriminant / (numPos * numNeg);
    [s, x]          = max(Discriminant - MRR);

    % Now we know that there are x-1 relevant docs in the max ranking
    % Construct Y from NegsBefore(x,:)

    Y = nan * ones(n,1);
    Y((1:numPos)' + sum(NegsBefore(:,x:end),2)) = Ipos;
    Y(isnan(Y))                     = Ineg;

    % Compute loss for this list
    Loss        = 1 - MRR(x);
end

