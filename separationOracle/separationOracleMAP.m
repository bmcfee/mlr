function [Y, Loss] = separationOracleMAP(q, D, pos, neg, k)
%
%   [Y,Loss]  = separationOracleMAP(q, D, pos, neg, k)
%
%   q   = index of the query point
%   D   = the current distance matrix
%   pos = indices of relevant results for q
%   neg = indices of irrelevant results for q
%   k   = length of the list to consider (unused in MAP)
%
%   Y is a permutation 1:n corresponding to the maximally
%   violated constraint
%
%   Loss is the loss for Y, in this case, 1-AP(Y)


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

    % Now, solve the DP for the interleaving

    numPos  = length(pos);
    numNeg  = length(neg);
    n       = numPos + numNeg;

    
    % Pre-generate the precision scores
%     H       = triu(1./bsxfun(@minus, (0:(numPos-1))', 1:n));
    H       = tril(1./bsxfun(@minus, 0:(numPos-1), (1:n)'));

    % Padded cumulative Vneg
    pcVneg  = cumsum([0  Vneg]);

    % Generate the discriminant scores
    H       = H + scoreChangeMatrix(Vpos, Vneg, n, pcVneg);

    % Cost of inserting the first + at position b
    P       = zeros(size(H));

    % Now recurse
    for a = 2:numPos

        % Fill in the back-pointers
        [m,p]           = cummax(H(:,a-1));
        % The best point is the previous row, up to b-1
        H(a:n,a)        = H(a:n,a) + (a-1)/a .* m(a-1:n-1)';
        P(a+1:n,a)      = p(a:n-1);
        P(a,a)          = a-1;
    end

    % Now reconstruct the permutation from the DP table
    Y           = nan * ones(n,1);
    [m,p]       = max(H(:,numPos));
    Y(p)        = Ipos(numPos);

    for a = numPos:-1:2
        p       = P(p,a);
        Y(p)    = Ipos(a-1);
    end
    Y(isnan(Y)) = Ineg;

    % Compute loss for this list
    Loss        = 1 - AP(Y, pos);
end

function C = scoreChangeMatrix(Vpos, Vneg, n, pcVneg)
    numNeg  = length(Vneg);
    numPos  = length(Vpos);

    % Inserting the a'th relevant document at position b
    % There are (b - (a - 1)) negative docs before a
    % And (numNeg - (b - (a - 1))) negative docs after
    %
    % The change in score is proportional to: 
    %
    %   sum_{negative j}  (Vpos(a) - Vneg(j)) * y_{aj}
    %
    %   = (numNeg - (b - (a - 1))) * Vpos(a)            # Negatives after a
    %   - (cVneg(end) - cVneg(b - (a - 1)))             Weight of negs after a
    %   - (b - (a - 1)) * Vpos(a)                       # Negatives before a
    %   + cVneg(b - (a - 1))                            Weight of negs before a
    %
    %   Rearrange:
    %
    %   (numNeg - 2 * (b - a + 1)) * Vpos(a)
    %   - cVneg(end) + 2 * cVneg(b - a + 1)
    %
    % Valid range of a:  1:numPos
    % Valid range of b:  a:n

    D   = bsxfun(@plus, 1-(1:numPos), (1:n)');
    C   = numNeg - 2 * D;
    C   = bsxfun(@times, Vpos, C);

    D(D < 1)                = 1;
    D(D > length(pcVneg))   = length(pcVneg);

    %     FIXME:  2011-01-28 21:13:37 by Brian McFee <bmcfee@cs.ucsd.edu>
    % brutal hack to get around matlab's screwy matrix reshaping 
    if numPos == 1
        pcVneg = pcVneg';
    end

    C   = C + 2 * pcVneg(D) - pcVneg(end);

    % Normalize
    C   = bsxfun(@ldivide, (1:numPos) * numNeg, C);

    % -Inf out the infeasible regions
    C   = C - triu(Inf * bsxfun(@gt, (1:numPos), (1:n)'),1);


end

function x = AP(Y, pos)
    % Indicator for relevant documents
    rel     = ismember(Y, pos);

    % Prec@k for all k
    Prec    = cumsum(rel)' ./ (1:length(Y));

    % Prec@k averaged over relevant positions
    x       = mean(Prec(rel));
end
