function [Y, Loss] = separationOraclePrecAtK(q, D, pos, neg, k)
%
%   [Y,Loss]  = separationOraclePrecAtK(q, D, pos, neg, k)
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
%   Loss is the loss for Y, in this case, 1-Prec@k(Y)


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


    % If we don't have enough positive (or negative) examples, scale k down
    k = min([k, numPos, numNeg]);

    % Algorithm:
    %   For each precision score in 0, 1/k, 2/k, ... 1
    %       Calculate maximum discriminant score for that precision level
    Precision       = (0:(1/k):1)';
    Discriminant    = zeros(k+1, 1);
    NegsBefore      = zeros(numPos, k+1);

    % For 0 precision, all positives go after the first k negatives

    NegsBefore(:,1) = k + binarysearch(Vpos, Vneg(k+1:end));

    Discriminant(1) = Vpos * (numNeg - 2 * NegsBefore(:,1)) + numPos * cVneg(end) ...
                    - 2 * sum(cVneg(NegsBefore((NegsBefore(:,1) > 0),1)));



    % For precision (a-1)/k, swap the (a-1)'th positive doc
    %   into the top (k-a) negative docs

    for a = 2:(k+1)
        NegsBefore(:,a) = NegsBefore(:,a-1);

        % We have a-1 positives, and k - (a-1) negatives
        NegsBefore(a-1, a) = binarysearch(Vpos(a-1), Vneg(1:(k-a+1)));

        % There were NegsBefore(a-1,a-1) negatives before (a-1)
        % Now there are NegsBefore(a,a-1)
        Discriminant(a) = Discriminant(a-1) ...
                        + 2 * (NegsBefore(a-1,a-1) - NegsBefore(a-1,a)) * Vpos(a-1);

        if NegsBefore(a-1,a-1) > 0
            Discriminant(a) = Discriminant(a) + 2 * cVneg(NegsBefore(a-1,a-1));
        end
        if NegsBefore(a-1,a) > 0
            Discriminant(a) = Discriminant(a) - 2 * cVneg(NegsBefore(a-1,a));
        end
    end

    % Normalize discriminant scores
    Discriminant    = Discriminant / (numPos * numNeg);
    [s, x]          = max(Discriminant - Precision);

    % Now we know that there are x-1 relevant docs in the max ranking
    % Construct Y from NegsBefore(x,:)

    Y = nan * ones(n,1);
    Y((1:numPos)' + NegsBefore(:,x)) = Ipos;
    Y(isnan(Y))                     = Ineg;

    % Compute loss for this list
    Loss        = 1 - Precision(x);
end

