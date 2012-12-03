function Perf = mlr_test(W, test_k, Xtrain, Ytrain, Xtest, Ytest)
%   Perf = mlr_test(W, test_k, Xtrain, Ytrain, Xtest, Ytest)
%
%       W       = d-by-d positive semi-definite matrix
%       test_k  = vector of k-values to use for KNN/Prec@k/NDCG
%       Xtrain  = d-by-n matrix of training data
%       Ytrain  = n-by-1 vector of training labels
%                   OR
%                 n-by-2 cell array where
%                   Y{q,1} contains relevant indices (in 1..n) for point q
%                   Y{q,2} contains irrelevant indices (in 1..n) for point q
%       Xtest   = d-by-m matrix of testing data
%       Ytest   = m-by-1 vector of training labels, or m-by-2 cell array 
%               
%
%   The output structure Perf contains the mean score for:
%       AUC, KNN, Prec@k, MAP, MRR, NDCG,
%   as well as the effective dimensionality of W, and
%   the best-performing k-value for KNN, Prec@k, and NDCG.
%

    Perf        = struct(                       ...
                            'AUC',      [],     ...
                            'KNN',      [],     ...
                            'PrecAtK',  [],     ...
                            'MAP',      [],     ...
                            'MRR',      [],     ...
                            'NDCG',     [],     ...
                            'dimensionality',   [],     ...
                            'KNNk',     [],     ...
                            'PrecAtKk', [],     ...
                            'NDCGk',    []     ...
                );

    [d, nTrain, nKernel] = size(Xtrain);
    % Compute dimensionality of the learned metric
    Perf.dimensionality = mlr_test_dimension(W, nTrain, nKernel);
    test_k      = min(test_k, nTrain);

    if nargin > 5
        % Knock out the points with no labels
        if ~iscell(Ytest)
            Ibad                = find(isnan(Ytrain));
            Xtrain(:,Ibad,:)    = inf;
        end

        % Build the distance matrix
        [D, I] = mlr_test_distance(W, Xtrain, Xtest);
    else
        % Leave-one-out validation

        if nargin > 4 
            % In this case, Xtest is a subset of training indices to test on
            testRange = Xtest;
        else
            testRange = 1:nTrain;
        end
        Xtest       = Xtrain(:,testRange,:);
        Ytest       = Ytrain(testRange);

        % compute self-distance
        [D, I]  = mlr_test_distance(W, Xtrain, Xtest);
        % clear out the self-link (distance = 0)
        I       = I(2:end,:);
        D       = D(2:end,:);
    end
    
    nTest       = length(Ytest);

    % Compute label agreement
    if ~iscell(Ytest)
        % First, knock out the points with no label
        Labels  = Ytrain(I);
        Agree   = bsxfun(@eq, Ytest', Labels); 

        % We only compute KNN error if Y are labels
        [Perf.KNN, Perf.KNNk] = mlr_test_knn(Labels, Ytest, test_k);
    else
        if nargin > 5
            Agree   = zeros(nTrain, nTest);
        else
            Agree   = zeros(nTrain-1, nTest);
        end
        for i = 1:nTest
            Agree(:,i) = ismember(I(:,i), Ytest{i,1});
        end

        Agree = reduceAgreement(Agree);
    end

    % Compute AUC score
    Perf.AUC    = mlr_test_auc(Agree);

    % Compute MAP score
    Perf.MAP    = mlr_test_map(Agree);

    % Compute MRR score
    Perf.MRR    = mlr_test_mrr(Agree);

    % Compute prec@k
    [Perf.PrecAtK, Perf.PrecAtKk] = mlr_test_preck(Agree, test_k);

    % Compute NDCG score
    [Perf.NDCG, Perf.NDCGk] = mlr_test_ndcg(Agree, test_k);

end


function [D,I] = mlr_test_distance(W, Xtrain, Xtest)

    % CASES:
    %   Raw:                        W = []
    
    %   Linear, full:               W = d-by-d
    %   Single Kernel, full:        W = n-by-n
    %   MKL, full:                  W = n-by-n-by-m

    %   Linear, diagonal:           W = d-by-1
    %   Single Kernel, diagonal:    W = n-by-1
    %   MKL, diag:                  W = n-by-m
    %   MKL, diag-off-diag:         W = m-by-m-by-n
    
    [d, nTrain, nKernel] = size(Xtrain);
    nTest = size(Xtest, 2);

    if isempty(W)
        % W = []  => native euclidean distances
        D = mlr_test_distance_raw(Xtrain, Xtest);

    elseif size(W,1) == d && size(W,2) == d
        % We're in a full-projection case
        D = setDistanceFullMKL([Xtrain Xtest], W, nTrain + (1:nTest), 1:nTrain);

    elseif size(W,1) == d && size(W,2) == nKernel
        % We're in a simple diagonal case
        D = setDistanceDiagMKL([Xtrain Xtest], W, nTrain + (1:nTest), 1:nTrain);

    else
        % Error?
        error('Cannot determine metric mode.');

    end
    
    D       = full(D(1:nTrain, nTrain + (1:nTest)));
    [v,I]   = sort(D, 1);
end



function dimension = mlr_test_dimension(W, nTrain, nKernel)

    % CASES:
    %   Raw:                        W = []
    
    %   Linear, full:               W = d-by-d
    %   Single Kernel, full:        W = n-by-n
    %   MKL, full:                  W = n-by-n-by-m

    %   Linear, diagonal:           W = d-by-1
    %   Single Kernel, diagonal:    W = n-by-1
    %   MKL, diag:                  W = n-by-m
    %   MKL, diag-off-diag:         W = m-by-m-by-n
    
    
    if size(W,1) == size(W,2)
        dim = [];
        for i = 1:nKernel
            [v,d]   = eig(0.5 * (W(:,:,i) + W(:,:,i)'));
            dim     = [dim ; abs(real(diag(d)))];
        end
    else
        dim       = W(:);
    end

    cd      = cumsum(dim) / sum(dim);
    dimension = find(cd >= 0.95, 1);
    if isempty(dimension)
        dimension = 0;
    end
end

function [NDCG, NDCGk] = mlr_test_ndcg(Agree, test_k)

    nTrain = size(Agree, 1);

    Discount        = zeros(1, nTrain);
    Discount(1:2)   = 1;

    NDCG   = -Inf;
    NDCGk  = 0;
    for k = test_k
        
        Discount(3:k)   = 1 ./ log2(3:k);
        Discount        = Discount / sum(Discount);

        b = mean(Discount * Agree);
        if b > NDCG
            NDCG = b;
            NDCGk = k;
        end
    end
end

function [PrecAtK, PrecAtKk] = mlr_test_preck(Agree, test_k)

    PrecAtK        = -Inf;
    PrecAtKk       = 0;
    for k = test_k
        b   = mean( mean( Agree(1:k, :), 1 ) );
        if b > PrecAtK
            PrecAtK = b;
            PrecAtKk = k;
        end
    end
end

function [KNN, KNNk] = mlr_test_knn(Labels, Ytest, test_k)

    KNN        = -Inf;
    KNNk       = 0;
    for k = test_k
        % FIXME:  2012-02-07 16:51:59 by Brian McFee <bmcfee@cs.ucsd.edu>
        %   fix these to discount nans 

        b   = mean( mode( Labels(1:k,:), 1 ) == Ytest');
        if b > KNN
            KNN    = b;
            KNNk   = k;
        end
    end
end

function MAP = mlr_test_map(Agree);

    nTrain      = size(Agree, 1);
    MAP         = bsxfun(@ldivide, (1:nTrain)', cumsum(Agree, 1));
    MAP         = mean(sum(MAP .* Agree, 1)./ sum(Agree, 1));
end

function MRR = mlr_test_mrr(Agree);

        nTest = size(Agree, 2);
        MRR        = 0;
        for i = 1:nTest
            MRR    = MRR  + (1 / find(Agree(:,i), 1));
        end
        MRR        = MRR / nTest;
end

function AUC = mlr_test_auc(Agree)

    TPR             = cumsum(Agree,     1);
    FPR             = cumsum(~Agree,    1);

    numPos          = TPR(end,:);
    numNeg          = FPR(end,:);

    TPR             = mean(bsxfun(@rdivide, TPR, numPos),2);
    FPR             = mean(bsxfun(@rdivide, FPR, numNeg),2);
    AUC             = diff([0 FPR']) * TPR;
end


function D = mlr_test_distance_raw(Xtrain, Xtest)

    [d, nTrain, nKernel] = size(Xtrain);
    nTest = size(Xtest, 2);

        % Not in kernel mode, compute distances directly
        D = 0;
        for i = 1:nKernel
            D = D + setDistanceDiag([Xtrain(:,:,i) Xtest(:,:,i)], ones(d,1), ...
                                    nTrain + (1:nTest), 1:nTrain);
        end
end

function A = reduceAgreement(Agree)
    nPos = sum(Agree,1);
    nNeg = sum(~Agree,1);

    goodI = find(nPos > 0 & nNeg > 0);
    A = Agree(:,goodI);
end
