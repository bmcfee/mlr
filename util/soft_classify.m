function Ypredict = soft_classify(W, test_k, Xtrain, Ytrain, Xtest, Testnorm)
% Ypredict = soft_classify(W, test_k, Xtrain, Ytrain, Xtest, Testnorm)
%
%       W       = d-by-d positive semi-definite matrix
%       test_k  = k-value to use for KNN
%       Xtrain  = d-by-n matrix of training data
%       Ytrain  = n-by-1 vector of training labels
%       Xtest   = d-by-m matrix of testing data
%       Testnorm= m-by-#kernels vector of k(i,i) for each i in test
%

    addpath('cuttingPlane', 'distance', 'feasible', 'initialize', 'loss', ...
            'metricPsi', 'regularize', 'separationOracle', 'util');

    [d, nTrain, nKernel] = size(Xtrain);
    nTest       = size(Xtest, 2);
    test_k      = min(test_k, nTrain);

    if nargin < 7
        Testnorm = [];
    end

    % Build the distance matrix
    [D, I] = mlr_test_distance(W, Xtrain, Xtest, Testnorm);

    % Compute label agreement
    Ypredict  = histc(Ytrain(I(1:test_k,:)), unique(Ytrain)');

end


function [D,I] = mlr_test_distance(W, Xtrain, Xtest, Testnorm)

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
        D = mlr_test_distance_raw(Xtrain, Xtest, Testnorm);

    elseif size(W,1) == d && size(W,2) == d
        % We're in a full-projection case
        D = setDistanceFullMKL([Xtrain Xtest], W, nTrain + (1:nTest), 1:nTrain);

    elseif size(W,1) == d && size(W,2) == nKernel
        % We're in a simple diagonal case
        D = setDistanceDiagMKL([Xtrain Xtest], W, nTrain + (1:nTest), 1:nTrain);

    elseif size(W,1) == nKernel && size(W,2) == nKernel && size(W,3) == nTrain
        % We're in DOD mode
        D = setDistanceDODMKL([Xtrain Xtest], W, nTrain + (1:nTest), 1:nTrain);

    else
        % Error?
        error('Cannot determine metric mode.');

    end
    
    D       = full(D(1:nTrain, nTrain + (1:nTest)));
    [v,I]   = sort(D, 1);
end



function D = mlr_test_distance_raw(Xtrain, Xtest, Testnorm)

    [d, nTrain, nKernel] = size(Xtrain);
    nTest = size(Xtest, 2);

    if isempty(Testnorm)
        % Not in kernel mode, compute distances directly
        D = 0;
        for i = 1:nKernel
            D = D + setDistanceDiag([Xtrain(:,:,i) Xtest(:,:,i)], ones(d,1), ...
                                    nTrain + (1:nTest), 1:nTrain);
        end
    else
        % We are in kernel mode
        D = sparse(nTrain + nTest, nTrain + nTest);
        for i = 1:nKernel
            Trainnorm = diag(Xtrain(:,:,i));
            D(1:nTrain, nTrain + (1:nTest)) = D(1:nTrain, nTrain + (1:nTest)) ...
                +  bsxfun(@plus, Trainnorm, bsxfun(@plus, Testnorm(:,i)', -2 * Xtest(:,:,i)));
        end
    end
end

