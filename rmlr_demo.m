function rmlr_demo()

    display('Loading Wine data');
    load Wine;

    noisedim = 96;
    [d,n] = size(X); 
    d = d + noisedim;

    %create covariance matrix
    var = randn(noisedim); var = var'*var;
    noise = sqrtm(var)* randn(noisedim, n);
    X = [X; noise];
    
    % z-score the input dimensions
    display('z-scoring features');
    X = zscore(X')';

    display('Generating a 80/20 training/test split');
    P       = randperm(n);
    Xtrain  = X(:,P(1:floor(0.8 * n)));
    Ytrain  = Y(P(1:floor(0.8*n)));
    Xtest   = X(:,P((1+floor(0.8*n)):end));
    Ytest   = Y(P((1+floor(0.8*n)):end));
    
    C = 1e6;
    lam = 0.1;

    display(sprintf('Training with C=%.2e, Delta=MAP', C));
    %learn metric with R-MLR
    [W_rmlr, Xi, Diagnostics_rmlr] = rmlr_train(Xtrain, Ytrain, C, 'map',3,1,0,0,lam);
    
    %learn metric with MLR
    [W_mlr, Xi, Diagnostics_mlr] = mlr_train(Xtrain, Ytrain, C, 'map');
    
    display('Test performance in the native (normalized) metric');
    mlr_test(eye(d), 3, Xtrain, Ytrain, Xtest, Ytest)

    display('Test performance with R-MLR metric');
    mlr_test(W_rmlr, 3, Xtrain, Ytrain, Xtest, Ytest)

    display('Test performance with MLR metric');
    mlr_test(W_mlr, 3, Xtrain, Ytrain, Xtest, Ytest)

    % Scatter-plot
    figure;
    subplot(1,3,1), drawData(eye(d), Xtrain, Ytrain, Xtest, Ytest), title('Native metric (z-scored)');
    subplot(1,3,2), drawData(W_rmlr, Xtrain, Ytrain, Xtest, Ytest), title('Learned metric (RMLR)');
    subplot(1,3,3), drawData(W_mlr, Xtrain, Ytrain, Xtest, Ytest), title('Learned metric (MLR)');

    figure;imagesc(W_rmlr);
    figure;imagesc(W_mlr);
%     Diagnostics_rmlr
%     Diagnostics_mlr

end


function drawData(W, Xtrain, Ytrain, Xtest, Ytest);

    n = length(Ytrain);
    m = length(Ytest);

    if size(W,2) == 1
        W = diag(W);
    end
    % PCA the learned metric
    Z = [Xtrain Xtest];
    A = Z' * W * Z;
    [v,d] = eig(A);

    L = (d.^0.5) * v';
    L = L(1:2,:);

    % Draw training points
    hold on;
    trmarkers = {'b+', 'r+', 'g+'};
    tsmarkers = {'bo', 'ro', 'go'};
    for i = min(Ytrain):max(Ytrain)
        points = find(Ytrain == i);
        scatter(L(1,points), L(2,points), trmarkers{i});
        points = n + find(Ytest == i);
        scatter(L(1,points), L(2,points), tsmarkers{i});
    end
    legend({'Training', 'Test'});

end