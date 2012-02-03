function mlr_demo()

    display('Loading Wine data');
    load Wine;

    % z-score the input dimensions
    display('z-scoring features');
    X = zscore(X')';

    [d,n] = size(X);
    
    % Generate a random training/test split
    display('Generating a 80/20 training/test split');
    P       = randperm(n);
    Xtrain  = X(:,P(1:floor(0.8 * n)));
    Ytrain  = Y(P(1:floor(0.8*n)));
    Xtest   = X(:,P((1+floor(0.8*n)):end));
    Ytest   = Y(P((1+floor(0.8*n)):end));


    % Optimize W for AUC
    C = 1e5;
    display(sprintf('Training with C=%.2e, Delta=mAP', C));
    [W, Xi, Diagnostics] = mlr_train(Xtrain, Ytrain, C, 'map');
%     [W, Xi, Diagnostics] = mlr_train_primal(Xtrain, Ytrain, C, 'map');

    display('Test performance in the native (normalized) metric');
    mlr_test(eye(d), 3, Xtrain, Ytrain, Xtest, Ytest)

    display('Test performance with MLR metric');
    mlr_test(W, 3, Xtrain, Ytrain, Xtest, Ytest)

    % Scatter-plot
    figure;
    subplot(1,2,1), drawData(eye(d), Xtrain, Ytrain, Xtest, Ytest), title('Native metric (z-scored)');
    subplot(1,2,2), drawData(W, Xtrain, Ytrain, Xtest, Ytest), title('Learned metric (MLR-mAP)');

    Diagnostics

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
