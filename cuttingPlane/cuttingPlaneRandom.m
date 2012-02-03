function [dPsi, M, SO_time] = cuttingPlaneRandom(k, X, W, Ypos, Yneg, batchSize, SAMPLES, ClassScores)
%
% [dPsi, M, SO_time] = cuttingPlaneRandom(k, X, W, Yp, Yn, batchSize, SAMPLES, ClassScores)
%
%   k           = k parameter for the SO
%   X           = d*n data matrix
%   W           = d*d PSD metric
%   Yp          = cell-array of relevant results for each point
%   Yn          = cell-array of irrelevant results for each point
%   batchSize   = number of points to use in the constraint batch
%   SAMPLES     = indices of valid points to include in the batch
%   ClassScores = structure for synthetic constraints
%
%   dPsi        = dPsi vector for this batch
%   M           = mean loss on this batch
%   SO_time     = time spent in separation oracle

    global SO PSI SETDISTANCE CPGRADIENT;

    [d,n]   = size(X);


    if length(SAMPLES) == n
        % All samples are fair game (full data)
        Batch   = randperm(n);
        Batch   = Batch(1:batchSize);
        D       = SETDISTANCE(X, W, Batch);

    else
        Batch   = randperm(length(SAMPLES));
        Batch   = SAMPLES(Batch(1:batchSize));

        Ito     = sparse(n,1);

        if isempty(ClassScores)
            for i = Batch
                Ito(Ypos{i}) = 1;
                Ito(Yneg{i}) = 1;
            end
            D       = SETDISTANCE(X, W, Batch, find(Ito));
        else
            D       = SETDISTANCE(X, W, Batch, 1:n);
        end
    end


    M       = 0;
    S       = zeros(n);
    dIndex  = sub2ind([n n], 1:n, 1:n);

    SO_time = 0;

    TS = zeros(batchSize, n);

    if isempty(ClassScores)
        parfor j = 1:batchSize
            i = Batch(j);
            if isempty(Yneg)
                Ynegative   = setdiff((1:n)', [i ; Ypos{i}]);
            else
                Ynegative   = Yneg{i};
            end
            SO_start        = tic();
                [yi, li]    =   SO(i, D, Ypos{i}, Ynegative, k);
            SO_time         = SO_time + toc(SO_start);
    
            M               = M + li /batchSize;
            TS(j,:)         = PSI(i, yi', n, Ypos{i}, Ynegative);
        end
        S(Batch,:)      = TS;
        S(:,Batch)      = S(:,Batch)    + TS';
        S(dIndex)       = S(dIndex)     - sum(TS, 1);
    else
        for j = 1:length(ClassScores.classes)
            c       = ClassScores.classes(j);
            points  = find(ClassScores.Y(Batch) == c);
            if ~any(points)
                continue;
            end

            Yneg    = find(ClassScores.Yneg{j});
            yp      = ClassScores.Ypos{j};

            for x = 1:length(points)
                i               = Batch(points(x));
                yp(i)           = 0;
                Ypos            = find(yp);
                SO_start        = tic();
                    [yi, li]    =   SO(i, D, Ypos, Yneg, k);
                SO_time         = SO_time + toc(SO_start);
    
                M               = M + li /batchSize;
                snew            = PSI(i, yi', n, Ypos, Yneg);
                S(i,:)          = S(i,:) + snew';
                S(:,i)          = S(:,i) + snew;
                S(dIndex)       = S(dIndex) - snew';

                yp(i)           = 1;
            end
        end
    end

    dPsi    = CPGRADIENT(X, S, batchSize);

end
