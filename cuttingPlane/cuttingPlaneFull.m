function [dPsi, M, SO_time] = cuttingPlaneParallel(k, X, W, Ypos, Yneg, batchSize, SAMPLES, ClassScores)
%
% [dPsi, M, SO_time] = cuttingPlaneParallel(k, X, W, Yp, Yn, batchSize, SAMPLES, ClassScores)
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

    global SO PSI DISTANCE CPGRADIENT;

    [d,n,m] = size(X);
    D       = DISTANCE(W, X);

    M       = 0;
    S       = zeros(n);
    dIndex  = sub2ind([n n], 1:n, 1:n);

    SO_time = 0;

    if isempty(ClassScores)
        TS  = zeros(batchSize, n);
        parfor i = 1:batchSize
            if i <= length(SAMPLES)
                j = SAMPLES(i);

                if isempty(Ypos{j})
                    continue;
                end
                if isempty(Yneg)
                    % Construct a negative set 
                    Ynegative = setdiff((1:n)', [j ; Ypos{j}]);
                else
                    Ynegative = Yneg{j};
                end
                SO_start        = tic();
                    [yi, li]    =   SO(j, D, Ypos{j}, Ynegative, k);
                SO_time         = SO_time + toc(SO_start);

                M               = M + li /batchSize;
                TS(i,:)         = PSI(j, yi', n, Ypos{j}, Ynegative);
            end
        end

        % Reconstruct the S matrix from TS
        S(SAMPLES,:)    = TS;
        S(:,SAMPLES)    = S(:,SAMPLES) + TS';
        S(dIndex)       = S(dIndex) - sum(TS, 1);
    else

        % Do it class-wise for efficiency
        for j = 1:length(ClassScores.classes)
            c       = ClassScores.classes(j);
            points  = find(ClassScores.Y == c);

            Yneg    = find(ClassScores.Yneg{j});
            yp      = ClassScores.Ypos{j};
            
            if length(points) <= 1
                continue;
            end

            TS      = zeros(length(points), n);
            parfor x = 1:length(points)
                i           = points(x);
                yl          = yp;
                yl(i)       = 0;
                Ypos        = find(yl);
                SO_start    = tic();
                    [yi, li]    = SO(i, D, Ypos, Yneg, k);
                SO_time     = SO_time + toc(SO_start);

                M           = M + li /batchSize;
                TS(x,:)     = PSI(i, yi', n, Ypos, Yneg);
            end

            S(points,:) = S(points,:) + TS;
            S(:,points) = S(:,points) + TS';
            S(dIndex)   = S(dIndex) - sum(TS, 1);
        end
    end

    dPsi    = CPGRADIENT(X, S, batchSize);

end
