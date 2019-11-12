close all, clear all,
N=1000; n = 2; K=10;
m = [0;0]; Sigma = [1 0;0 1];classPriors = [0.35,0.65];
thr = [0,cumsum(classPriors)];
u = rand(1,N); l = zeros(1,N); x = zeros(2,N);
figure(1),clf, colorList = 'rb';
for L = 1:2
    indices = find(thr(L)<=u & u<thr(L+1));
    l(1,indices) = L*ones(1,length(indices))-2;
end
rad = zeros(1,N); angle = zeros(1,N);
for i = indices
    l(1,indices) = 1;
    rad(i) = unifrnd(2,3);
    angle(i) = unifrnd(-pi,pi);
    x(1,i) = rad(i)*cos((angle(i)));
    x(2,i) = rad(i)*sin((angle(i)));
end
for i = 1:1000
    if(l(1,i) == -1)
        x(:,i) = mvnrnd(m,Sigma)';
    end
end
for L = 1:2
    indices = find(thr(L)<=u & u<thr(L+1));
    figure(1), plot(x(1,indices),x(2,indices),'.','MarkerFaceColor',colorList(L)); axis equal, hold on,
    figure(1),legend('-','+'),title('data'),
end
% Train a Linear kernel SVM with cross-validation
% to select hyperparameters that minimize probability 
% of error (i.e. maximize accuracy; 0-1 loss scenario)
dummy = ceil(linspace(0,N,K+1));
for k = 1:K, indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)]; end,
CList = 10.^linspace(-2,4,7);
for CCounter = 1:length(CList)
    [CCounter,length(CList)],
    C = CList(CCounter);
    for k = 1:K
        indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
        xValidate = x(:,indValidate); % Using folk k as validation set
        lValidate = l(indValidate);
        if k == 1
            indTrain = [indPartitionLimits(k,2)+1:N];
        elseif k == K
            indTrain = [1:indPartitionLimits(k,1)-1];
        else
            indTrain = [[1:indPartitionLimits(k-1,2)],[indPartitionLimits(k+1,1):N]];
        end
        % using all other folds as training set
        xTrain = x(:,indTrain); lTrain = l(indTrain);
        SVMk = fitcsvm(xTrain',lTrain,'BoxConstraint',C,'KernelFunction','linear');
        dValidate = SVMk.predict(xValidate')'; % Labels of validation data using the trained SVM
        indCORRECT = find(lValidate.*dValidate == 1); 
        Ncorrect(k)=length(indCORRECT);
    end 
    PCorrect(CCounter)= sum(Ncorrect)/N; 
end 
figure(2), subplot(1,2,1),
plot(log10(CList),PCorrect,'.',log10(CList),PCorrect,'-'),
xlabel('log_{10} C'),ylabel('K-fold Validation Accuracy Estimate'),
title('Linear-SVM Cross-Val Accuracy Estimate'), %axis equal,
[dummy,indi] = max(PCorrect(:)); [indBestC, indBestSigma] = ind2sub(size(PCorrect),indi);
pbest = dummy;
disp(pbest);

CBest= CList(indBestC); 
SVMBest = fitcsvm(x',l','BoxConstraint',CBest,'KernelFunction','linear');
d = SVMBest.predict(x')'; % Labels of training data using the trained SVM
indINCORRECT = find(l.*d == -1); % Find training samples that are incorrectly classified by the trained SVM
indCORRECT = find(l.*d == 1); % Find training samples that are correctly classified by the trained SVM
figure(2), subplot(1,2,2), 
plot(x(1,indCORRECT),x(2,indCORRECT),'g.'), hold on,
plot(x(1,indINCORRECT),x(2,indINCORRECT),'r.'), axis equal,
title('Training Data (RED: Incorrectly Classified)'),
pTrainingError = length(indINCORRECT)/N, % Empirical estimate of training error probability

u = rand(1,N); ly = zeros(1,N); y = zeros(2,N);
for L = 1:2
    indices = find(thr(L)<=u & u<thr(L+1));
    ly(1,indices) = L*ones(1,length(indices))-2;
end
rad = zeros(1,N); angle = zeros(1,N);
for i = indices
    ly(1,indices) = 1;
    rad(i) = unifrnd(2,3);
    angle(i) = unifrnd(-pi,pi);
    y(1,i) = rad(i)*cos((angle(i)));
    y(2,i) = rad(i)*sin((angle(i)));
end
for i = 1:1000
    if(ly(1,i) == -1)
        y(:,i) = mvnrnd(m,Sigma)';
    end
end
for L = 1:2
    indices = find(thr(L)<=u & u<thr(L+1));
    
end
% SVMBest = fitcsvm(y',ly','BoxConstraint',CBest,'KernelFunction','linear');
d = SVMBest.predict(y')'; % Labels of training data using the trained SVM
indINCORRECT = find(ly.*d == -1); % Find training samples that are incorrectly classified by the trained SVM
indCORRECT = find(ly.*d == 1); % Find training samples that are correctly classified by the trained SVM
figure(4), subplot(1,2,1), 
plot(y(1,indCORRECT),y(2,indCORRECT),'g.'), hold on,
plot(y(1,indINCORRECT),y(2,indINCORRECT),'r.'), axis equal,
title('Validate Data (RED: Incorrectly Classified)'),
pValidateError = length(indINCORRECT)/N, % Empirical estimate of training error probability


% Train a Gaussian kernel SVM with cross-validation
% to select hyperparameters that minimize probability 
% of error (i.e. maximize accuracy; 0-1 loss scenario)
dummy = ceil(linspace(0,N,K+1));
for k = 1:K, indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)]; end,
CList = 10.^linspace(-1,9,11); sigmaList = 10.^linspace(-2,2,7);
for sigmaCounter = 1:length(sigmaList)
    [sigmaCounter,length(sigmaList)],
    sigma = sigmaList(sigmaCounter);
    for CCounter = 1:length(CList)
        C = CList(CCounter);
        for k = 1:K
            indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
            xValidate = x(:,indValidate); % Using folk k as validation set
            lValidate = l(indValidate);
            if k == 1
                indTrain = [indPartitionLimits(k,2)+1:N];
            elseif k == K
                indTrain = [1:indPartitionLimits(k,1)-1];
            else
                indTrain = [indPartitionLimits(k-1,2)+1:indPartitionLimits(k+1,1)-1];
            end
            % using all other folds as training set
            xTrain = x(:,indTrain); lTrain = l(indTrain);
            SVMk = fitcsvm(xTrain',lTrain,'BoxConstraint',C,'KernelFunction','gaussian','KernelScale',sigma);
            dValidate = SVMk.predict(xValidate')'; % Labels of validation data using the trained SVM
            indCORRECT = find(lValidate.*dValidate == 1); 
            Ncorrect(k)=length(indCORRECT);
        end 
        PCorrect(CCounter,sigmaCounter)= sum(Ncorrect)/N;
    end 
end
disp(PCorrect)
figure(3), subplot(1,2,1),
contour(log10(CList),log10(sigmaList),PCorrect',20); xlabel('log_{10} C'), ylabel('log_{10} sigma'),
title('Gaussian-SVM Cross-Val Accuracy Estimate'), axis equal,
[dummy,indi] = max(PCorrect(:)); [indBestC, indBestSigma] = ind2sub(size(PCorrect),indi);
pbest = dummy;
disp(pbest);
CBest= CList(indBestC); sigmaBest= sigmaList(indBestSigma); 
SVMBest = fitcsvm(x',l','BoxConstraint',1000,'KernelFunction','gaussian','KernelScale',1);
d = SVMBest.predict(x')'; % Labels of training data using the trained SVM
indINCORRECT = find(l.*d == -1); % Find training samples that are incorrectly classified by the trained SVM
indCORRECT = find(l.*d == 1); % Find training samples that are correctly classified by the trained SVM
figure(3), subplot(1,2,2), 
plot(x(1,indCORRECT),x(2,indCORRECT),'g.'), hold on,
plot(x(1,indINCORRECT),x(2,indINCORRECT),'r.'), axis equal,
title('Training Data (RED: Incorrectly Classified)'),
pTrainingError = length(indINCORRECT)/N, % Empirical estimate of training error probability

% SVMBest = fitcsvm(y',ly','BoxConstraint',1000,'KernelFunction','gaussian','KernelScale',1);
d = SVMBest.predict(y')'; % Labels of training data using the trained SVM
indINCORRECT = find(ly.*d == -1); % Find training samples that are incorrectly classified by the trained SVM
indCORRECT = find(ly.*d == 1); % Find training samples that are correctly classified by the trained SVM
figure(4), subplot(1,2,2), 
plot(y(1,indCORRECT),y(2,indCORRECT),'g.'), hold on,
plot(y(1,indINCORRECT),y(2,indINCORRECT),'r.'), axis equal,
title('Validate Data (RED: Incorrectly Classified)'),
pValidateError = length(indINCORRECT)/N, % Empirical estimate of training error probability
