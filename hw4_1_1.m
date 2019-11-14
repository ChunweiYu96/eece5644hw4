A = double(imread('EECE5644_2019Fall_Homework4Questions_3096_colorPlane.jpg'));

A = A / 255; % Divide by 255 so that all values are in the range 0 - 1

% Size of the image
img_size = size(A);
for i = 1:img_size(1)
    for j = 1:img_size(2)
        A(i,j,4) = i/img_size(1);
        A(i,j,5) = j/img_size(2);
    end
end

X = reshape(A, img_size(1) * img_size(2), 5);


K = 5; 
max_iters =20;

initial_centroids = kMeansInitCentroids(X, K);

% Run K-Means
[centroids, idx] = runkMeans(X, initial_centroids, max_iters);


fprintf('\nApplying K-Means to compress an image.\n\n');


idx = findClosestCentroids(X, centroids);
for i = 1:K
    centroids(i,:) = [0.99/K*i,0.5,0.99/K*i,0.99/K*i,0.99/K*i];

end
X_recovered = centroids(idx,:);
X_recovered(:,4) = [];
X_recovered(:,4) = [];
    
% Reshape the recovered image into proper dimensions
X_recovered = reshape(X_recovered, img_size(1), img_size(2), 3);


% Display the original image 
subplot(1, 2, 1);
A(:,:,4)=[];
A(:,:,4)=[];
imagesc(A); 
title('Original');

% Display compressed image side by side
subplot(1, 2, 2);
imagesc(X_recovered)
title(sprintf('Compressed, with %d colors.', K));

function centroids = kMeansInitCentroids(X, K)
centroids = zeros(K, size(X, 2));
% Initialize the centroids to be random examples
% Randomly reorder the indices of examples
randidx = randperm(size(X, 1));
% Take the first K examples as centroids
centroids = X(randidx(1:K), :);


end
function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z =  randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N)
end
function centroids = computeCentroids(X, idx, K)
% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);

for i=1:K
       centroids(i,:) =  mean( X( find(idx==i) , :) );   % 
end
end
function idx = findClosestCentroids(X, centroids)
idx = zeros(size(X,1), 1);
for i=1:length(idx)
    distanse = pdist2(centroids,X(i,:));   % compute the distance(K,1)   pdist2 is a good function  
       [C,idx(i)]=min(distanse);           % find the minimum
end
end
function [centroids, idx] = runkMeans(X, initial_centroids, ...
                                      max_iters, plot_progress)
if ~exist('plot_progress', 'var') || isempty(plot_progress)
    plot_progress = false;
end

% Plot the data if we are plotting progress
if plot_progress
    figure;
    hold on;
end

% Initialize values
[m n] = size(X);
K = size(initial_centroids, 1);
centroids = initial_centroids;
previous_centroids = centroids;
idx = zeros(m, 1);

% Run K-Means
for i=1:max_iters
    
    % Output progress
    fprintf('K-Means iteration %d/%d...\n', i, max_iters);
    if exist('OCTAVE_VERSION')
        fflush(stdout);
    end
    
    % For each example in X, assign it to the closest centroid
    idx = findClosestCentroids(X, centroids);
    
    % Optionally, plot progress here
    if plot_progress
        plotProgresskMeans(X, centroids, previous_centroids, idx, K, i);
        previous_centroids = centroids;
        fprintf('Press enter to continue.\n');
        pause;
    end
    
    % Given the memberships, compute new centroids
    centroids = computeCentroids(X, idx, K);
    
end

% Hold off if we are plotting progress
if plot_progress
    hold off;
end

end