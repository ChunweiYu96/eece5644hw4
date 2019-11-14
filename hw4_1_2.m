

%  Load an image of a bird
 A = double(imread('EECE5644_2019Fall_Homework4Questions_3096_colorPlane.jpg'));
%A = double(imread('EECE5644_2019Fall_Homework4Questions_42049_colorBird.jpg'));
% If imread does not work for you, you can try instead
%   load ('bird_small.mat');

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
N = img_size(1) * img_size(2);
K = 5;
d = 5;
delta = 0.1;
regWeight = 1e-10;

[alpha,mu,Sigma] = EMforGMM(N,X',K,d,delta,regWeight);
p = zeros(2,N);X_recovered = zeros(N,5);
for i = 1:K
    centroids(i,:) = [0.99/K*i,0.5,.99/K*i,0.99/K*i,0.99/K*i];

end
for i = 1:N
    for j = 1:K
        p(j,i) = alpha(j)*evalGaussian(X(i,:)',mu(:,j),Sigma(:,:,j));
    end
    a = max(p(:,i));
    b(i) = find(p(:,i)==a);
    X_recovered(i,:)=centroids(b(i),:)';
    
end
        
X_recovered(:,4) = [];
X_recovered(:,4) = [];

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
function [alpha,mu,Sigma] = EMforGMM(N,x,M,d,delta,regWeight)
% Initialize the GMM to randomly selected samples
alpha = ones(1,M)/M;
shuffledIndices = randperm(N);
mu = x(:,shuffledIndices(1:M)); % pick M random samples as initial mean estimates
[~,assignedCentroidLabels] = min(pdist2(mu',x'),[],1); % assign each sample to the nearest mean
for m = 1:M % use sample covariances of initial assignments as initial covariance estimates
    Sigma(:,:,m) = cov(x(:,find(assignedCentroidLabels==m))') + regWeight*eye(d,d);
end
t = 0; %displayProgress(t,x,alpha,mu,Sigma);

Converged = 0; % Not converged at the beginning
while ~Converged
    for l = 1:M
        temp(l,:) = repmat(alpha(l),1,N).*evalGaussian(x,mu(:,l),Sigma(:,:,l));
    end
    plgivenx = temp./sum(temp,1);
    alphaNew = mean(plgivenx,2);
    w = plgivenx./repmat(sum(plgivenx,2),1,N);
    muNew = x*w';
    for l = 1:M
        v = x-repmat(muNew(:,l),1,N);
        u = repmat(w(l,:),d,1).*v;
        SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
    end
    Dalpha = sum(abs(alphaNew-alpha'));
    Dmu = sum(sum(abs(muNew-mu)));
    DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
    Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
    alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
    t = t+1; 
    if(t==300)
        Converged =1;
    end
   
end
end

function x = randGMM(N,alpha,mu,Sigma)
d = size(mu,1); % dimensionality of samples
cum_alpha = [0,cumsum(alpha)];
u = rand(1,N); x = zeros(d,N); labels = zeros(1,N);
for m = 1:length(alpha)
    ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1)); 
    x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
end
end

%%%
function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z =  randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);
end

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end

