var = [ 1.638 0.562; 0.562 1.919];
x = randn(1000,2)*sqrt(var);

[N,D] = size(x);
x_c = bsxfun(@minus, X, mean(X));

rbf = @(x,z) exp(-1*((x-z)'*(x-z)));

K = zeros(N,N);
for i =1:length(N)
    for j=1:length(N)
       K(i,j) = rbf(x(i, :), x(j,:));      
    end
end    

