function W = train(X_train, y_train, num_iter, lr)
% Train the One-versus-All classifier
% X_train: N by 785 matrix
% y_train: N by 1 matrix
% num_iter: number of iterations
% lr: learning rate
% W: 785 by 10 matrix

D = size(X_train, 2); % Number of features
C = 10; % Number of classes
W =  zeros(D, C);

y_converted = convert_to_binary_class(y_train, C);
for i = 1: num_iter
    grad = gradient_descent(W, X_train, y_converted);
    W = W - lr * grad;
end


end

function y_out = convert_to_binary_class(y_in, C)
% Convert y from a multiclass problem to a binary class one
% y_in: N by 1 matrix
% C: Number of class
% y_out: N by 10 matrix, each column consists of +1 or -1
% hint: you can use for loop for this part. And you may need logical indexing
%% TODO

end

function grad = gradient_descent(W, X, y)
% Gradient descent for c-th classifier
% way to computing gradient on Canvas
%
% W: 785 by 10 matrix
% X: N by 785 matrix
% y: N by 10 matrix
% grad: gradient of W, 785 by 10 matrix
% hint: you may find sigmoid() below useful
%% TODO


end



function y = sigmoid(z)
% Sigmoid function
y = zeros(size(z));
mask = (z >= 0.0);
y(mask) = 1./(1 + exp(-z(mask)));
y(~mask) = exp(z(~mask))./(1 + exp(z(~mask)));
end
