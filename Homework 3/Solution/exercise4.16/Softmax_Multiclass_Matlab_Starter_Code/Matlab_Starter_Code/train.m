function W = train(X_train, y_train, num_iter, lr)
% Train the multiclass softmax classifier
% X_train: N by 785 matrix
% y_train: N by 1 matrix
% num_iter: number of iterations
% lr: learning rate
% W: 785 by 10 matrix

D = size(X_train, 2); % Number of features
C = 10; % Number of classes
W = zeros(D, C);

for i = 1: num_iter
    grad = gradient_descent(W, X_train, y_train);
    W = W - lr * grad;
end

end

function grad = gradient_descent(W, X, y)
% Gradient descent, refer to 4.57 and I will post a note about an efficient
% way to computing gradient on Canvas
%
% W: 785 by 10 matrix
% X: N by 785 matrix
% y: N by 1 matrix
% grad: gradient of W, 785 by 10 matrix
% hint: you may find softmax()(in softmax.m) and sub2ind(, [1:785], y') very useful

%% TODO

end

