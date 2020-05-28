function softmax_scores = softmax(X, W)
% Return the softmax scores, you can refer to "softmax function" on Wikipedia
% X: N by 785 matrix
% W: 785 by 10 matrix
% softmax_scores: N by 10 matrix
% hint: you may find subtract_max_score()(in this file), exp(), sum(A, 2), and repmat(B, [1, 10]) useful

%% TODO

end

function scores_out = subtract_max_score(scores_in)
% Subtract the max scores of each data point to avoid exponential overflow issue
% scores_in: N by 10 matrix
% score_out: N by 10 matrix
% hint: you may find max(A,[],2) and repmat(B, [1, 10]) useful

%% TODO

end