clear;
clc;
load('../data/xordata.mat');
% rng('shuffle')

% important data
m = size(X,1);
n = size(X,2);

input_layer_size = n;
hidden_layer_size = 10;
num_labels = size(y, 2);

% Randomizing data 
sel = randperm(m);
X = X(sel, :);
y = y(sel, :);

e = 0.2;
initial_theta1 = randomInitializeWeights(input_layer_size, hidden_layer_size, e);
initial_theta2 = randomInitializeWeights(hidden_layer_size, num_labels, e);
initial_omega = randomInitializeWeights(0, hidden_layer_size - 1, e);

initial_nn_params = [initial_theta1(:); initial_theta2(:); initial_omega(:)];

J = nnCostFunction(initial_nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y);

options = optimset('MaxIter', 200);

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, 0);

[nn_params, cost] = nnCostFunction(initial_nn_params, input_layer_size, hidden_layer_size, ...
                                   num_labels, X, y, 0.001);
cost
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
	                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
	                 num_labels, (hidden_layer_size + 1));

p = predict(Theta1, Theta2, X);
sum(p ~= y)
% p~=y 
% X
