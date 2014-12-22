clear;
clc;
% load('ionosphere.mat');   % 100
% load('3parity.mat');      % 100
load('../data/hepatitis.mat');
% load('iris.mat');
rng('shuffle')

% important data
m = size(X,1);
n = size(X,2);

input_layer_size = size(X,2);
hidden_layer_size = input_layer_size;
num_labels = size(y,2);
% Randomizing data 
sel = randperm(m);
X = X(sel, :);
y = y(sel, :);

initial_theta1 = randomInitializeWeights(input_layer_size, hidden_layer_size);
initial_theta2 = randomInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params = [initial_theta1(:); initial_theta2(:)];

lambda = 0;

J = nnCostFunction(initial_nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X, y, lambda);

options = optimset('MaxIter', 200);

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
% cost

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
	                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
	                 num_labels, (hidden_layer_size + 1));

p = predict(Theta1, Theta2, X);
false = 100 * sum(p ~= y)/length(y)
% p~=y 
% X