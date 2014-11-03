clear;
clc;
load('../data/xordata.mat');
rng('shuffle')

% important data
m = size(X,1);
n = size(X,2);

input_layer_size = n;
hidden_layer_size = 2;
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

[Theta1, Theta2, cost] = stoch_grad(initial_nn_params, input_layer_size, hidden_layer_size, ...
                                   num_labels, X, y, 10^-4);
                               
p = predict(Theta1, Theta2, X);
sum(p ~= y)
% p~=y 
% X
