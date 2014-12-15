clear;
clc;
% load('../data/carcount.mat');
% load('../data/mackey.mat');
load('../data/sunspots.mat');

% X = X(1:2,:);
X = featureNormalize(X);
y = X;

% important data
m = size(X,1);
n = size(X,2);

input_layer_size = n;
hidden_layer_size = input_layer_size * 2;
num_labels = 1;

% Randomizing data
rng('shuffle')
sel = randperm(m);
X = X(sel, :);
y = y(sel, :);

e = 0.2;
initial_theta1 = randomInitializeWeights(input_layer_size, hidden_layer_size, e);
initial_theta2 = randomInitializeWeights(hidden_layer_size, num_labels, e);
initial_omega = rand(1, hidden_layer_size - 1);

initial_nn_params = [initial_theta1(:); initial_theta2(:); initial_omega(:)];

[nn_params, cost, epochs] = stoch_grad_approximation(initial_nn_params, input_layer_size, hidden_layer_size, ...
                                   num_labels, X, y, 10^-5, 1, 0.3);
p = predict_approximation(nn_params, input_layer_size, hidden_layer_size, ...
            num_labels, X);
correct = sum(abs(y - p) < 0.001);
fprintf('\nTraining took: %i epochs.\n', epochs);
fprintf('Accuracy: %.2f%% \n', (correct / m* 100));