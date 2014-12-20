clear;
clc;
% load('../data/xordata.mat');
load('../data/3parity.mat');
% load('../data/ionosphere.mat');
% load('../data/pima.mat');
% load('../data/iris.mat');
% load('../data/wisconsin.mat');
% load('../data/hepatitis.mat');
% load('../data/waveform.mat');
% load('../data/mackey.mat');
% load('../data/sunspots.mat');
% load('../data/carcount.mat');
rng('shuffle')
                                                              
% important data
m = size(X,1);
n = size(X,2);

input_layer_size = n;
hidden_layer_size = n * 2;
num_labels = size(y, 2);

% Randomizing data 
sel = randperm(m);
X = X(sel, :);
y = y(sel, :);

e = 0.2;
initial_theta1 = randomInitializeWeights(input_layer_size, hidden_layer_size, e);
initial_theta2 = randomInitializeWeights(hidden_layer_size, num_labels, e);
initial_omega = tril(rand(hidden_layer_size),-1);

initial_nn_params = [initial_theta1(:); initial_theta2(:); initial_omega(:)];

[nn_params, hidden_layer_size] = prune(initial_nn_params, input_layer_size, ...
hidden_layer_size, num_labels, X, y);
                         
hidden_layer_size
p = predict(nn_params, input_layer_size, hidden_layer_size, ...
            num_labels, X);

correct = sum(p == y);
fprintf('Accuracy: %.2f%% \n', (correct / m * 100));
confumat = confusionmat(logical(y), p)

