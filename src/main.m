clear;
clc;
load('../data/xordata.mat')
% load('../data/ionosphere.mat'); %1000 iterations
% load('../data/pima.mat'); %1000 iterations
% >>>>>>>>>>>>>>..
% load('../data/iris.mat');
% load('../data/hepatitis.mat');
% load('../data/waveform.mat');

% rng('shuffle')
rng('default')

% important data
m = size(X,1);
n = size(X,2);

input_layer_size = n;
hidden_layer_size = input_layer_size * 2;
num_labels = size(y, 2);

if length(unique(y)) > 2,
  original_y = y;
  y_new = zeros(m,num_labels);
  for i= 1 : m,
    value=y(i,1);
    y_new(i,value)=1;
  end
  y=y_new;
  num_labels = size(y, 2);
end

% Randomizing data
sel = randperm(m);
X = X(sel, :);
y = y(sel, :);

e = 0.2;
initial_theta1 = randomInitializeWeights(input_layer_size, hidden_layer_size, e);
initial_theta2 = randomInitializeWeights(hidden_layer_size, num_labels, e);
initial_omega = tril(rand(hidden_layer_size),-1);

initial_nn_params = [initial_theta1(:); initial_theta2(:); initial_omega(:)];

[nn_params, cost, epochs] = stoch_grad(initial_nn_params, input_layer_size, hidden_layer_size, ...
                                   num_labels, X, y, 10^-3, 1, 0.3);
p = predict(nn_params, input_layer_size, hidden_layer_size, ...
            num_labels, X);
if exist('original_y'),
  y = original_y;
end
correct = sum(y == p);
fprintf('\nTraining took: %i epochs.\n', epochs);
fprintf('Accuracy: %.2f%% \n', (correct / m* 100));
% confumat = confusionmat(y, p)
