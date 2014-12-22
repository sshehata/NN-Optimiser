clear;
clc;
load('../data/ionosphere.mat');
% load('../data/hepatitis.mat');
% load('../data/pima.mat');
% load('../data/wisconsin.mat');
% load('../data/hepatitis.mat');
% load('../data/waveform.mat');
% load('../data/iris.mat');
rng('shuffle')

% important data
m = size(X,1);
n = size(X,2);

input_layer_size = size(X,2);
hidden_layer_size = n*2;
num_labels = 1;
% Randomizing data
sel = randperm(m);
X = X(sel, :);
y = y(sel, :);

for i = 1: hidden_layer_size,
  dlmwrite('ionosphere.txt','hidden nodes','-append','delimiter','','newline','pc');
  dlmwrite('ionosphere.txt',i,'-append','delimiter','\t','newline','pc');
  initial_theta1 = randomInitializeWeights(input_layer_size, i);
  initial_theta2 = randomInitializeWeights(i, num_labels);

  initial_nn_params = [initial_theta1(:); initial_theta2(:)];

  lambda = 0;

  J = nnCostFunction(initial_nn_params, input_layer_size, i, ...
                     num_labels, X, y, lambda);

  options = optimset('MaxIter', 500);

  costFunction = @(p) nnCostFunction(p, ...
                                     input_layer_size, ...
                                     i, ...
                                     num_labels, X, y, lambda);

  [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
  % cost
  Theta1 = reshape(nn_params(1:i * (input_layer_size + 1)), ...
                     i, (input_layer_size + 1));

  Theta2 = reshape(nn_params((1 + (i * (input_layer_size + 1))):end), ...
                     num_labels, (i + 1));

  p = predict(Theta1, Theta2, X);
  accuracy = sum(p == y) * 100 / length(y);
  wrong = sum(p ~= y);
  dlmwrite('ionosphere.txt','accuracy','-append','delimiter','','newline','pc');
  dlmwrite('ionosphere.txt',accuracy,'-append','delimiter','\t','newline','pc');
  dlmwrite('ionosphere.txt','number of wrong predictions','-append','delimiter','','newline','pc');
  dlmwrite('ionosphere.txt',wrong,'-append','delimiter','\t','newline','pc');
end