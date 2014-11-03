clear;
clc;
load('../data/3parity.mat');
rng('shuffle')

% important data
m = size(X,1);
n = size(X,2);

input_layer_size = n;
hidden_layer_size = 4;
num_labels = size(y, 2);

% Randomizing data 
sel = randperm(m);
X = X(sel, :);
y = y(sel, :);

e = 0.2;
initial_theta1 = randomInitializeWeights(input_layer_size, hidden_layer_size, e);
initial_theta2 = randomInitializeWeights(hidden_layer_size, num_labels, e);
initial_omega = zeros(1, hidden_layer_size - 1);

initial_nn_params = [initial_theta1(:); initial_theta2(:); initial_omega(:)];

[nn_params, cost, epochs] = stoch_grad(initial_nn_params, input_layer_size, hidden_layer_size, ...
                                   num_labels, X, y, 10^-3, 1);
                               
p = predict(nn_params, input_layer_size, hidden_layer_size, ...
            num_labels, X);
        
correct = sum(p == y);
fprintf('\nTraining took: %i epochs.\n', epochs);
fprintf('Accuracy: %d%% \n', (correct / m * 100));
confumat = confusionmat(logical(y), p)

