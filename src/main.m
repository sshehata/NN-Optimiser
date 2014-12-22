clear;
clc;
%load('../data/xordata.mat');    % 100
% load('../data/3parity.mat');    
 %load('../data/ionosphere.mat'); % 88.32
% load('../data/pima.mat');     % 65.10 
% load('../data/iris.mat');       % 41.33
% load('../data/wisconsin.mat');
% load('../data/hepatitis.mat');
 load('../data/waveform.mat');     % 34.1
% load('../data/mackey.mat');
% load('../data/sunspots.mat');
% load('../data/carcount.mat');
 rng('shuffle')
                                                              
% important data
m = size(X,1);
n = size(X,2);
figure 

input_layer_size = n;
hidden_layer_size = 2 * n;
num_labels = size(y, 2);
if length(unique(y)) > 2,
  original_y = y;
  y_new = zeros(m,length(unique(y)));
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

fprintf('Note: Due to random initial weights,\n')
fprintf('some runs may take a long time to converge.\n\n');

fprintf('Training started with network %i - %i - %i.\n', ...
        input_layer_size, ...
        hidden_layer_size, ...
        num_labels);

[nn_params, hidden_layer_size] = prune(initial_nn_params, input_layer_size, ...
hidden_layer_size, num_labels, X, y);
                         
hidden_layer_size
p = predict(nn_params, input_layer_size, hidden_layer_size, ...
            num_labels, X);

 if exist('original_y'),
  y = original_y;
end
correct = sum(y == p);
fprintf('Accuracy: %.2f%% \n', (correct / m* 100));
confumat = confusionmat(y, double(p))