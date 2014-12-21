function [J, grad] = nnCostFunction(nn_params, input_layer_size, ...
                                   hidden_layer_size, num_labels, ...
                                   X, y, lambda)
                                
	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
	                 hidden_layer_size, (input_layer_size + 1));

	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
	                 num_labels, (hidden_layer_size + 1));

	m = size(X, 1);
y_new = zeros(m,num_labels);
for i= 1 : m,
	value=y(i,1);
	y_new(i,value)=1;
end
y=y_new;	

% Feedforward the neural network and calculate the cost in J 
	a1 = X;
	a1 = [ones(size(a1,1),1), a1];
	z2 = a1 * Theta1';
	a2 = sigmoid(z2);
	a2 = [ones(size(a2,1),1), a2];
	z3 = a2 * Theta2';
	a3 = sigmoid(z3);
  
  temp_theta1 = Theta1;
  temp_theta2 = Theta2;
  temp_theta1(:,1) = [];
  temp_theta2(:,1) = [];
% cost of the neural network
  J = (-1/m)* sum( sum(y .* log(a3))+ sum((1.-y) .* log(1-a3))) + (lambda/(2*m))* (sum(sum(tempTheta1.^2)) + sum(sum(tempTheta2.^2)));
  
  % Vectorized BackPropagation algoritm to get both Theta1_grad and Theta2_grad
	delta_3 = a3 - y;
	delta_2 = (delta_3 * Theta2) .* ( a2' .* ( 1 - a2' ))'; %I am using it directly instead of using sigmoid gradient

	Theta2_delta = delta_3' * a2; 
	Theta1_delta = delta_2' * a1;

	Theta1_delta(1,:) = [];
	Theta1_grad = (1/m) * Theta1_delta + (lambda/m) * Theta1;
	Theta2_grad = (1/m) * Theta2_delta + (lambda/m) * Theta2;
	Theta1_grad(:,1) = (1/m) * Theta1_delta(:,1);
	Theta2_grad(:,1) = (1/m) * Theta2_delta(:,1);

	grad = [Theta1_grad(:) ; Theta2_grad(:)];

end