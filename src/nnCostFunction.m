function [nn_params, Jo] = nnCostFunction(nn_params, input_layer_size, ...
                                   hidden_layer_size, num_labels, ...
                                   X, y, e)
                                
	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
	                 hidden_layer_size, (input_layer_size + 1));
  nn_params = nn_params(prod(size(Theta1))+1:end);

	Theta2 = reshape(nn_params(1:(hidden_layer_size + 1) * num_labels), ...
	                           num_labels, (hidden_layer_size + 1));
  nn_params = nn_params(prod(size(Theta2))+1:end);

  Omega = nn_params;

  m = size(X, 1);
  Jo = 0;

  while Jo > e
    Jo = 0;
    for sample=1:m
      % Feedforward the neural network and calculate the cost in J 
      a1 = X(sample,:);
      a1 = [ones(size(a1,1),1), a1];
      z2 = a1 * Theta1';
      a2 = sigmoid(z2);
      a2 = [ones(size(a2,1),1), a2];
      z3 = a2 * Theta2';
      a3 = sigmoid(z3);

      e = (y(sample) - a3)^2 
      Jo = Jo + e;

      % Vectorized BackPropagation algoritm 
      delta_3 = (y(sample) - a3) * a3 * (1 - a3);
      delta_2 = (delta_3 * Theta2) * ( a2 * ( 1 - a2 ));
      Theta2_delta = delta_3 * a2;
      Theta1_delta = delta_2 * a1
      Omega_delta = delta_2(2:end) *  a2(1:end-1); 

      Theta2 = Theta2 - Theta2_delta;
      Theta1 = Theta1 - Theta1_delta;
      Omega = Omega - Omega_delta;

      % updating ThetaH and OmegaH using Jh
      Theta1_delta = a2 .* a2 .* (1 - a2) .* a1;
      Omega_delta = a2(2:end) .* a2(2:end) .* (1 - a2(2:end)) .* a2(1:end-1);
      Theta1 = Theta1 - Theta1_delta;
      Omega = Omega - Omega_delta;
    end
  end

  nn_params = [Theta1(:); Theta2(:); Omega(:)];
end
