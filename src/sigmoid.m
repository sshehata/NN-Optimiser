function [g] = sigmoid(z)
  g = 0.5 * ((1 - exp(-z)) ./ (1 + exp(-z)));
end
