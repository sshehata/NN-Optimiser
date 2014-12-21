function [W] = randomInitializeWeights(in, out, epsilon)
%   epsilon is used so that the range of weights at the begining is going
%   to be -epsilon till epsilon
  W = zeros(out, in+1) * 2 * epsilon - 0;
end
