function output = feedForNN(input, inputToHiddenWeights, hiddenToOutputWeights, hiddenBias, outputBias,m_load,k,edge_result1_k)
    % Define network architecture
    %inputSize = 3;
    %outputSize = 2;
    %hiddenSize = 5;

    %inputSize = size(inputToHiddenWeights, 2);
    %outputSize = size(hiddenToOutputWeights, 1);
    %hiddenSize = size(hiddenBias, 1);
    
    % Define the network weights
    %inputToHiddenWeights = randn(hiddenSize, inputSize);
    %hiddenToOutputWeights = randn(outputSize, hiddenSize);

    % Define the bias terms
    %hiddenBias = randn(hiddenSize, 1);
    %outputBias = randn(outputSize, 1);

    % Input to hidden layer
    hiddenInput = inputToHiddenWeights * input;
    hiddenOutput = leaky_ReLU(hiddenInput + hiddenBias,m_load,k,edge_result1_k);

    % Hidden to output layer
    outputInput = hiddenToOutputWeights * hiddenOutput;
    output_cur = leaky_ReLU(outputInput + outputBias,m_load,k,edge_result1_k);
    output_cur(1,2) = 0;
    output_cur(2,1) = 0;
    output = output_cur; 
end

% function sig = sigmoid(x)
%     % Sigmoid activation function
%     sig = 1 ./ (1 + exp(-x));
% end

% function relu = ReLU(x)
%     % activation function
%     relu = max(0,x);
% end

function lrelu = leaky_ReLU(x,m_load,k,edge_result1_k)
    % activation function
    if m_load>0
        if k==edge_result1_k
           alpha = 0.0005;%0.0005
        else
           alpha = 0.005;%0.005 % Slope for negative inputs
        end
    else
        alpha=0.005;%0.005
    end
    lrelu = max(alpha*x,x);
end