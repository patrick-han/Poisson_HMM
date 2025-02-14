function [bestPath,maxPathLogProb,PI,A,B,gamma] = poissHMM(spikes,nStates,dt,maxIter, Aij_init, lambda_init)
% Inputs:
% spikes  - the N x T matrix of spikes (N neurons and T timesteps) where
%           spikes are indicated by 1's and all other elements are 0
% nStates - The number of hidden states expected
% dt      - the size (in seconds) of each timestep
% maxIter - maximum number of iterations through the data

% Estimate state transition matrix and firing rates
[PI,A,B,alpha,beta,gamma,epsilons] = myBaumWelch(spikes,nStates,dt,maxIter, Aij_init, lambda_init);
% Get most likely state sequence
[bestPath,maxPathLogProb,T1,T2] = myViterbi(spikes,PI,A,B,dt);
end

