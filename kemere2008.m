
R_raw = load('data/R,2004-12-17,1,hs-spline.mat');
reachData = load('data/ReachData.mat');
R_raw = reachData.r;
R = R_raw;
R = R( [R.isSuccessful] ); % filter successful trials 
R = R( logical( [R.hasSpikes] )); % filter for trials with spikes


M = length(R); % number of (successful) trials = 1127
K = length(R(1).unit);  % number of units = 190


tths = [R.timeTouchHeld];
tvs = [R.timeVelMax];
tgcs = [R.timeGoCue];
ttas = [R.timeTargetAcquire];


% diff(X,N,DIM) first-order difference along DIM
% plan_len = diff(vertcat(tths-200, tgcs), 1, 1) / 1000.0; 
% move_len = diff(vertcat(tgcs, ttas), 1, 1) / 1000.0; 
base_len = 350 / 1000.0;
plan_len = 600 / 1000.0; 
move_len = 600 / 1000.0; 

% spike rates with (trial_i, base/plan/move epoch, unit_i)
rates = zeros(M, 3, K);
for trial_i = 1:M
    for neur_j = 1:K
        
        spikeTs = R(trial_i).unit(neur_j).spikeTimes;
        
        rates(trial_i, 1, neur_j) = length( spikeTs( spikeTs > tths(trial_i)-200 & spikeTs < tths(trial_i)+150 ) ) / base_len;
        rates(trial_i, 2, neur_j) = length( spikeTs( spikeTs > tths(trial_i)+150 & spikeTs < tths(trial_i)+750 ) ) / plan_len;
        rates(trial_i, 3, neur_j) = length( spikeTs( spikeTs > tvs(trial_i)-250 & spikeTs < tvs(trial_i)+350 ) ) / move_len;
        
    end
end




avgRates = squeeze(mean(rates, 1)); % avg rates across all trials

% find the k neurons with maximum variability between epochs
diffs = sum( abs(diff(avgRates, 1)) , 1); 
[dummy_, maxVarNeurs_idxs] = maxk(diffs, 10);


% get left trials and right trials
params = [R.TrialParams]; 
targs = [params.targetAngularDirection];
targs30_idxs = find( (targs == 30) ); 
targs190_idxs = find( (targs == 190) ); 


% plot two examples of spike train data, one moving L and one R
% figure; 
% subplot(2, 1, 1); hold on;
% plotspikes(randsample(targs30_idxs, 1), maxVarNeurs_idxs, R);
% subplot(2, 1, 2); hold on;
% plotspikes(randsample(targs190_idxs, 1), maxVarNeurs_idxs, R);
% plotspikes(1, maxVarNeurs_idxs, R);


% randomly sample 50 trials (indexes) which correspond to trials 
% with appropriate target directions
% train30_idxs = randsample(targs30_idxs, 50);  
% train190_idxs = randsample(targs190_idxs, 50); 
train30_idxs = targs30_idxs(1:50);  
train190_idxs = targs190_idxs(1:50); 



% transition matrix A for simple HMM, 5 baseline states
% and a plan and move state for L and R
trans = zeros(9, 9);
trans( 1:5, 1:7 ) = 1/7;
trans( 6, 6 ) = 0.9; trans( 7, 7 ) = 0.9;
trans( 6, 8 ) = 0.1; trans( 7, 9 ) = 0.1;
trans( 8, 8 ) = 1;   trans( 9, 9 ) = 1;


% initial state probabilities, equal for all baseline states
pi = [0.2 0.2 0.2 0.2 0.2 0 0 0 0];
A = trans;

% [1 x M trials] cell, each [K units x T time] 
Nm = getNeuralObservations(train30_idxs, R, 'units', maxVarNeurs_idxs); 
Nm = [Nm getNeuralObservations(train190_idxs, R, 'units', maxVarNeurs_idxs)];
    
M = length(Nm); 

% find average rates for poisson observations
lam_base = squeeze( mean( rates( [train30_idxs, train190_idxs], 1,  maxVarNeurs_idxs), 1))';
lam_plan30 = squeeze( mean( rates( train30_idxs, 2,  maxVarNeurs_idxs), 1))';
lam_plan190 = squeeze( mean( rates( train190_idxs, 2,  maxVarNeurs_idxs), 1))';
lam_move30 = squeeze( mean( rates( train30_idxs, 3,  maxVarNeurs_idxs), 1))';
lam_move190 = squeeze( mean( rates( train190_idxs, 3,  maxVarNeurs_idxs), 1))';

% should be L x Nn
lambda = [lam_base; lam_base; lam_base; lam_base; lam_base];
lambda = [lambda; lam_plan30; lam_plan190 ]; 
lambda = [lambda; lam_move30; lam_move190 ]; 
% enforce a minimum firing rate of 1Hz
lambda(lambda < 1) = 1;



%%

alpha = cell(1, M); 
beta = cell(1, M); 
b = cell(1, M);
log_Pm = zeros(1, M);
for i = 1:M
    [alpha{i}, beta{i}, b{i}, log_Pm(i)] = alphaBeta(pi, A, lambda, Nm{i}); 
end
[pi_bar, Aij_bar, lambda_bar] = maximization(alpha, beta, b, A, Nm, log_Pm);
lambda_bar(lambda_bar < 1) = 1;



%%

% Test time - find state likelihoods given a sequence of observations
test_trial_i = randsample(train190_idxs, 1);
% test_trial_i = train190_idxs(end);

Nm = getNeuralObservations(test_trial_i, R, 'units', maxVarNeurs_idxs);
N_test = Nm{1};
% [alphaTest, betaTest, b, Pm] = alphaBeta(pi, A, lambda, N_test); 
[alphaTest, betaTest, b, Pm] = alphaBeta(pi_bar, Aij_bar, lambda_bar, N_test); 

T = size(N_test, 2);
L = size(A, 2);

% a posteriori likelihood
apl = alphaTest .* betaTest;

% apl = zeros(L, T);
% apl(:, 1) = pi' .* b(:, 1);
% for t = 2:T
%     for j = 1:L 
%         for i = 1:L
%             apl(j, t) = apl(j, t) + alphaTest(i, t-1) * A(i, j) * b(j, t);
%         end        
%     end
% end

P_base = sum( apl(  1:5, :), 1 ) ./ sum(apl, 1); 
P_plan = sum( apl( [6 7], :), 1 ) ./ sum(apl, 1); 
P_move = sum( apl( [8 9], :), 1 ) ./ sum(apl, 1); 


figure;
subplot(2, 1, 1); hold on; 
spikeTimes = R(test_trial_i).unit(maxVarNeurs_idxs);
plot( [R(test_trial_i).timeTouchHeld R(test_trial_i).timeTouchHeld], [11, 0], 'm', 'LineWidth', 2 );
plot( [R(test_trial_i).timeGoCue R(test_trial_i).timeGoCue], [11, 0], 'c', 'LineWidth', 2 );
spikeTimesCells = squeeze( struct2cell(spikeTimes) );
spikeTimesCellsT  = cellfun(@transpose, spikeTimesCells, 'UniformOutput', false);
plotSpikeRaster(spikeTimesCellsT,'PlotType','vertline');
title(sprintf('Spiketimes for max variability neurons: trial %i, angle %i', test_trial_i, R(test_trial_i).TrialParams.targetAngularDirection ));
xlabel('time (ms)');
ylabel('neuron');
legend('time touch held', 'go cue');

subplot(2, 1, 2); hold on;
plot( ( 1:length(apl) )*10,  P_base); 
plot( ( 1:length(apl) )*10,  P_plan ); 
plot( ( 1:length(apl) )*10,  P_move ); 
set(findall(gca, 'Type', 'Line'),'LineWidth',2);
xlabel('time (ms)');
ylabel('aposterior likelihood');
legend('base', 'plan', 'move');


%% 




%%


function [Nm] = getNeuralObservations(trial_indices, R, varargin)
    % output is a (1 x M) cell with each cell an (K x T) array of
    % observations
    %generating neuron counts N for each trial for each neuron in 10ms bins
    %trial 1:50 is all right direcs, trial 51:100 is all the left direcs
    
    if (nargin > 2)
        units = varargin{2};
    else
        units = 1:(length(R(1).unit));
    end
    
    N = [];
    %first iterting through right direcs
    for i=1:length(trial_indices)
        tTarget = R(trial_indices(i)).timeTouchHeld;
        tVelmax = R(trial_indices(i)).timeVelMax;
        lenOfCntCell = ceil( (tVelmax + 350 - tTarget + 200) / 10); 

        offset = tTarget - 200;
        neuronSpikeCnts = cell(1,length(R(i).unit));

        %going through each neural unit
        for i2=1:length(R(i).unit)
            spikeTimes = R(i).unit(i2).spikeTimes;
            spikeTimes = spikeTimes(spikeTimes <= (tVelmax + 350 - tTarget + 200));
            spikeTimes = spikeTimes - offset;
            spikeTimes = spikeTimes(spikeTimes >= 0);


            spikeCnts = zeros(1,lenOfCntCell);

            %calculating num spikes in each 10ms window
            for i3=1:length(spikeTimes)
                windowLoc = floor(spikeTimes(i3)/10) + 1;
                spikeCnts(windowLoc) = spikeCnts(windowLoc) + 1;

            end

            neuronSpikeCnts{i2} = spikeCnts;

        end
        N = [N; neuronSpikeCnts];
    end
    
    N = N(:, units, :);
    
    Nm = {};
    for trial_i = 1:size(N, 1)
        Nm{trial_i} = vertcat( N{trial_i, :} ) ; 
    end

end




function [S] = getSeqStates(indices, R)

    %generating sequence of states S for each right, left trials chosen for
    %training
    %1 = Baseline, 6 = Plan Right, 7 = Plan Left, Move Right = 9,
    %Move left = 9
    %trial 1:50 is all right direcs, trial 51:100 is all the left direcs
    S = [];

    %first iterting through right direcs
    for i=1:length(indices)

        tTarget = R(indices(i)).timeTouchHeld;
        tVelmax = R(indices(i)).timeVelMax;
        lenOfsm = ceil( (tVelmax + 350 - tTarget + 200) / 10); 
        sm = zeros(1, lenOfsm);

        %assigning baseline states
        sm(1 : 34) = 1;

        planMovSplit = floor( ((tVelmax - 250) - (tTarget +750)) / 20 ); 
        %assigning plan states
        sm( 35 : 35 + planMovSplit - 1) = 6;

        %assigning perimov states
        sm( 35 + planMovSplit :lenOfsm) = 8; 

        smCell = mat2cell(sm, [1]);

        S = [S;smCell];
    end
    
end





function [alpha, beta, b, log_Pm] = alphaBeta(pi, A, lambda, N)
    % M = number of trials, Tm = number of states in the trial sequence
    % K = number of neurons/units
    % L = number of states
    %
    % pi: 1 x L initial state probabilities
    % A: L x L transition matrix
    % lambda: L x Nn
    % N: (Nn x T) the number of spikes at a given time
    T = size(N, 2);
    K = size(N, 1);
    L = size(A, 1);
    
    
    % b: L x Tm state-dependent probabilities of observing the observed 
    % spikes for every state at a given time
    b = ones(L, T);
    for l = 1:L
        for t = 1:T
            for neur = 1:K
                lamb = lambda(l,neur) * 0.01;
%                 b(l, t) = b(l, t) * exp(-lamb) * (lamb ^ N(neur, t)) / factorial(N(neur, t)) ;
                b(l, t) = b(l, t) * exp(-lamb) * (lamb ^ N(neur, t)) ;
            end
        end
    end
    
%     for neur = 1:Nn
%         lambda_i = lambda(:, neur);
%         lamRep = repelem( lambda_i, 1, T ); % L x T
%         N_rep = repelem( N(neur, :), L, 1 ); % L x T
%         b = b .* exp(-lamRep*0.01) .* lamRep .^ N_rep;
%     end
    
    % alpha 
    al = zeros(L, T); 
    c_t = zeros(1, T);
    
    al(:, 1) = pi' .* b(:, 1);
    c_t(1) = 1 / sum( al(:, 1) );
    al(:, 1) = al(:, 1) * c_t(1);
    
    for t = 2:T
        for j = 1:L 
            for i = 1:L
                al(j, t) = al(j, t) + ( A(i, j) * al(i, t-1) * b(j, t) );
            end 
        end
        
        c_t(t) = 1 / sum( al(:, t) );
        al(:, t) = al(:, t) * c_t(t);
    end
    
    
    % beta
    be = zeros(L, T);
    be(:, T) = ones(L, 1);
    be(:, T) = be(:, T) * c_t(T);
    
    for t = T-1:-1:1
        for i = 1:L 
            for j = 1:L
                be(i, t) = be(i, t) + ( A(i,j) * b(j,t+1) * be(j,t+1) );
            end
        end
        be(:, t) = be(:, t) * c_t(t);
    end
    
    alpha = al; 
    beta = be;
    log_Pm = - sum( log(c_t) );
    
end



function [pi_bar, Aij_bar, lambda] = maximization(alph, bet, b, Aij, N, log_Pm)

    % alph, bet, b
    % alph: M by [L by T] (cell)
    % bet: M by [L by T] (cell)
    % b: M by [L by T] (cell)
    % P_M: M by 1 probability of a trial
    % Aij: L by L
    % N: [1 x M cell] by N by T
    M = length(alph);
    L = size(alph{1}(:,1),1);
    
    
%     P_M = ones(M, 1)/M;  
    pi_bar = zeros(1,L);
    for l = 1:L
        for m = 1:M
            alph_m = alph{m}; bet_m = bet{m};
            log_pi_m = -log_Pm(m) + log( alph_m(l,1) * bet_m(l,1) );
            pi_bar(l) = pi_bar(l) + exp( log_pi_m );
        end
        
    end
        

    % Calculate A bar ij
    Aij_bar = zeros(L,L);

    % Calculate Lambda

    % 10 neurons
    lambda = zeros(L,10);

    for j = 1:L
        for i = 1:L
            A_top = 0;
            A_bot = 0;
            for m = 1:M
%                 P_m = log_Pm(m);
                T_m = size(alph{m},2); % size of T
                A_top_m = 0;
                for t = 1:(T_m - 1)
                    A_top_m = A_top_m + alph{m}(i,t) * Aij(i,j) * b{m}(j,t+1) * bet{m}(j,t+1);  
                end
                A_top = A_top + exp( -log_Pm(m) + log(A_top_m) );
                
                A_bot_m = 0;
                for t = 1:(T_m )
                    A_bot_m = A_bot_m + alph{m}(j,t) * bet{m}(j,t) ;
                end
                A_bot = A_bot + exp( -log_Pm(m) + log(A_bot_m) );
            end
            Aij_bar(i,j) = A_top / A_bot;
        end


        % lambda
        for k = 1:10
            lam_top = 0;
            lam_bot = 0;
            for m = 1:M
%                 P_m = log_Pm(m);
                T_m = size(alph{m},2); % size of T
                
                lam_top_m = 0;
                for t = 1:(T_m - 1)
                    lam_top_m = lam_top_m + alph{m}(i,t) * N{m}(k, t+1) * bet{m}(j,t);
                end
                lam_top = lam_top + exp( -log_Pm(m) + log( lam_top_m ) );
                
                lam_bot_m = 0;
                for t = 1:(T_m)
                    lam_bot_m = lam_bot_m + alph{m}(j,t) * bet{m}(j,t);
                end
                lam_bot = lam_bot + exp( -log_Pm(m) + log( lam_bot_m ) );
            end
            lambda(j,k) = lam_top / lam_bot;
        end

    end

end






function dummy_ = plotspikes(trial_i, maxVarNeurs_i, R)
     
    spikeTimes = R(trial_i).unit(maxVarNeurs_i);
    plot( [R(trial_i).timeTouchHeld R(trial_i).timeTouchHeld], [11, 0], 'm', 'LineWidth', 2 );
    plot( [R(trial_i).timeGoCue R(trial_i).timeGoCue], [11, 0], 'c', 'LineWidth', 2 );
    spikeTimesCells = squeeze( struct2cell(spikeTimes) );
    spikeTimesCellsT  = cellfun(@transpose, spikeTimesCells, 'UniformOutput', false);
    plotSpikeRaster(spikeTimesCellsT,'PlotType','vertline');
    title(sprintf('Spiketimes for max variability neurons: trial %i, angle %i', trial_i, R(trial_i).TrialParams.targetAngularDirection ));
    legend('timeTimeHeld', 'goCue');
    xlabel('time (ms)');
    ylabel('neuron');
    
end




function s = logsumexp(a, dim)
    % Returns log(sum(exp(a),dim)) while avoiding numerical underflow.
    % Default is dim = 1 (columns).
    % logsumexp(a, 2) will sum across rows instead of columns.
    % Unlike matlab's "sum", it will not switch the summing direction
    % if you provide a row vector.

    % Written by Tom Minka
    % (c) Microsoft Corporation. All rights reserved.
    if nargin < 2
      dim = 1;
    end

    % subtract the largest in each column
    [y, i] = max(a,[],dim);
    dims = ones(1,ndims(a));
    dims(dim) = size(a,dim);
    a = a - repmat(y, dims);
    s = y + log(sum(exp(a),dim));
    i = find(~isfinite(y));
    if ~isempty(i)
      s(i) = y(i);
    end
end



























