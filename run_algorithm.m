
% Load Data
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


concat_trials = Nm{1};
for trial = 2:100
    concat_trials = horzcat(concat_trials, Nm{trial}); % get 10 (N) by T trial and concat horz
end


% Aij_init
trans = zeros(9, 9);
trans( 1:5, 1:7 ) = 1/7;
trans( 6, 6 ) = 0.9; trans( 7, 7 ) = 0.9;
trans( 6, 8 ) = 0.1; trans( 7, 9 ) = 0.1;
trans( 8, 8 ) = 1;   trans( 9, 9 ) = 1;


[bestPath_out,maxPathLogProb_out,PI_out,A_out,B_out,gamma] = poissHMM(concat_trials, 9, 0.01, 1, trans, lambda');


% Get test_trial
test_trial_i = randsample(train190_idxs, 1);
Nm = getNeuralObservations(test_trial_i, R, 'units', maxVarNeurs_idxs);
N_test = Nm{1};

[PI_test,A_test,lambda_test,alpha_test,beta_test,gamma_test,epsilons_test] = myBaumWelch(N_test,9,0.01,1,A_out,B_out);


% plot stuff
T = size(N_test, 2);
% L = size(A_test, 2);

% a posteriori likelihood
apl = alpha_test .* beta_test;

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
