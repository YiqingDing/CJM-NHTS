clear
clc

n = 250; %Number of transitions for MC
m = 8; %Number of actual MCs 
s = 5; %Number of states, also the # of trials in binomial sampling
z = 16; %Number of probability distributions
d = 10;%Number of chains each MC simulates
rng(7)
p = binornd(10, 0.5,1,z)/10; %Generate an array of p for candidate prob dists
P_trans = randsample(p,s*m,true)'; %Produce m trans matrices (each has s lines in trans)

rng shuffle
% states from 0 to s-1 (s in tot)
states = repmat(linspace(0,s-1,s), s*m,1); %state mat (m mats in tot)
trans_mat = binopdf(states,s-1,repmat(P_trans, 1, s)); %Generate the transitional matrices
% y = binocdf(states,s-1,repmat(P_trans, 1, s)); %Generate the cdf for each state (for different dist dep on rows)

%Simulate MCs using the following loop
mc_tot = [];
for i = 1:m
    trans_ind = trans_mat(s*i-s+1 : s*i,:); %Find individual trans mat for current mc
    mc = dtmc(trans_ind); %Create a MC object
    for j = 1:d
        mc_sim = simulate(mc, n)'; %Simulate the steps
        mc_tot = [mc_tot;mc_sim]; %Save to the array
    end
%     figure;
%     graphplot(mc,'ColorEdges',true);
end
writematrix(mc_tot, 'matlab_data.csv')
writematrix(trans_mat, 'matlab_trans.csv')