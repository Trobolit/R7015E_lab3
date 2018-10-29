%% Load data
CO2_training = textread('training-CO2.txt','%f');
occupancy_training = textread('training-occupancy.txt','%f');
ventilation_training = textread('training-ventilation.txt','%f');

CO2_test = textread('testing-CO2.txt','%f');
occupancy_test = textread('testing-occupancy.txt','%f');
ventilation_test = textread('testing-ventilation.txt','%f');

%% Ass 1

% Since we assume gaussian noise we will have that the likelihood = least
% squares. Use mldivide.

u = [CO2_training(1:end-1)'; ventilation_training(1:end-1)'; occupancy_training(1:end-1)'];
y = CO2_training(2:end)';
thetahat = y/u;
a = thetahat(1);
bu = thetahat(2);
bo = thetahat(3);

%% Solution not using Least Squares, but Maximum Likelihood.

u = [CO2_training(1:end-1)'; ventilation_training(1:end-1)'; occupancy_training(1:end-1)']; % Inputs
y = CO2_training(2:end)';   % Outputs
%x0 = [a,bu,bo,var(y - thetahat*u)];
x0 = [1,0,1,1];             %Initial paramerer guesses.
flh = @(x)LH(x,y,u);        %function handle for funciton to minimize
[x,fval] = fmincon(flh,x0,[],[],[],[],[0,-inf,0,0],[1,0,inf,inf]); % Minimize!
fval    %Resulting function value
x       %Parameters

%% fig 1, histograms for data error distribution

figure(1);
hold on;
% Plot distrubution of errors in estimation, the mean subtracted
histogram(y-x(1:3)*u - mean(y-x(1:3)*u),'BinWidth',0.5,'Normalization','probability');
%histogram(y-x(1:3)*u,'BinWidth',0.5,'Normalization','probability');

%histogram(normrnd(zeros(1000*numel(y),1),var(y - thetahat*u).^0.5),'BinWidth',0.5,'DisplayStyle','stairs','Normalization','probability');

% Plot generated gaussian distribution with parameters we have
varmodifier = 1; %0.8;
histogram(normrnd(zeros(1000*numel(y),1),varmodifier*x(4)),'BinWidth',0.5,'Normalization','probability');
xlim([-20,20]);
legend('errors in predictions training','simulated normrnd');
hold off;

%% figure 1.1, histogram on errors in estimates on test data.
figure(11);
ytest = CO2_test(2:end)';
utest = [CO2_test(1:end-1)'; ventilation_test(1:end-1)'; occupancy_test(1:end-1)'];
hold on;
histogram(ytest - thetahat*utest,'DisplayStyle','stairs','BinWidth',1);
%histogram(normrnd(zeros(numel(y),1),var(y - thetahat*u).^0.5),'DisplayStyle','stairs');
histogram(normrnd(zeros(numel(ytest),1),4.1283),'BinWidth',1);
xlim([-20,20]);
legend('errors in predictions test data','simulated normrnd');
hold off;

%% Figure 2, time plots of estimates
figure(2);
hold on;
%plot(CO2_training(1:end-1));
%plot(u(1,:))
plot(y); % equal to CO2_training(2:end)
plot(thetahat*u);

%plot(100*logical(diff(u(3,:))));
legend('y','estimated');
hold off;

%% Figure 2.1 time plots of errors in estimates

figure(21);
hold on;
%plot(CO2_training(1:end-1));
plot(u(1,:)-thetahat*u)
plot(y-thetahat*u); % equal to CO2_training(2:end)
legend('shifter','supposedily well estimated');
hold off;
%% USE MODEL TO ESTIMATE CO2 "OCCUPANCY RELATED ACITIVITES" IN "DAMIANOS ROOM".

% (1/bo) * (y-ay(t-1) -bu(t-1)) = o(t-1) %Estimate now does not depend on
% new people since they first have to breathe... So estimate o(t-1).
% Also, dead people do not count. Wierdo.

foest = @(y,yl,ul) max((1/x(3))*(y-x(1)*yl-x(2)*ul),0);

oest = foest(y,u(1,:),u(2,:));

figure(6969);
hold on;

bws = 1;
plotdata = round(conv(bartlett(bws),oest)/sum(bartlett(bws)));
plot(plotdata);
plot(u(3,:));

plot(plotdata-u(3,:));
xlim([400,1500]);
legend('estimated','real', 'diff');
hold off;

%%
figure(69);
hold on;

histogram(plotdata-u(3,:),'BinWidth',0.5);

hold off;

