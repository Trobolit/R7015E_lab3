diffs = logical(diff(occupancy_training)); % one less than N
N_openings = sum(logical(diff(occupancy_training)));

max_ppl = 4;
x0o = zeros(N,1);     %Initial occupancy guesses.
ohat = x0o;

x0 = [1,0,1,1];             %Initial paramerer guesses.
xhat = x0;

y = CO2_training(2:end)';   % Outputs


for i=1:100
    % Solution not using Least Squares, but Maximum Likelihood.
    u = [CO2_training(1:end-1)'; ventilation_training(1:end-1)'; ohat']; % Inputs
    
    flh = @(x)LH(x,y,u);        %function handle for funciton to minimize
    [xhat,fval] = fmincon(flh,x0,[],[],[],[],[0,-inf,0,0],[1,0,inf,inf]); % Minimize!


    % Use first model but use fmincon on u instead of parameters.
    uo_extra = [CO2_training(1:end-1)'; ventilation_training(1:end-1)'];
    yo = CO2_training(2:end)';   % Outputs
    N = numel(occupancy_training)-1;

    flho_int = @(uo)LHO_int(uo,uo_extra, yo,xhat,diffs);
    
    [xoi,fvaloi] = ga(flho_int,N_openings+1,[],[],[],[],zeros(N_openings+1,1),max_ppl*ones(N_openings+1,1),[],[1:N_openings+1]'); % Minimize!


    ohat = fill_out( xoi, diffs, N );
    i

end

%%
figure();
hold on;
plot(y); % equal to CO2_training(2:end)
plot(xhat(1:3)*u);

%plot(100*logical(diff(u(3,:))));
legend('y','estimated', 'estimated using estimated occupancy');
title('CO2 training');
hold off;