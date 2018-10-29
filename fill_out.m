function [ u_full ] = fill_out( u, diffs, N )
%FILL_OUT Summary of this function goes here
%   Detailed explanation goes here
u_full = nan(N,1);
k=1;
for i=1:N
    u_full(i) = u(k); 
    if i<N % if we are at last step we wont copy anything else
        if diffs(i) % if the door opened until next timestamp take next signal.
            k = k+1;
        end
    end
end

end

