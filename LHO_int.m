function [lho] = LHO_int(u, u_extra, y,x, diffs)
%x(1)=a, x(2)=bu, x(3)=bo, x(4)=sigma2
N=numel(y);

u_full = fill_out( u, diffs, N );

e = y-x(1)*u_extra(1,:)-x(2)*u_extra(2,:)-x(3)*u_full';
%lho = N*0.5*log(2*pi) + N*log(x(4)) + (1 /( 2*x(4)^2 ))*sum( (e-mean(e)).^2 );

lho = sum(e.^2);
end

