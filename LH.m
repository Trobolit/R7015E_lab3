function [lh] = LH(x,y,u)

%x(1)=a, x(2)=bu, x(3)=bo, x(4)=sigma2
N=numel(y);
e = y-x(1)*u(1,:)-x(2)*u(2,:)-x(3)*u(3,:);
lh = N*0.5*log(2*pi) + N*log(x(4)) + (1 /( 2*x(4)^2 ))*sum( (e-mean(e)).^2 );

end

