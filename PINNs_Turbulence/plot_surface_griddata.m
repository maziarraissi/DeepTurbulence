% @author: Maziar Raissi

function plot_surface_griddata(X_star, u_star, xlab, ylab, tit)

x_l = min(X_star(:,1));
x_r = max(X_star(:,1));

y_l = min(X_star(:,2));
y_r = max(X_star(:,2));


nn = 100;
x = linspace(x_l, x_r, nn)';
y = linspace(y_l, y_r, nn)';
[Xplot, Yplot] = meshgrid(x,y);

Zplot = griddata(X_star(:,1),X_star(:,2),u_star,Xplot,Yplot,'cubic');

surf(Xplot, Yplot, Zplot);
xlabel(xlab);
ylabel(ylab);
title(tit);

axis tight
axis square
colormap jet
shading interp
% colorbar
set(gca,'FontSize',15);
set(gcf, 'Color', 'w');

%text(0.25*(x_l+x_r),0.5*(y_l+y_r),1.5*max(zlim),tit,'FontSize',15);