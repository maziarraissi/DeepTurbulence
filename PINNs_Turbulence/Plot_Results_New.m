clear
close all

addpath ~/export_fig
set(0,'defaulttextinterpreter','latex')

load turbulence_1D_dissipation_results_20_09_2018.mat

xlab = '$t$';
ylab = '$\psi$';


fig = figure();
set(fig,'units','normalized','outerposition',[0 0 1 0.45])

clf
subplot(2,3,1)
plot_surface_griddata([t,x], u, xlab, ylab, 'Exact $P(t,\psi)$')
lim = zlim;
view(37.5, 30)

subplot(2,3,2)
plot_surface_griddata([t,x], double(u_pred), xlab, ylab, 'Learned $P(t,\psi)$')
zlim(lim)
view(37.5, 30)

subplot(2,3,3)
plot_surface_griddata([t,x], u - double(u_pred), xlab, ylab, 'Exact minus Learned')
view(37.5, 30)

subplot(2,3,4)
plot_surface_griddata([t,x], e, xlab, ylab, 'Exact $\mathcal{E}(t,\psi)$')
lim = zlim;
view(-37.5, 30)

subplot(2,3,5)
plot_surface_griddata([t,x], double(e_pred), xlab, ylab, 'Learned $\mathcal{E}(t,\psi)$')
zlim(lim)
view(-37.5, 30)

subplot(2,3,6)
plot_surface_griddata([t,x], e - double(e_pred), xlab, ylab, 'Exact minus Learned')
view(-37.5, 30)

export_fig ./turbulence_1D_dissipation.png -r300

error_u = norm(double(u_pred) - u)/norm(u);
error_e = norm(double(e_pred) - e)/norm(e);

fprintf('Error P: %e, Error E: %e\n',error_u, error_e)

%%%%%% Error Plot

T = reshape(t,[100,200]);
X = reshape(x,[100,200]);
U = reshape(u,[100,200]);
U_pred = reshape(double(u_pred),[100,200]);
E = reshape(e,[100,200]);
E_pred = reshape(double(e_pred),[100,200]);


% figure; surf(T,X,U); axis tight; axis square; colormap jet; shading interp; view(37.5, 30);
% figure; surf(T,X,U_pred); axis tight; axis square; colormap jet; shading interp; view(37.5, 30);
% figure; surf(T,X,E); axis tight; axis square; colormap jet; shading interp; view(-37.5, 30);
% figure; surf(T,X,E_pred); axis tight; axis square; colormap jet; shading interp; view(-37.5, 30);

errors_u = zeros(100,1);
errors_e = zeros(100,1);
for i = 1:100
    errors_u(i) = norm(U(i,:) - U_pred(i,:))/norm(U(i,:));
    errors_e(i) = norm(E(i,:) - E_pred(i,:))/norm(E(i,:));
end

fig = figure();
set(fig,'units','normalized','outerposition',[0 0 1 0.2])

subplot(1,2,1)
plot(T(:,1),errors_u, 'LineWidth', 2)
xlabel('$t$')
ylabel('Rel. $L_2$ Error')
title('$P(t,\psi)$')
axis tight
set(gca,'FontSize',14);
set(gcf, 'Color', 'w');

subplot(1,2,2)
plot(T(:,1),errors_e, 'LineWidth', 2)
xlabel('$t$')
ylabel('Rel. $L_2$ Error')
title('$\mathcal{E}(t,\psi)$')
axis tight
set(gca,'FontSize',14);
set(gcf, 'Color', 'w');

export_fig ./turbulence_1D_dissipation_errors.png -r300