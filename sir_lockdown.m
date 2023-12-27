clear all; clc; close all; set(0,'DefaultLegendAutoUpdate','off');

% Set latex as default interpreter
set(groot, 'defaulttextinterpreter','latex');  
set(groot, 'defaultAxesTickLabelInterpreter','latex');  
set(groot, 'defaultLegendInterpreter','latex');

% Epidemiological Parameters
Rnot     = 2.5;            % Total number of transmissions of an infectious (I) person (Source: EC), Atkeson is 3
Tinf     = 7;              % Duration of infectious period (Source: EC), Atkeson is 18
beta     = Rnot/Tinf;      % Transmission rate of an infectious (I) person 
gamma    = 1/Tinf;         % Exit rate from I state into recovery or death (R)

implicit = 0;

T    = 400;    % Length of simulation
dt   = 0.1;    % Time step 
Nt   = T/dt+1;
time = linspace(0,T,Nt);

mu = zeros(3,Nt); 
S = zeros(Nt,1); I = zeros(Nt,1); R = zeros(Nt,1);
N = zeros(Nt,1); D = zeros(Nt,1);
 
% Transition matrix
A_t = cell(Nt,1);

% Initial condition
I0 = 0.35*10^(-4); % March 1st 2020 so that March 31st there are appx. 150,000 cases (50% more than measured)
S0 = 1-I0;
R0 = 0;


%% SIR MODEL EVENTUAL CASES FOR COMPARISON: solution to log(S) = R_0*(S - 1) or log(1-R) = -R_0*R
fun = @(x) log(x) - Rnot*(x-1);    % function of x alone
S_st = fzero(fun,[0.0001,0.99999]);
R_st = 1-S_st;

S_herd = 1./Rnot;
R_herd = (1-1./Rnot);

mu0 = [S0,I0,R0]';
mu(:,1) = mu0;

% DO NOTHING

for n=1:Nt
    S(n) = mu(1,n);
    I(n) = mu(2,n);
    R(n) = mu(3,n);
    Re(n) = Rnot.*S(n);
    A_t{n} = [
        % into S     into I   into R
        -beta*I(n), beta*I(n),  0;    % from S
        0,    -gamma ,   gamma;       % from I
        0,   0,    0];       % from R

    if implicit == 0
        mu(:,n+1) = dt*A_t{n}'*mu(:,n) + mu(:,n);
    else
        mu(:,n+1) = (eye(3)-dt*A_t{n}') \  mu(:,n);
    end
end
mu(:,Nt+1)=[];

%Convert to percentage terms
mu = 100*mu;

S = mu(1,:)'; I = mu(2,:)'; R = mu(3,:)'; N = S+I+R; D = 100 - N;
S_donothing = S; I_donothing = I; R_donothing = R;

%%%%%%%%
% SHORT LOCKDOWN
%%%%%%%

lock_start = 37; lock_end = lock_start+30; lock_severity = 0.7;
[value n_start] = min(abs(time - lock_start));
[value n_end] = min(abs(time - lock_end));

% Initial condition
I0 = 0.35*10^(-4); % March 1st 2020 so that March 31st there are appx. 150,000 cases (50% more than measured)
S0 = 1-I0;
R0 = 0;

mu0 = [S0,I0,R0]';
mu(:,1) = mu0;

lockdown = zeros(Nt,1);

for n=1:Nt
    S(n) = mu(1,n);
    I(n) = mu(2,n);
    R(n) = mu(3,n);
    if n>=n_start && n<=n_end
        lockdown(n) = lock_severity;
    end
    Re(n) = Rnot.*(1-lockdown(n))*S(n);
    A_t{n} = [
        % into S     into I   into R
        -beta*(1-lockdown(n))*I(n), beta*(1-lockdown(n))*I(n),  0;    % from S
        0,    -gamma ,   gamma;       % from I
        0,       0,     0];       % from R

    if implicit == 0
        mu(:,n+1) = dt*A_t{n}'*mu(:,n) + mu(:,n);
    else
        mu(:,n+1) = (eye(3)-dt*A_t{n}') \  mu(:,n);
    end
end
mu(:,Nt+1)=[];


%Convert to percentage terms
mu = 100*mu;

S = mu(1,:)'; I = mu(2,:)'; R = mu(3,:)'; N = S+I+R; D = 100 - N;

S_st = 100*S_st; R_st = 100*R_st;
R_herd = R_herd*100;
S_herd = S_herd*100;

Tplot = 150;

Rnot_path = beta.*(1-lockdown)/gamma;

figure(1)
subplot(2,3,1)
plot(time,lockdown,'LineWidth',2)
xlim([0 Tplot])
ylim([0 1])
line([lock_start lock_start],[0 1],'Color','k')
line([lock_end lock_end],[0 1],'Color','k')
title('Lockdown Severity, $\ell_t$')
grid on
xlabel('Days')

subplot(2,3,2)
plot(time,Rnot_path,time,ones(1,Nt),'k--','LineWidth',2)
xlim([0 Tplot])
ylim([0 3])
line([lock_start lock_start],[0 3],'Color','k')
line([lock_end lock_end],[0 3],'Color','k')
title('Normalized transmission rate, $\tilde{\mathcal{R}}_t$')
grid on
xlabel('Days')


fig1 = figure(1);
pos_fig1 = [300 300 700 400];
set(fig1,'Position',pos_fig1)

print -depsc SIR_lockdown_short0.eps

figure(2)
subplot(2,3,1)
plot(time,lockdown,'LineWidth',2)
xlim([0 Tplot])
ylim([0 1])
line([lock_start lock_start],[0 1],'Color','k')
line([lock_end lock_end],[0 1],'Color','k')
title('Lockdown Severity, $\ell_t$')
grid on
xlabel('Days')

subplot(2,3,2)
plot(time,Rnot_path,time,ones(1,Nt),'k--','LineWidth',2)
xlim([0 Tplot])
ylim([0 3])
line([lock_start lock_start],[0 3],'Color','k')
line([lock_end lock_end],[0 3],'Color','k')
title('Normalized transmission rate, $\tilde{\mathcal{R}}_t$')
grid on
xlabel('Days')

subplot(2,3,3)
plot(time,Re,time,ones(1,Nt),'k--','LineWidth',2)
xlim([0 Tplot])
ylim([0 3])
line([lock_start lock_start],[0 100],'Color','k')
line([lock_end lock_end],[0 3],'Color','k')
title('Effective reprod number, $\mathcal{R}^e = \mathcal{R}_0 \times S$')
grid on
xlabel('Days')

subplot(2,3,4)
plot(time,S,time,S_st*ones(1,Nt),'--',time,S_herd*ones(1,Nt),'-.','LineWidth',2)
legend('$S$','Do-nothing $S_\infty$','Do-nothing $S^*$','Location','SouthWest')
line([lock_start lock_start],[0 100],'Color','k')
line([lock_end lock_end],[0 100],'Color','k')
xlim([0 Tplot])
ylim([0 100])
title('Susceptible $S$ (\% of Population)')
grid on
xlabel('Days')

subplot(2,3,5)
plot(time,I,'LineWidth',2)
line([lock_start lock_start],[0 25],'Color','k')
line([lock_end lock_end],[0 25],'Color','k')
xlim([0 Tplot])
ylim([0 25])
title('Infectious $I$ (\% of Population)')
grid on
xlabel('Days')

subplot(2,3,6)
plot(time,R,time,R_st*ones(1,Nt),'--',time,R_herd*ones(1,Nt),'-.','LineWidth',2)
legend('$R$','Do-nothing $R_\infty$','Do-nothing $R^*$','Location','NorthWest')
line([lock_start lock_start],[0 100],'Color','k')
line([lock_end lock_end],[0 100],'Color','k')
xlim([0 Tplot])
ylim([0 100])
title('Recovered or Dead $R$ (\% of Population)')
grid on
xlabel('Days')

fig2 = figure(2);
pos_fig2 = [300 300 700 400];
set(fig2,'Position',pos_fig2)

print -depsc SIR_lockdown_short.eps


%%

figure(3)
plot(S,I,'r','LineWidth',2)
hold on
plot(S_donothing,I_donothing,'b--','LineWidth',2)
ylim([0 max(I_donothing*1.1)])
xlim([0 100])
line([S_herd S_herd],[0 100],'Color','k','LineWidth',2)
plot(min(S_donothing),0,'ob','MarkerFaceColor','b','MarkerSize',10)
plot(min(S),0,'or','MarkerFaceColor','r','MarkerSize',10)
plot(max(S),100*I0,'or','MarkerSize',10)
legend('Short, Tight Lockdown','Do-Nothing Trajectory','Do-Nothing Herd Imm $S^*$')
pbaspect([1.35 1 1])
n1 = 85*5-20; n2 = 115*5;
arrow3([S_donothing(n1),I_donothing(n1)],[S_donothing(n1),I_donothing(n1)+7])
arrow3([S_donothing(n1),I_donothing(n1)],[S_donothing(n1)-20,I_donothing(n1)])
arrow3([S_donothing(n2),I_donothing(n2)],[S_donothing(n2),I_donothing(n2)-7])
arrow3([S_donothing(n2),I_donothing(n2)],[S_donothing(n2)-20,I_donothing(n2)])
hold off
xlabel('Susceptibles $S$ (\% of Population)','Fontsize',20)
ylabel('Infectious $I$ (\% of Population)','Fontsize',20)
set(gca,'FontSize',16)
grid on
hold off

print -depsc SIR_lockdown_short_phase.eps









%%%%%%%%
% LONG LOCKDOWN
%%%%%%%

lock_start = 37; lock_end = lock_start+30; lock_severity = 0.3;
[value n_start] = min(abs(time - lock_start));
[value n_end] = min(abs(time - lock_end));

% Initial condition
I0 = 0.35*10^(-4); % March 1st 2020 so that March 31st there are appx. 150,000 cases (50% more than measured)
S0 = 1-I0;
R0 = 0;

mu0 = [S0,I0,R0]';
mu(:,1) = mu0;

lockdown = zeros(Nt,1);

for n=1:Nt
    S(n) = mu(1,n);
    I(n) = mu(2,n);
    R(n) = mu(3,n);
    if n>=n_start && n<=n_end
        lockdown(n) = lock_severity;
    end
    Re(n) = Rnot.*(1-lockdown(n))*S(n);
    A_t{n} = [
        % into S     into I   into R
        -beta*(1-lockdown(n))*I(n), beta*(1-lockdown(n))*I(n),  0;    % from S
        0,    -gamma ,   gamma;       % from I
        0,       0,     0];       % from R

    if implicit == 0
        mu(:,n+1) = dt*A_t{n}'*mu(:,n) + mu(:,n);
    else
        mu(:,n+1) = (eye(3)-dt*A_t{n}') \  mu(:,n);
    end
end
mu(:,Nt+1)=[];


%Convert to percentage terms
mu = 100*mu;

S = mu(1,:)'; I = mu(2,:)'; R = mu(3,:)'; N = S+I+R; D = 100 - N;


Tplot = 150;

Rnot_path = beta.*(1-lockdown)/gamma;


close all

figure(2)
subplot(2,3,1)
plot(time,lockdown,'LineWidth',2)
xlim([0 Tplot])
ylim([0 1])
line([lock_start lock_start],[0 1],'Color','k')
line([lock_end lock_end],[0 1],'Color','k')
title('Lockdown Severity, $\ell_t$')
grid on
xlabel('Days')

subplot(2,3,2)
plot(time,Rnot_path,time,ones(1,Nt),'k--','LineWidth',2)
xlim([0 Tplot])
ylim([0 3])
line([lock_start lock_start],[0 3],'Color','k')
line([lock_end lock_end],[0 3],'Color','k')
title('Normalized transmission rate, $\tilde{\mathcal{R}}_t$')
grid on
xlabel('Days')

subplot(2,3,3)
plot(time,Re,time,ones(1,Nt),'k--','LineWidth',2)
xlim([0 Tplot])
ylim([0 3])
line([lock_start lock_start],[0 100],'Color','k')
line([lock_end lock_end],[0 3],'Color','k')
title('Effective reprod number, $\mathcal{R}^e = \mathcal{R}_0 \times S$')
grid on
xlabel('Days')

subplot(2,3,4)
plot(time,S,time,S_st*ones(1,Nt),'--',time,S_herd*ones(1,Nt),'-.','LineWidth',2)
%legend('$S$','Do-nothing $S_\infty$','Do-nothing $S^*$','Location','SouthWest')
line([lock_start lock_start],[0 100],'Color','k')
line([lock_end lock_end],[0 100],'Color','k')
xlim([0 Tplot])
ylim([0 100])
title('Susceptible $S$ (\% of Population)')
grid on
xlabel('Days')

subplot(2,3,5)
plot(time,I,'LineWidth',2)
line([lock_start lock_start],[0 25],'Color','k')
line([lock_end lock_end],[0 25],'Color','k')
xlim([0 Tplot])
ylim([0 25])
title('Infectious $I$ (\% of Population)')
grid on
xlabel('Days')

subplot(2,3,6)
plot(time,R,time,R_st*ones(1,Nt),'--',time,R_herd*ones(1,Nt),'-.','LineWidth',2)
%legend('$R$','Do-nothing $R_\infty$','Do-nothing $R^*$','Location','NorthWest')
line([lock_start lock_start],[0 100],'Color','k')
line([lock_end lock_end],[0 100],'Color','k')
xlim([0 Tplot])
ylim([0 100])
title('Recovered or Dead $R$ (\% of Population)')
grid on
xlabel('Days')

fig2 = figure(2);
pos_fig2 = [300 300 700 400];
set(fig2,'Position',pos_fig2)

print -depsc SIR_lockdown_loose.eps


%%

figure(3)
plot(S,I,'r','LineWidth',2)
hold on
plot(S_donothing,I_donothing,'b--','LineWidth',2)
ylim([0 max(I_donothing*1.1)])
xlim([0 100])
line([S_herd S_herd],[0 100],'Color','k','LineWidth',2)
plot(min(S_donothing),0,'ob','MarkerFaceColor','b','MarkerSize',10)
plot(min(S),0,'or','MarkerFaceColor','r','MarkerSize',10)
plot(max(S),100*I0,'or','MarkerSize',10)
legend('Loose Lockdown','Do-Nothing Trajectory','Do-Nothing Herd Imm $S^*$')
pbaspect([1.35 1 1])
n1 = 85*5-20; n2 = 115*5;
arrow3([S_donothing(n1),I_donothing(n1)],[S_donothing(n1),I_donothing(n1)+7])
arrow3([S_donothing(n1),I_donothing(n1)],[S_donothing(n1)-20,I_donothing(n1)])
arrow3([S_donothing(n2),I_donothing(n2)],[S_donothing(n2),I_donothing(n2)-7])
arrow3([S_donothing(n2),I_donothing(n2)],[S_donothing(n2)-20,I_donothing(n2)])
hold off
xlabel('Susceptibles $S$ (\% of Population)','Fontsize',20)
ylabel('Infectious $I$ (\% of Population)','Fontsize',20)
set(gca,'FontSize',16)
grid on
hold off

print -depsc SIR_lockdown_loose_phase.eps







%%%%
% INTERMITTENT LOCKDOWNS
%%%%

%%

lock_severity = 0.8;

% Initial condition
I0 = 0.35*10^(-4); % March 1st 2020 so that March 31st there are appx. 150,000 cases (50% more than measured)
S0 = 1-I0;
R0 = 0;

mu0 = [S0,I0,R0]';
mu(:,1) = mu0;

lockdown = zeros(Nt,1);

capacity = 0.07;
n_start = -1000;

for n=1:Nt
    S(n) = mu(1,n);
    I(n) = mu(2,n);
    R(n) = mu(3,n);
    if I(n)>capacity
        n_start = n;
    end
    if n<n_start + 25/dt
        lockdown(n)=lock_severity;
    end
    Re(n) = Rnot.*(1-lockdown(n))*S(n);
    A_t{n} = [
        % into S     into I   into R
        -beta*(1-lockdown(n))*I(n), beta*(1-lockdown(n))*I(n),  0;    % from S
        0,    -gamma ,   gamma;       % from I
        0,       0,     0];       % from R

    if implicit == 0
        mu(:,n+1) = dt*A_t{n}'*mu(:,n) + mu(:,n);
    else
        mu(:,n+1) = (eye(3)-dt*A_t{n}') \  mu(:,n);
    end
end
mu(:,Nt+1)=[];


%Convert to percentage terms
mu = 100*mu;

S = mu(1,:)'; I = mu(2,:)'; R = mu(3,:)'; N = S+I+R; D = 100 - N;


Tplot = 300;

Rnot_path = beta.*(1-lockdown)/gamma;

close all

figure(2)
subplot(2,3,1)
plot(time,lockdown,'LineWidth',2)
xlim([0 Tplot])
ylim([0 1])
title('Lockdown Severity, $\ell_t$')
grid on
xlabel('Days')

subplot(2,3,2)
plot(time,Rnot_path,time,ones(1,Nt),'k--','LineWidth',2)
xlim([0 Tplot])
ylim([0 3])
title('Normalized transmission rate, $\tilde{\mathcal{R}}_t$')
grid on
xlabel('Days')

subplot(2,3,3)
plot(time,Re,time,ones(1,Nt),'k--','LineWidth',2)
xlim([0 Tplot])
ylim([0 3])
title('Effective reprod number, $\mathcal{R}^e = \mathcal{R}_0 \times S$')
grid on
xlabel('Days')

subplot(2,3,4)
plot(time,S,time,S_st*ones(1,Nt),'--',time,S_herd*ones(1,Nt),'-.','LineWidth',2)
xlim([0 Tplot])
ylim([0 100])
title('Susceptible $S$ (\% of Population)')
grid on
xlabel('Days')

subplot(2,3,5)
plot(time,I,'LineWidth',2)
xlim([0 Tplot])
ylim([0 25])
title('Infectious $I$ (\% of Population)')
grid on
xlabel('Days')

subplot(2,3,6)
plot(time,R,time,R_st*ones(1,Nt),'--',time,R_herd*ones(1,Nt),'-.','LineWidth',2)
xlim([0 Tplot])
ylim([0 100])
title('Recovered or Dead $R$ (\% of Population)')
grid on
xlabel('Days')

fig2 = figure(2);
pos_fig2 = [300 300 700 400];
set(fig2,'Position',pos_fig2)

print -depsc SIR_lockdown_intermittent.eps




figure(3)
plot(S,I,'r','LineWidth',2)
hold on
plot(S_donothing,I_donothing,'b--','LineWidth',2)
ylim([0 max(I_donothing*1.1)])
xlim([0 100])
line([S_herd S_herd],[0 100],'Color','k','LineWidth',2)
plot(min(S_donothing),0,'ob','MarkerFaceColor','b','MarkerSize',10)
plot(min(S),0,'or','MarkerFaceColor','r','MarkerSize',10)
plot(max(S),100*I0,'or','MarkerSize',10)
legend('Intermittent Lockdowns','Do-Nothing Trajectory','Do-Nothing Herd Imm $S^*$')
pbaspect([1.35 1 1])
n1 = 85*5-20; n2 = 115*5;
arrow3([S_donothing(n1),I_donothing(n1)],[S_donothing(n1),I_donothing(n1)+7])
arrow3([S_donothing(n1),I_donothing(n1)],[S_donothing(n1)-20,I_donothing(n1)])
arrow3([S_donothing(n2),I_donothing(n2)],[S_donothing(n2),I_donothing(n2)-7])
arrow3([S_donothing(n2),I_donothing(n2)],[S_donothing(n2)-20,I_donothing(n2)])
hold off
xlabel('Susceptibles $S$ (\% of Population)','Fontsize',20)
ylabel('Infectious $I$ (\% of Population)','Fontsize',20)
set(gca,'FontSize',16)
grid on
hold off

print -depsc SIR_lockdown_intermittent_phase.eps






%%%%
% INTERMITTENT LOCKDOWNS + ICU CAPACITY CONSTRAINT
%%%%

%%

lock_severity = 0.8;

% Initial condition
I0 = 0.35*10^(-4); % March 1st 2020 so that March 31st there are appx. 150,000 cases (50% more than measured)
S0 = 1-I0;
R0 = 0;

mu0 = [S0,I0,R0]';
mu(:,1) = mu0;

lockdown = zeros(Nt,1);

capacity = 0.05;
n_start = -1000;

for n=1:Nt
    S(n) = mu(1,n);
    I(n) = mu(2,n);
    R(n) = mu(3,n);
    if I(n)>capacity
        n_start = n;
    end
    if n<n_start + 25/dt
        lockdown(n)=lock_severity;
    end
    Re(n) = Rnot.*(1-lockdown(n))*S(n);
    A_t{n} = [
        % into S     into I   into R
        -beta*(1-lockdown(n))*I(n), beta*(1-lockdown(n))*I(n),  0;    % from S
        0,    -gamma ,   gamma;       % from I
        0,       0,     0];       % from R

    if implicit == 0
        mu(:,n+1) = dt*A_t{n}'*mu(:,n) + mu(:,n);
    else
        mu(:,n+1) = (eye(3)-dt*A_t{n}') \  mu(:,n);
    end
end
mu(:,Nt+1)=[];


%Convert to percentage terms
mu = 100*mu;

S = mu(1,:)'; I = mu(2,:)'; R = mu(3,:)'; N = S+I+R; D = 100 - N;


Tplot = 300;

Rnot_path = beta.*(1-lockdown)/gamma;

close all

figure(2)
subplot(2,3,1)
plot(time,lockdown,'LineWidth',2)
xlim([0 Tplot])
ylim([0 1])
title('Lockdown Severity, $\ell_t$')
grid on
xlabel('Days')

subplot(2,3,2)
plot(time,Rnot_path,time,ones(1,Nt),'k--','LineWidth',2)
xlim([0 Tplot])
ylim([0 3])
title('Normalized transmission rate, $\tilde{\mathcal{R}}_t$')
grid on
xlabel('Days')

subplot(2,3,3)
plot(time,Re,time,ones(1,Nt),'k--','LineWidth',2)
xlim([0 Tplot])
ylim([0 3])
title('Effective reprod number, $\mathcal{R}^e = \mathcal{R}_0 \times S$')
grid on
xlabel('Days')

subplot(2,3,4)
plot(time,S,time,S_st*ones(1,Nt),'--',time,S_herd*ones(1,Nt),'-.','LineWidth',2)
xlim([0 Tplot])
ylim([0 100])
title('Susceptible $S$ (\% of Population)')
grid on
xlabel('Days')

subplot(2,3,5)
plot(time,I,'LineWidth',2)
hold on 
plot(time,100*capacity*ones(Nt,1),'g','LineWidth',2)
legend('$I$','ICU Capacity')
xlim([0 Tplot])
ylim([0 25])
title('Infectious $I$ (\% of Population)')
grid on
xlabel('Days')
set(findobj('color','g'),'Color',[0 0.5 0]);

subplot(2,3,6)
plot(time,R,time,R_st*ones(1,Nt),'--',time,R_herd*ones(1,Nt),'-.','LineWidth',2)
xlim([0 Tplot])
ylim([0 100])
title('Recovered or Dead $R$ (\% of Population)')
grid on
xlabel('Days')

fig2 = figure(2);
pos_fig2 = [300 300 700 400];
set(fig2,'Position',pos_fig2)

print -depsc SIR_lockdown_ICU.eps


figure(3)
plot(S,I,'r','LineWidth',2)
hold on
plot(linspace(0,100,100),100*capacity*ones(100,1),'g','LineWidth',2)
plot(S_donothing,I_donothing,'b--','LineWidth',2)
ylim([0 max(I_donothing*1.1)])
xlim([0 100])
line([S_herd S_herd],[0 100],'Color','k','LineWidth',2)
plot(min(S_donothing),0,'ob','MarkerFaceColor','b','MarkerSize',10)
plot(min(S),0,'or','MarkerFaceColor','r','MarkerSize',10)
plot(max(S),100*I0,'or','MarkerSize',10)
legend('Intermittent Lockdowns','ICU Capacity','Do-Nothing Trajectory','Do-Nothing Herd Imm $S^*$')
pbaspect([1.35 1 1])
n1 = 85*5-20; n2 = 115*5;
arrow3([S_donothing(n1),I_donothing(n1)],[S_donothing(n1),I_donothing(n1)+7])
arrow3([S_donothing(n1),I_donothing(n1)],[S_donothing(n1)-20,I_donothing(n1)])
arrow3([S_donothing(n2),I_donothing(n2)],[S_donothing(n2),I_donothing(n2)-7])
arrow3([S_donothing(n2),I_donothing(n2)],[S_donothing(n2)-20,I_donothing(n2)])
hold off
xlabel('Susceptibles $S$ (\% of Population)','Fontsize',20)
ylabel('Infectious $I$ (\% of Population)','Fontsize',20)
set(gca,'FontSize',16)
grid on
hold off
set(findobj('color','g'),'Color',[0 0.5 0]);

print -depsc SIR_lockdown_ICU_phase.eps

%%

figure(4)
plot(linspace(0,100,100),100*capacity*ones(100,1),'g','LineWidth',2)
hold on
plot(S_donothing,I_donothing,'b--','LineWidth',2)
ylim([0 max(I_donothing*1.1)])
xlim([0 100])
line([S_herd S_herd],[0 100],'Color','k','LineWidth',2)
plot(min(S_donothing),0,'ob','MarkerFaceColor','b','MarkerSize',10)
plot(max(S),100*I0,'or','MarkerSize',10)
legend('ICU Capacity','Do-Nothing Trajectory','Do-Nothing Herd Imm $S^*$')
pbaspect([1.35 1 1])
n1 = 85*5-20; n2 = 115*5;
arrow3([S_donothing(n1),I_donothing(n1)],[S_donothing(n1),I_donothing(n1)+7])
arrow3([S_donothing(n1),I_donothing(n1)],[S_donothing(n1)-20,I_donothing(n1)])
arrow3([S_donothing(n2),I_donothing(n2)],[S_donothing(n2),I_donothing(n2)-7])
arrow3([S_donothing(n2),I_donothing(n2)],[S_donothing(n2)-20,I_donothing(n2)])
hold off
xlabel('Susceptibles $S$ (\% of Population)','Fontsize',20)
ylabel('Infectious $I$ (\% of Population)','Fontsize',20)
set(gca,'FontSize',16)
grid on
hold off
set(findobj('color','g'),'Color',[0 0.5 0]);

print -depsc SIR_ICU_phase.eps















%%%%
% ROLLOVER LOCKDOWN + ICU CAPACITY CONSTRAINT
%%%%

%%

lock_severity = 0.8;

% Initial condition
I0 = 0.35*10^(-4); % March 1st 2020 so that March 31st there are appx. 150,000 cases (50% more than measured)
S0 = 1-I0;
R0 = 0;

mu0 = [S0,I0,R0]';
mu(:,1) = mu0;

lockdown = zeros(Nt,1);

lock_end = 105;
[value n_end] = min(abs(time - lock_end));

capacity = 0.05;
n_start = 1000;

for n=1:Nt
    S(n) = mu(1,n);
    I(n) = mu(2,n);
    R(n) = mu(3,n);
    if I(n)>capacity
        n_start = n;
    end
    if n>=n_start && S(n)<=S_herd
        beta_lock = gamma/S(n);
        lockdown(n) = max(min(1 - beta_lock/beta,1),0);
    end
    if n>=n_end
        Rnot_lock = log(Rnot*S(n_end))/(1 - 1/Rnot - R(n_end));
        lockdown(n) = 1 - Rnot_lock/Rnot;
    end
    Re(n) = Rnot.*(1-lockdown(n))*S(n);
    A_t{n} = [
        % into S     into I   into R
        -beta*(1-lockdown(n))*I(n), beta*(1-lockdown(n))*I(n),  0;    % from S
        0,    -gamma ,   gamma;       % from I
        0,       0,     0];       % from R

    if implicit == 0
        mu(:,n+1) = dt*A_t{n}'*mu(:,n) + mu(:,n);
    else
        mu(:,n+1) = (eye(3)-dt*A_t{n}') \  mu(:,n);
    end
end
mu(:,Nt+1)=[];


%Convert to percentage terms
mu = 100*mu;

S = mu(1,:)'; I = mu(2,:)'; R = mu(3,:)'; N = S+I+R; D = 100 - N;


Tplot = 150;

Rnot_path = beta.*(1-lockdown)/gamma;

close all

figure(2)
subplot(2,3,1)
plot(time,lockdown,'LineWidth',2)
xlim([0 Tplot])
ylim([0 1])
title('Lockdown Severity, $\ell_t$')
grid on
xlabel('Days')

subplot(2,3,2)
plot(time,Rnot_path,time,ones(1,Nt),'k--','LineWidth',2)
xlim([0 Tplot])
ylim([0 3])
title('Normalized transmission rate, $\tilde{\mathcal{R}}_t$')
grid on
xlabel('Days')

subplot(2,3,3)
plot(time,Re,time,ones(1,Nt),'k--','LineWidth',2)
xlim([0 Tplot])
ylim([0 3])
title('Effective reprod number, $\mathcal{R}^e = \mathcal{R}_0 \times S$')
grid on
xlabel('Days')

subplot(2,3,4)
plot(time,S,time,S_st*ones(1,Nt),'--',time,S_herd*ones(1,Nt),'-.','LineWidth',2)
xlim([0 Tplot])
ylim([0 100])
title('Susceptible $S$ (\% of Population)')
grid on
xlabel('Days')

subplot(2,3,5)
plot(time,I,'LineWidth',2)
hold on 
plot(time,100*capacity*ones(Nt,1),'g--','LineWidth',2)
legend('$I$','ICU Capacity')
xlim([0 Tplot])
ylim([0 25])
title('Infectious $I$ (\% of Population)')
grid on
xlabel('Days')
set(findobj('color','g'),'Color',[0 0.5 0]);

subplot(2,3,6)
plot(time,R,time,R_st*ones(1,Nt),'--',time,R_herd*ones(1,Nt),'-.','LineWidth',2)
xlim([0 Tplot])
ylim([0 100])
title('Recovered or Dead $R$ (\% of Population)')
grid on
xlabel('Days')

fig2 = figure(2);
pos_fig2 = [300 300 700 400];
set(fig2,'Position',pos_fig2)

print -depsc SIR_lockdown_rollover.eps


figure(3)
plot(S,I,'r','LineWidth',2)
hold on
plot(linspace(0,100,100),100*capacity*ones(100,1),'g--','LineWidth',2)
plot(S_donothing,I_donothing,'b--','LineWidth',2)
ylim([0 max(I_donothing*1.1)])
xlim([0 100])
line([S_herd S_herd],[0 100],'Color','k','LineWidth',2)
plot(min(S_donothing),0,'ob','MarkerFaceColor','b','MarkerSize',10)
plot(min(S),0,'or','MarkerFaceColor','r','MarkerSize',10)
plot(max(S),100*I0,'or','MarkerSize',10)
legend('Lockdown with $\mathcal{R}^e_t=1$','ICU Capacity','Do-Nothing Trajectory','Do-Nothing Herd Imm $S^*$')
pbaspect([1.35 1 1])
n1 = 85*5-20; n2 = 115*5;
arrow3([S_donothing(n1),I_donothing(n1)],[S_donothing(n1),I_donothing(n1)+7])
arrow3([S_donothing(n1),I_donothing(n1)],[S_donothing(n1)-20,I_donothing(n1)])
arrow3([S_donothing(n2),I_donothing(n2)],[S_donothing(n2),I_donothing(n2)-7])
arrow3([S_donothing(n2),I_donothing(n2)],[S_donothing(n2)-20,I_donothing(n2)])
hold off
xlabel('Susceptibles $S$ (\% of Population)','Fontsize',20)
ylabel('Infectious $I$ (\% of Population)','Fontsize',20)
set(gca,'FontSize',16)
grid on
hold off
set(findobj('color','g'),'Color',[0 0.5 0]);
