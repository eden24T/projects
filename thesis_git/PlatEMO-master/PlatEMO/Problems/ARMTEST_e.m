classdef ARMTEST_e < PROBLEM
    % <multi> <real> <large/none> <expensive/none>

    properties(SetAccess = protected)
        initFcn    = {};        % Function for initializing solutions
        evalFcn    = {};        % Function for evaluating solutions
        decFcn     = {};    	% Function for repairing invalid solutions
        objFcn     = {};     	% Objective functions
        conFcn     = {};     	% Constraint functions
        objGradFcn = {};        % Function for calculating the gradients of objectives
        conGradFcn = {};        % Function for calculating the gradients of constraints
        data       = {};        % Data of the problem
        
    end
    
    methods
        %% Constructor
        
        function obj = ARMTEST_e(varargin)
            isStr = find(cellfun(@ischar,varargin(1:end-1))&~cellfun(@isempty,varargin(2:end)));
            for i = isStr(ismember(varargin(isStr),{'N','M','D','maxFE','maxRuntime','encoding','lower','upper','initFcn','evalFcn','decFcn','objFcn','conFcn','objGradFcn','conGradFcn','data'}))
                obj.(varargin{i}) = varargin{i+1};
            end
            obj.data       = Str2Fcn(obj.data,1,[],'dataset');
            obj.initFcn    = Str2Fcn(obj.initFcn,2,~isempty(obj.data),'initialization function');
            obj.evalFcn    = Str2Fcn(obj.evalFcn,3,~isempty(obj.data),'evaluation function');
            obj.decFcn     = Str2Fcn(obj.decFcn,3,~isempty(obj.data),'repair function');
            obj.objFcn     = Strs2Fcns(obj.objFcn,4,~isempty(obj.data),'objective function f');
            obj.conFcn     = Strs2Fcns(obj.conFcn,4,~isempty(obj.data),'constraint function g');
            obj.objGradFcn = Strs2Fcns(obj.objGradFcn,3,~isempty(obj.data),'gradient of objective fg');
            obj.conGradFcn = Strs2Fcns(obj.conGradFcn,3,~isempty(obj.data),'gradient of constraint gg');
            C   = obj.Initialization(100);
        end
        %% Default settings of the problem
        function Setting(obj)
            obj.M = 2;
            if isempty(obj.D); obj.D = 32; end
            obj.lower    = -100*ones(1,obj.D);
            obj.upper    = 100*ones(1,obj.D);
            obj.encoding = ones(1,obj.D);
        end
        
        %% Generate initial solutions
        function Population = Initialization(obj,N)
            if nargin < 2
                N = obj.N;
            end
            if ~isempty(obj.initFcn)
                Population = obj.Evaluation(CallFcn(obj.initFcn,N,obj.data,'initialization function',[N obj.D]));
            else
                Population = Initialization@PROBLEM(obj,N);
            end
            disp(size(Population.objs));
            obj.optimum = min(Population.objs,[],1);
        end
        %% Evaluate multiple solutions
        function Population = Evaluation(obj,varargin)
            if ~isempty(obj.evalFcn)
                for i = 1 : size(varargin{1},1)
                    [PopDec(i,:),PopObj(i,:),PopCon(i,:)] = CallFcn(obj.evalFcn,varargin{1}(i,:),obj.data,'evaluation function',[1 obj.D]);
                end
                Population = SOLUTION(PopDec,PopObj,PopCon,varargin{2:end});
                obj.FE     = obj.FE + length(Population);
            else
                a=varargin{1};
                Population = Evaluation@PROBLEM(obj,varargin{1});
            end
        end
        
        %% Calculate objective values
        function PopObj = CalObj(obj,PopDec)
            M = obj.M;
            N = obj.N;
            PopObj = zeros(N,M);
            f_obj = zeros(M,1);
            %PopCon = obj.CalCon(PopDec);
            PopCon = CalCon(obj,PopDec);
            edge_result1 = [10000, 10000];
            t1 = 0 : 0.1 : 5;
            test = 1;
            edge_result1_k=0;
            for k = 1:N
                    for i = 1:M
                        if i==1
                            if test==0 % Original Adham test
                              m_load = 0.7; %mass of the load at the end of the arm;
                            elseif test==1 %Ami & Adham test
                                  m_load = 0;%0
                            elseif test==2 %Ami & Adham test
                                m_load = 0;
                            elseif test==3 %Ami & Adham test
                                m_load = 1;
                            elseif test==4 % Eden test for very different trajectories
                                m_load=0.5;%0
                            elseif test==5 % Eden test for very different trajectories
                                m_load=0.4;    
                            end
                        else
                            if test==0
                              m_load = 1; %mass of the load at the end of the arm;
                            elseif test==1
                                  m_load = 1;%1
                            elseif test==2
                                m_load = 2;
                            elseif test==3
                                m_load = 2;
                            elseif test==4
                                m_load=0.5;%0.5
                            elseif test==5
                                m_load=0;    
                            end
                        end
                        %all_time = 0.01 : 0.01 : 5;
                        m1_sam = 0.1; % Mass of link 1
                        m2 = 0.2; % Mass of link 2
                        L1_sam = 1; % Length of link 1 % Adham (changed to suit the working area)
                        L2_sam = 0.5; % Length of link 2 % Adham (changed to suit the working area)
                        c1_sam = 0.03; % Viscouse friction coefficient for joint 1
                        c2_sam = 0.05; % Viscouse friction coefficient for joint 2
                        power = 1;
                        m2_sam = m2 + m_load; % Mass of link 2 including load at the end of the arm
                        g = 9.81;
                        tspan = 0:0.1:5;
                        y0 = [-12.61*pi/180; 69.36*pi/180; 0; 0]; % Initial condition [theta1 theta2; theta1_dot theta2_dot]
                        m1_model = 0.15;
                        m2_model_no_load = 0.25;
                        m_load_model = 0.6;%0.3
                        m2_model = m2_model_no_load + m_load_model;
                        L1_model = 1;
                        L2_model = 0.5;
                        c1_model = 0.04;
                        c2_model = 0.06;
                        
                        %% Encoding the ANN
                        %						Define the network weights
                        inputToHiddenWeights_Kv = PopDec(k,1:4);
                        hiddenToOutputWeights_Kv = PopDec(k,5:8);
                        inputToHiddenWeights_Kp = PopDec(k,17:20);
                        hiddenToOutputWeights_Kp = PopDec(k,21:24);
                        % Define the bias terms
                        hiddenBias_Kv = PopDec(k,9:12);
                        outputBias_Kv = PopDec(k,13:16);
                        hiddenBias_Kp = PopDec(k,25:28);
                        outputBias_Kp = PopDec(k,29:32);
                        length_line = length(inputToHiddenWeights_Kv)/2;
                        line1 = inputToHiddenWeights_Kv(1,1:length_line);
                        a = length_line+1;
                        l = length(inputToHiddenWeights_Kv);
                        line2 = inputToHiddenWeights_Kv(1,a:l);
                        inputToHiddenWeights_Kv = [line1; line2];
    
                        length_line = length(hiddenToOutputWeights_Kv)/2;
                        line1 = hiddenToOutputWeights_Kv(1,1:length_line);
                        a = length_line+1;
                        l = length(hiddenToOutputWeights_Kv);
                        line2 = hiddenToOutputWeights_Kv(1,a:l);
                        hiddenToOutputWeights_Kv = [line1; line2];
    
                        length_line = length(inputToHiddenWeights_Kp)/2;
                        line1 = inputToHiddenWeights_Kp(1,1:length_line);
                        a = length_line+1;
                        l = length(inputToHiddenWeights_Kp);
                        line2 = inputToHiddenWeights_Kp(1,a:l);
                        inputToHiddenWeights_Kp = [line1; line2];
    
                        length_line = length(hiddenToOutputWeights_Kp)/2;
                        line1 = hiddenToOutputWeights_Kp(1,1:length_line);
                        a = length_line+1;
                        l = length(hiddenToOutputWeights_Kp);
                        line2 = hiddenToOutputWeights_Kp(1,a:l);
                        hiddenToOutputWeights_Kp = [line1; line2];
    
                        length_line = length(hiddenBias_Kv)/2;
                        line1 = hiddenBias_Kv(1,1:length_line);
                        a = length_line+1;
                        l = length(hiddenBias_Kv);
                        line2 = hiddenBias_Kv(1,a:l);
                        hiddenBias_Kv = [line1; line2];
    
                        length_line = length(outputBias_Kv)/2;
                        line1 = outputBias_Kv(1,1:length_line);
                        a = length_line+1;
                        l = length(outputBias_Kv);
                        line2 = outputBias_Kv(1,a:l);
                        outputBias_Kv = [line1; line2];
    
                        length_line = length(hiddenBias_Kp)/2;
                        line1 = hiddenBias_Kp(1,1:length_line);
                        a = length_line+1;
                        l = length(hiddenBias_Kp);
                        line2 = hiddenBias_Kp(1,a:l);
                        hiddenBias_Kp = [line1; line2];
    
                        length_line = length(outputBias_Kp)/2;
                        line1 = outputBias_Kp(1,1:length_line);
                        a = length_line+1;
                        l = length(outputBias_Kp);
                        line2 = outputBias_Kp(1,a:l);
                        outputBias_Kp = [line1; line2];
                        % Define network architecture
                        inputSize = size(inputToHiddenWeights_Kv, 2);
                        outputSize = size(hiddenToOutputWeights_Kv, 1);
                        hiddenSize = size(hiddenBias_Kv, 1);
                        % Input data
                        %                       inputData = randn(inputSize,hiddenSize);
                        % Call the calcGains function to obtain K_v and K_p
                        %[K_v, K_p] = calcGains(inputData, inputToHiddenWeights_Kv, hiddenToOutputWeights_Kv, hiddenBias_Kv, outputBias_Kv, inputToHiddenWeights_Kp, hiddenToOutputWeights_Kp, hiddenBias_Kp, outputBias_Kp);
                        [t, y] = ode45(@(t, y) arm(t, y, m1_sam, m2_sam, m2, L1_sam, L2_sam, c1_sam, c2_sam, g, power, m1_model, m2_model, m2_model_no_load, L1_model, L2_model, c1_model, c2_model,inputToHiddenWeights_Kv, hiddenToOutputWeights_Kv, inputToHiddenWeights_Kp, hiddenToOutputWeights_Kp, hiddenBias_Kv, outputBias_Kv, hiddenBias_Kp, outputBias_Kp,m_load,k,edge_result1_k), tspan, y0);
                        for j = 1:numel(t)
                            [~,theta_2dot_actual(j,:),tau] = arm(t(j), y(j,:), m1_sam, m2_sam, m2, L1_sam, L2_sam, c1_sam, c2_sam, g, power, m1_model, m2_model, m2_model_no_load, L1_model, L2_model, c1_model, c2_model, inputToHiddenWeights_Kv, hiddenToOutputWeights_Kv, inputToHiddenWeights_Kp, hiddenToOutputWeights_Kp, hiddenBias_Kv, outputBias_Kv, hiddenBias_Kp, outputBias_Kp,m_load,k,edge_result1_k);
                        end
                        [x_t_D, Vx_t_D, Ax_t_D, y_t_D, Vy_t_D, Ay_t_D] = trajectory_org(t);
                        [x_t_D_k, Vx_t_D_k, Ax_t_D_k, y_t_D_k, Vy_t_D_k, Ay_t_D_k] = trajectory_org(t1);
                        [x_t_D_s, Vx_t_D_s, Ax_t_D_s, y_t_D_s, Vy_t_D_s, Ay_t_D_s] = cube_spline_trajectory(t1);
                        [x_t_D_c, Vx_t_D_c, Ax_t_D_c, y_t_D_c, Vy_t_D_c, Ay_t_D_c] = circle_trajectory(t1);
                        [t1, y_k] = ode45(@(t1, y_k) arm(t1, y_k, m1_sam, m2_sam, m2, L1_sam, L2_sam, c1_sam, c2_sam, g, power, m1_model, m2_model, m2_model_no_load, L1_model, L2_model, c1_model, c2_model,inputToHiddenWeights_Kv, hiddenToOutputWeights_Kv, inputToHiddenWeights_Kp, hiddenToOutputWeights_Kp, hiddenBias_Kv, outputBias_Kv, hiddenBias_Kp, outputBias_Kp,m_load,k,edge_result1_k), tspan, y0);
                        for j = 1:numel(t1)
                            [~,theta_2dot_actual_k(j,:),tau] = arm(t1(j), y_k(j,:), m1_sam, m2_sam, m2, L1_sam, L2_sam, c1_sam, c2_sam, g, power, m1_model, m2_model, m2_model_no_load, L1_model, L2_model, c1_model, c2_model, inputToHiddenWeights_Kv, hiddenToOutputWeights_Kv, inputToHiddenWeights_Kp, hiddenToOutputWeights_Kp, hiddenBias_Kv, outputBias_Kv, hiddenBias_Kp, outputBias_Kp,m_load,k,edge_result1_k);
                        end
                        [x_actual_k, y_actual_k, x_dot_actual_k, y_dot_actual_k, x_2dot_actual_k, y_2dot_actual_k] = FOR_KIN(y_k(:,1), y_k(:,2), y_k(:,3), y_k(:,4), theta_2dot_actual_k(:,1), theta_2dot_actual_k(:,2), L1_model, L2_model);
                        [x_actual, y_actual, x_dot_actual, y_dot_actual, x_2dot_actual, y_2dot_actual] = FOR_KIN(y(:,1), y(:,2), y(:,3), y(:,4), theta_2dot_actual(:,1), theta_2dot_actual(:,2), L1_model, L2_model);
                        %[x_actual, y_actual, x_dot_actual, y_dot_actual, x_2dot_actual, y_2dot_actual] = FOR_KIN(y(k,1), y(k,2), y(k,3), y(k,4), theta_2dot_actual(k,1), theta_2dot_actual(k,2), L1_model, L2_model);
                        R_a =  sqrt((x_actual.^2)+(y_actual.^2));
                        R_d =  sqrt((x_t_D.^2)+(y_t_D.^2));
                        %R_a_dot = ((x_actual.*x_dot_actual)+(y_actual.*y_dot_actual))/sqrt((x_actual.^2)+(y_actual.^2));
                        %R_d_dot = ((x_t_D.*Vx_t_D)+(y_t_D.*Vy_t_D))/sqrt((x_t_D.^2)+(y_t_D.^2));
                        %R_a_2dot = sqrt((x_2dot_actual^2)+(y_2dot_actual^2));
                        %R_d_2dot = sqrt((Ax_t_D^2)+(Ay_t_D^2));
                        %f1 = @(t) (R_a - R_d).^2;
                        ts_i = length(t);
                        numerator = trapz(t(1:ts_i),(R_a(1:ts_i) - R_d(1:ts_i)).^2);
                        denominator = trapz(t(1:ts_i),ones(size(t(1:ts_i))));
                        f_obj(i) = numerator/denominator;
                        %f_obj(i) = (sum((R_a(1:ts_i) - R_d(1:ts_i)).^2))/ts_i;
                        %f_obj(i) = @(t) ((R_a - R_d).^2);
                        %PopObj(:,i) = integral(f1,1,1.5,'ArrayValued',true);
                        %PopObj(1:size(integral),i) = integral;
                        %if PopCon(k)>0
                        if max(PopCon,[],2)>0    
                            PopObj(k,i) = f_obj(i)+80;
                        else
                            PopObj(k,i) = f_obj(i);
                        end
                        if i==1    
                            x_actual_for_i2 = x_actual_k;
                            x_dot_actual_for_i2 = x_dot_actual_k;
                            x_2dot_actual_for_i2 =  x_2dot_actual_k;
                            y_actual_for_i2 = y_actual_k;
                            y_dot_actual_for_i2 = y_dot_actual_k;
                            y_2dot_actual_for_i2 = y_2dot_actual_k;
                            %x_t_D_spline = x_t_D_s;
                            %Vx_t_D_spline = Vx_t_D_s;
                            %Ax_t_D_spline = Ax_t_D_s;
                            %y_t_D_spline = y_t_D_s;
                            %Vy_t_D_spline = Vy_t_D_s;
                            %Ay_t_D_spline = Ay_t_D_s;
                            %x_t_D_circle = x_t_D_c;
                            %Vx_t_D_circle = Vx_t_D_c;
                            %Ax_t_D_circle = Ax_t_D_c;
                            %y_t_D_circle = y_t_D_c;
                            %Vy_t_D_circle = Vy_t_D_c;
                            %Ay_t_D_circle = Ay_t_D_c;
                            if PopObj(k,1) < edge_result1(:,1)
                                edge_result1(:,1) = PopObj(k,1);
                                edge_result1_k=k;
                                x_actual_k1 = x_actual_k;
                                x_dot_actual_k1 = x_dot_actual_k;
                                x_2dot_actual_k1 =  x_2dot_actual_k;
                                y_actual_k1 = y_actual_k;
                                y_dot_actual_k1 = y_dot_actual_k;
                                y_2dot_actual_k1 = y_2dot_actual_k;
                            end
                        end
                        if i==2
                            if k==edge_result1_k
                                x_actual_k2 = x_actual_k;
                                x_dot_actual_k2 = x_dot_actual_k;
                                x_2dot_actual_k2 =  x_2dot_actual_k;
                                y_actual_k2 = y_actual_k;
                                y_dot_actual_k2 = y_dot_actual_k;
                                y_2dot_actual_k2 = y_2dot_actual_k;
                            end
                            if PopObj(k,2) < edge_result1(:,2)
                                edge_result1(:,2) = PopObj(k,2);
                                %edge_result2_k = k;
                                x_actual_k4 = x_actual_k;
                                x_dot_actual_k4 = x_dot_actual_k;
                                x_2dot_actual_k4 =  x_2dot_actual_k;
                                y_actual_k4 = y_actual_k;
                                y_dot_actual_k4 = y_dot_actual_k;
                                y_2dot_actual_k4 = y_2dot_actual_k;
                                x_actual_k3 = x_actual_for_i2;
                                x_dot_actual_k3 = x_dot_actual_for_i2;
                                x_2dot_actual_k3 =  x_2dot_actual_for_i2;
                                y_actual_k3 = y_actual_for_i2;
                                y_dot_actual_k3 = y_dot_actual_for_i2;
                                y_2dot_actual_k3 = y_2dot_actual_for_i2;
                            end
                        end
                    
                    end  
                    
                    if k==N
                        
                        figure(1)
                        clf;
                        subplot(3,2,1)
                        hold on
                        plot(t1,x_actual_k1,'.k')
                        plot(t1,x_actual_k2,'ob')
                        plot(t1,x_t_D_k,'-r')
                        legend('for obj 1','for obj 2','desired')
                        grid on;xlabel('t[sec]');ylabel('x[m]')
                        
                        subplot(3,2,2)
                        hold on
                        plot(t1,y_actual_k1,'.k')
                        plot(t1,y_actual_k2,'ob')
                        plot(t1,y_t_D_k,'-r')
                        legend('for obj 1','for obj 2','desired')
                        grid on;xlabel('t[sec]');ylabel('y[m]')
                        
                        
                        subplot(3,2,3)
                        hold on
                        plot(t1,x_dot_actual_k1,'.k')
                        plot(t1,x_dot_actual_k2,'ob')
                        plot(t1,Vx_t_D_k,'-r')
                        legend('for obj 1','for obj 2','desired')
                        grid on;xlabel('t[sec]');ylabel('Vx[m/s]')
                        
                        subplot(3,2,4)
                        hold on
                        plot(t1,y_dot_actual_k1,'.k')
                        plot(t1,y_dot_actual_k2,'ob')
                        plot(t1,Vy_t_D_k,'-r')
                        legend('for obj 1','for obj 2','desired')
                        grid on;xlabel('t[sec]');ylabel('Vy[m/s]')
                        
                        subplot(3,2,5)
                        hold on
                        plot(t1,x_2dot_actual_k1,'.k')
                        plot(t1,x_2dot_actual_k2,'ob')
                        plot(t1,Ax_t_D_k,'-r')
                        legend('for obj 1','for obj 2','desired')
                        grid on;xlabel('t[sec]');ylabel('Ax[m/s2]')
                        
                        subplot(3,2,6)
                        hold on
                        plot(t1,y_2dot_actual_k1,'.k')
                        plot(t1,y_2dot_actual_k2,'ob')
                        plot(t1,Ay_t_D_k,'-r')
                        legend('for obj 1','for obj 2','desired')
                        grid on;xlabel('t[sec]');ylabel('Ay[m/s2]')

                        figure(2)
                        clf;
                        subplot(3,2,1)
                        hold on
                        plot(t1,x_actual_k3,'.k')
                        plot(t1,x_actual_k4,'ob')
                        plot(t1,x_t_D_k,'-r')
                        legend('for obj 1','for obj 2','desired')
                        grid on;xlabel('t[sec]');ylabel('x[m]')
                        
                        subplot(3,2,2)
                        hold on
                        plot(t1,y_actual_k3,'.k')
                        plot(t1,y_actual_k4,'ob')
                        plot(t1,y_t_D_k,'-r')
                        legend('for obj 1','for obj 2','desired')
                        grid on;xlabel('t[sec]');ylabel('y[m]')
                        
                        
                        subplot(3,2,3)
                        hold on
                        plot(t1,x_dot_actual_k3,'.k')
                        plot(t1,x_dot_actual_k4,'ob')
                        plot(t1,Vx_t_D_k,'-r')
                        legend('for obj 1','for obj 2','desired')
                        grid on;xlabel('t[sec]');ylabel('Vx[m/s]')
                        
                        subplot(3,2,4)
                        hold on
                        plot(t1,y_dot_actual_k3,'.k')
                        plot(t1,y_dot_actual_k4,'ob')
                        plot(t1,Vy_t_D_k,'-r')
                        legend('for obj 1','for obj 2','desired')
                        grid on;xlabel('t[sec]');ylabel('Vy[m/s]')
                        
                        subplot(3,2,5)
                        hold on
                        plot(t1,x_2dot_actual_k3,'.k')
                        plot(t1,x_2dot_actual_k4,'ob')
                        plot(t1,Ax_t_D_k,'-r')
                        legend('for obj 1','for obj 2','desired')
                        grid on;xlabel('t[sec]');ylabel('Ax[m/s2]')
                        
                        subplot(3,2,6)
                        hold on
                        plot(t1,y_2dot_actual_k3,'.k')
                        plot(t1,y_2dot_actual_k4,'ob')
                        plot(t1,Ay_t_D_k,'-r')
                        legend('for obj 1','for obj 2','desired')
                        grid on;xlabel('t[sec]');ylabel('Ay[m/s2]')
    
                    end
            end
        end
    end
end

%% Calculate constraint violations
function PopCon = CalCon(obj,PopDec)
    % This function calculates the constraint violations.
    % PopDec is the decision variable matrix. Each row is a solution.
    if ~isempty(obj.conFcn)
        PopCon = zeros(size(PopDec,1),length(obj.conFcn));
        for i = 1 : size(PopDec,1)
            for j = 1 : length(obj.conFcn)
                PopCon(i,j) = CallFcn(obj.conFcn{j},PopDec(i,:),obj.data,sprintf('constraint function g%d',j),[1 1]);
            end
        end
    else
        N = obj.N;
        PopCon = zeros(N, 5); % 5 columns for 5 constraints
        for k = 1:N
            solution = PopDec(k,:);
            PopCon(k,:) = constraint(obj, solution);
        end
        %PopCon = constraint(obj, solution);
    end
end


function [K_v, K_p] = calcGains(inputData, inputToHiddenWeights_Kv, hiddenToOutputWeights_Kv, hiddenBias_Kv, outputBias_Kv, inputToHiddenWeights_Kp, hiddenToOutputWeights_Kp, hiddenBias_Kp, outputBias_Kp,m_load,k,edge_result1_k)
% Call the feedForNN function with provided weights and bias, we want the matrices to be diagonal
K_v = feedForNN(inputData, inputToHiddenWeights_Kv, hiddenToOutputWeights_Kv, hiddenBias_Kv, outputBias_Kv,m_load,k,edge_result1_k);
K_p = feedForNN(inputData, inputToHiddenWeights_Kp, hiddenToOutputWeights_Kp, hiddenBias_Kp, outputBias_Kp,m_load,k,edge_result1_k);
end

function [dydt,theta_2dot_actual,tau] = arm(t, y, m1_sam, m2_sam, m2, L1_sam, L2_sam, c1_sam, c2_sam, g, power, m1_model, m2_model, m2_model_no_load, L1_model, L2_model, c1_model, c2_model, inputToHiddenWeights_Kv, hiddenToOutputWeights_Kv, inputToHiddenWeights_Kp, hiddenToOutputWeights_Kp, hiddenBias_Kv, outputBias_Kv, hiddenBias_Kp, outputBias_Kp,m_load,k,edge_result1_k)
theta1 = y(1);
theta2 = y(2);
theta1_dot = y(3);
theta2_dot = y(4);

theta =  [theta1; theta2];
theta_dot = [theta1_dot; theta2_dot];


[x_t_D, Vx_t_D, Ax_t_D, y_t_D, Vy_t_D, Ay_t_D] = trajectory_org(t);
theta_D_inv = INV_KIN(x_t_D, y_t_D, L1_model, L2_model);
[J, J_inv, det_J] = jacobian(theta_D_inv, L1_model, L2_model, t);
theta_dotD_inv = J_inv * [Vx_t_D; Vy_t_D];
J_inv_dot = jacobian_inv_dot(theta_D_inv, theta_dotD_inv, J, det_J, L1_model, L2_model, t);
theta_2dotD_inv = J_inv_dot * [Vx_t_D; Vy_t_D] + J_inv * [Ax_t_D; Ay_t_D];



% Input data
%inputData = randn(inputSize,2);
error = [theta - theta_D_inv, theta - theta_dotD_inv];
inputData = error;
% Call the calcGains function to obtain K_v and K_p
[K_v, K_p] = calcGains(inputData, inputToHiddenWeights_Kv, hiddenToOutputWeights_Kv, hiddenBias_Kv, outputBias_Kv, inputToHiddenWeights_Kp, hiddenToOutputWeights_Kp, hiddenBias_Kp, outputBias_Kp,m_load,k,edge_result1_k);
%disp(K_p);
%disp(K_v);


M = [m2_sam * L2_sam^2 + 2  * m2 * L1_sam *  L2_sam * cos(theta2) + (m1_sam + m2_sam) * L1_sam^2   m2_sam * L2_sam^2 + m2_sam * L1_sam * L2_sam * cos(theta2);
    m2_sam * L2_sam^2 + m2_sam * L1_sam * L2_sam * cos(theta2)      m2_sam * L2_sam^2]; % Inertia matrix
Cor = [-m2_sam * L1_sam * L2_sam * sin(theta2) * theta1_dot^2 + -2 * m2_sam * L1_sam * L2_sam * sin(theta2) * theta2_dot * theta2_dot;  m2_sam * L1_sam * L2_sam * sin(theta2) * theta1_dot^2]; % Coriolis and centrifugal vector (V vector from stage 1)
G = [(m1_sam  + m2_sam) * L1_sam * g * cos(theta1) + m2_sam * L2_sam * g * cos(theta1 + theta2);m2_sam * L2_sam * g * cos(theta1 + theta2)]; % Gravity vector
F = [-c1_sam*(theta1_dot^power);
    -c2_sam*(theta2_dot^power)]; % Viscous friction vector


M_model = [m2_model * L2_model^2 + 2  * m2_model_no_load * L1_model *  L2_model * cos(theta_D_inv(2)) + (m1_model + m2_model) * L1_model^2   m2_model * L2_model^2 + m2_model * L1_model * L2_model * cos(theta_D_inv(2));
    m2_model * L2_model^2 + m2_model * L1_model * L2_model * cos(theta_D_inv(2))      m2_model * L2_model^2]; % Inertia matrix calculated by the model parameters
Cor_model = [-m2_model * L1_model * L2_model * sin(theta_D_inv(2)) * theta_dotD_inv(1)^2 + -2 * m2_model * L1_model * L2_model * sin(theta_D_inv(2)) * theta2_dot * theta2_dot;  m2_model * L1_model * L2_model * sin(theta_D_inv(2)) * theta_dotD_inv(1)^2]; % Coriolis and centrifugal vector (V vector from stage 1) calculated by the model parameters
G_model = [(m1_model  + m2_model) * L1_model * g * cos(theta_D_inv(1)) + m2_model * L2_model * g * cos(theta_D_inv(1) + theta_D_inv(2));m2_model * L2_model * g * cos(theta_D_inv(1) + theta_D_inv(2))]; % Gravity vector calculated by the model parameters
F_model = [-c1_model*(theta_dotD_inv(1)^power);
    -c2_model*(theta_dotD_inv(2)^power)]; % Viscous friction vector calculated by the model parameters

alpha = M_model;
beta = Cor_model + G_model + F_model;
ftag = theta_2dotD_inv + K_v*(theta_dot-theta_dotD_inv) + K_p*(theta - theta_D_inv);
f = alpha * ftag + beta; % Control law
denomenator = f - Cor - G - F;
theta_2dot_trn = denomenator \ M;
theta_2dot = theta_2dot_trn.';


theta_2dot_actual = theta_2dot;
tau = f;


dydt = [theta1_dot;
    theta2_dot;
    theta_2dot(1);
    theta_2dot(2)];
end

%Defining desired trajectory as functions of time
function [x_t_D, Vx_t_D, Ax_t_D, y_t_D, Vy_t_D, Ay_t_D] = trajectory_org(t)
    % options = linspace(1.2, 1.3, 51); % Creates 51 evenly spaced numbers from 1 to 1.75
    % x0 = options(randi(length(options))); % Randomly selects one number from these options
    % %x0 = 1.250; % Start point
    % options = linspace(0.1, 0.3, 51); % Creates 51 evenly spaced numbers from 1 to 1.75
    % y0 = options(randi(length(options))); % Randomly selects one number from these options
    % %y0 = 0.2; % Start point
    % options = linspace(-0.3, -0.1, 51); % Creates 51 evenly spaced numbers from 1 to 1.75
    % xf = options(randi(length(options))); % Randomly selects one number from these options
    % %xf = -0.3; % End point
    % options = linspace(1.1, 1.3, 51); % Creates 51 evenly spaced numbers from 1 to 1.75
    % yf = options(randi(length(options))); % Randomly selects one number from these options
    % %yf = 1.3; % End point
    % tf = 5; % Time
    % Vx_0 = 0; % Start velocity
    % Vy_0 = 0; % Start velocity
    % Vx_f = 0; % End velocity
    % Vy_f = 0; % End velocity
    % a0 = x0; % a0=0
    % a1 = Vx_0; % a1=0
    % a2 = (3/(tf^2))*(xf-x0) - (2/tf)*Vx_0 - Vx_f/tf;
    % a3 = (-2/(tf^3))*(xf-x0) + (Vx_f+Vx_0)/(tf^2);
    % x_t_D = a0 + a1*t + a2*(t.^2) + a3*(t.^3);
    % Vx_t_D = a1 + 2*a2*t + 3*a3*(t.^2);
    % Ax_t_D = 2*a2 + 6*a3*t;
    % b0 = y0; % b0=0
    % b1 = Vy_0; % b1=0
    % b2 = (3/(tf^2))*(yf-y0) - (2/tf)*Vy_0 - Vy_f/tf;
    % b3 = (-2/(tf^3))*(yf-y0) + (Vy_f+Vy_0)/(tf^2);
    % y_t_D = b0 + b1*t + b2*(t.^2) + b3*(t.^3);
    % Vy_t_D = b1 + 2*b2*t + 3*b3*(t.^2);
    % Ay_t_D = 2*b2 + 6*b3*t;
    x0 = 1.25; % Start point%1.25
    y0 = 0.2; % Start point%0.2
    xf = -0.3; % End point%0.3
    yf = 1.3; % End point%1
    tf = 5; % Time
    Vx_0 = 0; % Start velocity
    Vy_0 = 0; % Start velocity
    Vx_f = 0; % End velocity
    Vy_f = 0; % End velocity
    a0 = x0; % a0=0
    a1 = Vx_0; % a1=0
    a2 = (3/(tf^2))*(xf-x0) - (2/tf)*Vx_0 - Vx_f/tf;
    a3 = (-2/(tf^3))*(xf-x0) + (Vx_f+Vx_0)/(tf^2);
    x_t_D = a0 + a1*t + a2*(t.^2) + a3*(t.^3);
    Vx_t_D = a1 + 2*a2*t + 3*a3*(t.^2);
    Ax_t_D = 2*a2 + 6*a3*t;
    b0 = y0; % b0=0
    b1 = Vy_0; % b1=0
    b2 = (3/(tf^2))*(yf-y0) - (2/tf)*Vy_0 - Vy_f/tf;
    b3 = (-2/(tf^3))*(yf-y0) + (Vy_f+Vy_0)/(tf^2);
    y_t_D = b0 + b1*t + b2*(t.^2) + b3*(t.^3);
    Vy_t_D = b1 + 2*b2*t + 3*b3*(t.^2);
    Ay_t_D = 2*b2 + 6*b3*t;
end

%Defining desired trajectory as functions of time
function [x_t_D, Vx_t_D, Ax_t_D, y_t_D, Vy_t_D, Ay_t_D] = cube_spline_trajectory(t)
    x_points = cos(t);
    y_points = sin(t);
    % Generate splines
    x_spline = spline(t,x_points);
    y_spline = spline(t,y_points);
    x_t_D = ppval(x_spline, t);
    y_t_D = ppval(y_spline,t);
    pp_d1_x = x_spline; % Duplicate x_spline structure for differentiation
    pp_d1_y = y_spline; % Duplicate y_spline structure for differentiation
    pp_d1_x.order = x_spline.order-1; % Decrease order by 1 for the derivative
    pp_d1_y.order = y_spline.order-1; % Decrease order by 1 for the derivative
    % Each row of coefs represents the coefficients of a polynomial segment
    % Multiply each by their respective powers to differentiate
    pp_d1_x.coefs = zeros(size(x_spline.coefs,1),3);
    pp_d1_y.coefs = zeros(size(y_spline.coefs,1),3);
    for i = 1:size(x_spline.coefs,1)
        pp_d1_x.coefs(i,:) = polyder(x_spline.coefs(i,:));
    end
    for i = 1:size(y_spline.coefs,1)
        pp_d1_y.coefs(i,:) = polyder(y_spline.coefs(i,:));
    end
    Vx_t_D = ppval(pp_d1_x, t); % Evaluate the first derivative at all points in t
    Vy_t_D = ppval(pp_d1_x, t); % Evaluate the first derivative at all points in t
    pp_d2_x = pp_d1_x;
    pp_d2_y = pp_d1_y;
    pp_d2_x.order = pp_d1_x.order-1;
    pp_d2_y.order = pp_d1_y.order-1;
    pp_d2_x.coefs = zeros(size(pp_d1_x.coefs,1),2);
    pp_d2_y.coefs = zeros(size(pp_d1_y.coefs,1),2);
    for i = 1:size(pp_d1_x.coefs,1)
       pp_d2_x.coefs(i,:) = polyder(pp_d1_x.coefs(i,:));
    end
    for i = 1:size(pp_d1_y.coefs,1)
       pp_d2_y.coefs(i,:) = polyder(pp_d1_y.coefs(i,:));
    end
    Ax_t_D = ppval(pp_d2_x, t);
    Ay_t_D = ppval(pp_d2_y, t);
end

function [x_t_D, Vx_t_D, Ax_t_D, y_t_D, Vy_t_D, Ay_t_D] = circle_trajectory(t)
   x_center = 0.475;
   y_center =0.75; 
   radius = 0.55;
   omega = 65*pi/180; % Omega is the angular velocity in rad/sec
   x_t_D = x_center + radius * cos(omega * t);
   y_t_D = y_center + radius * sin(omega * t);
   Vx_t_D = -radius * omega * sin(omega * t);
   Vy_t_D = radius * omega * cos(omega * t);
   Ax_t_D = -radius * omega^2 * cos(omega * t);
   Ay_t_D = -radius * omega^2 * sin(omega * t);
end

function theta_D_inv = INV_KIN(x_t_D, y_t_D, L1_model, L2_model)
	r = sqrt(x_t_D.^2 + y_t_D.^2); %The distance from the origin to the desired end effector position
	cos_theta2 = (r.^2 - L1_model^2 - L2_model^2) / (2 * L1_model * L2_model); %Calculate theta2 using the law of cosines
	if abs(cos_theta2) > 1
	    error('Desired position is out of reach');
	end
	theta2_D_inv = acos(cos_theta2);
	theta1_D_inv = atan2(y_t_D, x_t_D) - atan2((L2_model * sin(theta2_D_inv)), (L1_model + L2_model * cos(theta2_D_inv))); % Calculate theta1 using trigonometry
	theta_D_inv = [theta1_D_inv; theta2_D_inv]; %Vector of desired angels after inverse kinematic conversion
end


function [J, J_inv, det_J] = jacobian(theta_D_inv, L1_model, L2_model, t)
	J = zeros(2*length(t), 2); %Initialize the Jacobian matrix
	J_inv = zeros(2*length(t), 2); %Initialize the inverse Jacobian matrix
	% Calculate the elements of the Jacobian matrix
	J(1, 1) = -L1_model * sin(theta_D_inv(1)) - L2_model * sin(theta_D_inv(1) + theta_D_inv(2));
	J(1, 2) = -L2_model * sin(theta_D_inv(1) + theta_D_inv(2));
	J(2, 1) = L1_model * cos(theta_D_inv(1)) + L2_model * cos(theta_D_inv(1) + theta_D_inv(2));
	J(2, 2) = L2_model * cos(theta_D_inv(1) + theta_D_inv(2));
	det_J = J(1, 1) * J(2, 2) - J(1, 2) * J(2, 1);
	J_inv(1,1) = J(2, 2)/det_J;
	J_inv(1,2) = -J(1, 2)/det_J;
	J_inv(2,1) = -J(2, 1)/det_J;
	J_inv(2,2) = J(1, 1)/det_J;
end


function J_inv_dot = jacobian_inv_dot(theta_D_inv, theta_dotD_inv, J, det_J, L1_model, L2_model,t)
    J_inv_dot = zeros(2*length(t), 2); %Initialize the derivative of inverse Jacobian matrix
    dot_det_J = (-L1_model*cos(theta_D_inv(1))*theta_dotD_inv(1) + -L2_model*cos(theta_D_inv(1)+theta_D_inv(2))*(theta_dotD_inv(1)+theta_dotD_inv(2)))*(L2_model*cos(theta_D_inv(1)+theta_D_inv(2))) + (-L2_model*sin(theta_D_inv(1)+theta_D_inv(2))*(theta_dotD_inv(1)+theta_dotD_inv(2)))*(-L1_model*sin(theta_D_inv(1)) + -L2_model*sin(theta_D_inv(1)+theta_D_inv(2))) - ((-L2_model*cos(theta_D_inv(1)+theta_D_inv(2))*(theta_dotD_inv(1)+theta_dotD_inv(2)))*(L1_model*cos(theta_D_inv(1))+L2_model*cos(theta_D_inv(1)+theta_D_inv(2))) + (-L1_model*sin(theta_D_inv(1))*theta_dotD_inv(1) + -L2_model*sin(theta_D_inv(1)+theta_D_inv(2))*(theta_dotD_inv(1)+theta_dotD_inv(2)))*(-L2_model*sin(theta_D_inv(1)+theta_D_inv(2))));
    % Calculate the elements of the derivative of inverse jacobian matrix
    J_inv_dot(1,1) = (((-L2_model * sin(theta_D_inv(1) + theta_D_inv(2))*(theta_dotD_inv(1) + theta_dotD_inv(2)))*det_J)-dot_det_J*J(2, 2))/(det_J^2);
    J_inv_dot(1,2) = ((L2_model * cos(theta_D_inv(1) + theta_D_inv(2))*(theta_dotD_inv(1) + theta_dotD_inv(2))*det_J)+dot_det_J*J(1, 2))/(det_J^2);
    J_inv_dot(2,1) = ((L1_model * sin(theta_D_inv(1))*theta_dotD_inv(1) + L2_model * sin(theta_D_inv(1) + theta_D_inv(2))*(theta_dotD_inv(1) + theta_dotD_inv(2)))*det_J + dot_det_J*J(2, 1))/(det_J^2);
    J_inv_dot(2,2) = ((-L1_model * cos(theta_D_inv(1))*theta_dotD_inv(1)- L2_model * cos(theta_D_inv(1) + theta_D_inv(2))*(theta_dotD_inv(1) + theta_dotD_inv(2)))*det_J - dot_det_J*J(1, 1))/(det_J^2);
end


function [x_D_FOR, y_D_FOR, x_dot_D_FOR, y_dot_D_FOR, x_2dot_D_FOR, y_2dot_D_FOR] = FOR_KIN(theta1_D_inv, theta2_D_inv, theta1_dotD_inv, theta2_dotD_inv, theta1_2dotD_inv, theta2_2dotD_inv, L1_model, L2_model)
	x_D_FOR = L1_model * cos(theta1_D_inv) + L2_model * cos(theta1_D_inv + theta2_D_inv); % Desired position at x axis after forward kinematic conversion
	y_D_FOR = L1_model * sin(theta1_D_inv) + L2_model * sin(theta1_D_inv + theta2_D_inv); % Desired position at y axis after forward kinematic conversion
	x_dot_D_FOR = -L1_model .* sin(theta1_D_inv) .* theta1_dotD_inv -L2_model .* sin(theta1_D_inv + theta2_D_inv) .* (theta1_dotD_inv + theta2_dotD_inv); % Desired velocity at x axis after forward kinematic conversion
	y_dot_D_FOR = L1_model .* cos(theta1_D_inv) .* theta1_dotD_inv + L2_model .* cos(theta1_D_inv + theta2_D_inv) .* (theta1_dotD_inv + theta2_dotD_inv); % Desired velocity at y axis after forward kinematic conversion
	x_2dot_D_FOR = -L1_model .* cos(theta1_D_inv) .* (theta1_dotD_inv.^2) -L1_model .* sin(theta1_D_inv) .* theta1_2dotD_inv - L2_model .* cos(theta1_D_inv + theta2_D_inv) .* ((theta1_dotD_inv + theta2_dotD_inv).^2) -L2_model .* sin(theta1_D_inv + theta2_D_inv) .* (theta1_2dotD_inv + theta2_2dotD_inv); % Desired acceleration at x axis after forward kinematic conversion
	y_2dot_D_FOR = -L1_model .* sin(theta1_D_inv) .* (theta1_dotD_inv.^2) + L1_model .* cos(theta1_D_inv) .* theta1_2dotD_inv -L2_model .* sin(theta1_D_inv + theta2_D_inv) .* ((theta1_dotD_inv + theta2_dotD_inv).^2) + L2_model .* cos(theta1_D_inv + theta2_D_inv) .* (theta1_2dotD_inv + theta2_2dotD_inv); % Desired acceleration at y axis after forward kinematic conversion
	%x_2dot_D_FOR = -L1_model * cos(theta1_D_inv) * (theta1_dotD_inv.^2) -L1_model * sin(theta1_D_inv) * theta1_2dotD_inv - L2_model * cos(theta1_D_inv + theta2_D_inv) * ((theta1_dotD_inv + theta2_dotD_inv).^2) -L2_model * sin(theta1_D_inv + theta2_D_inv) * (theta1_2dotD_inv + theta2_2dotD_inv); % Desired acceleration at x axis after forward kinematic conversion
	%y_2dot_D_FOR = -L1_model * sin(theta1_D_inv) * (theta1_dotD_inv.^2) + L1_model * cos(theta1_D_inv) * theta1_2dotD_inv -L2_model * sin(theta1_D_inv + theta2_D_inv) * ((theta1_dotD_inv + theta2_dotD_inv).^2) + L2_model * cos(theta1_D_inv + theta2_D_inv) * (theta1_2dotD_inv + theta2_2dotD_inv); % Desired acceleration at y axis after forward kinematic conversion
end

function violation = constraint(obj, solution)
  % Solution is a row in PopDec
  PopDec = solution;
  M = obj.M;
  test=1;
  for i=1:M
    if i==1
        if test==0 % Original Adham test
          m_load = 0.7; %mass of the load at the end of the arm;
        elseif test==1 %Ami & Adham test
              m_load = 0;%0
        elseif test==2 %Ami & Adham test
            m_load = 0;
        elseif test==3 %Ami & Adham test
            m_load = 1;
        elseif test==4 % Eden test for very different trajectories
            m_load=0.5;%0
        elseif test==5 % Eden test for very different trajectories
            m_load=0.4;    
        end
    else
        if test==0
          m_load = 1; %mass of the load at the end of the arm
        elseif test==1
              m_load = 1;%1
        elseif test==2
            m_load = 2;
        elseif test==3
            m_load = 2;
        elseif test==4
            m_load=0.5;%0.5
        elseif test==5
            m_load=0 ;    
        end
    end
      m1_sam = 0.1; % Mass of link 1
      m2 = 0.2; % Mass of link 2
      L1_sam = 1; % Length of link 1 % Adham (changed to suit the working area)
      L2_sam = 0.5; % Length of link 2 % Adham (changed to suit the working area)
      c1_sam = 0.03; % Viscouse friction coefficient for joint 1
      c2_sam = 0.05; % Viscouse friction coefficient for joint 2
      power = 1;
      m2_sam = m2 + m_load; % Mass of link 2 including load at the end of the arm
      g = 9.81;
      tspan = 0:0.1:5;
      y0 = [-12.61*pi/180; 69.36*pi/180; 0; 0]; % Initial condition [theta1 theta2; theta1_dot theta2_dot]
      m1_model = 0.15;
      m2_model_no_load = 0.25;
      m_load_model = 0.6;%0.3
      m2_model = m2_model_no_load + m_load_model;
      L1_model = 1;
      L2_model = 0.5;
      c1_model = 0.04;
      c2_model = 0.06;
      % Define the network weights
      inputToHiddenWeights_Kv = PopDec(1,1:4);
      hiddenToOutputWeights_Kv = PopDec(1,5:8);
      inputToHiddenWeights_Kp = PopDec(1,17:20);
      hiddenToOutputWeights_Kp = PopDec(1,21:24);
      % Define the bias terms
      %hiddenBias_Kv = randn(hiddenSize, 2);
      %outputBias_Kv = randn(outputSize, 2);
      %hiddenBias_Kp = randn(hiddenSize, 2);
      %outputBias_Kp = randn(outputSize, 2);
      hiddenBias_Kv = PopDec(1,9:12);
      outputBias_Kv = PopDec(1,13:16);
      hiddenBias_Kp = PopDec(1,25:28);
      outputBias_Kp = PopDec(1,29:32);
      length_line = length(inputToHiddenWeights_Kv)/2;
      line1 = inputToHiddenWeights_Kv(1,1:length_line);
      a = length_line+1;
      l = length(inputToHiddenWeights_Kv);
      line2 = inputToHiddenWeights_Kv(1,a:l);
      inputToHiddenWeights_Kv = [line1; line2];

      length_line = length(hiddenToOutputWeights_Kv)/2;
      line1 = hiddenToOutputWeights_Kv(1,1:length_line);
      a = length_line+1;
      l = length(hiddenToOutputWeights_Kv);
      line2 = hiddenToOutputWeights_Kv(1,a:l);
      hiddenToOutputWeights_Kv = [line1; line2];

      length_line = length(inputToHiddenWeights_Kp)/2;
      line1 = inputToHiddenWeights_Kp(1,1:length_line);
      a = length_line+1;
      l = length(inputToHiddenWeights_Kp);
      line2 = inputToHiddenWeights_Kp(1,a:l);
      inputToHiddenWeights_Kp = [line1; line2];

      length_line = length(hiddenToOutputWeights_Kp)/2;
      line1 = hiddenToOutputWeights_Kp(1,1:length_line);
      a = length_line+1;
      l = length(hiddenToOutputWeights_Kp);
      line2 = hiddenToOutputWeights_Kp(1,a:l);
      hiddenToOutputWeights_Kp = [line1; line2];

      length_line = length(hiddenBias_Kv)/2;
      line1 = hiddenBias_Kv(1,1:length_line);
      a = length_line+1;
      l = length(hiddenBias_Kv);
      line2 = hiddenBias_Kv(1,a:l);
      hiddenBias_Kv = [line1; line2];
    
      length_line = length(outputBias_Kv)/2;
      line1 = outputBias_Kv(1,1:length_line);
      a = length_line+1;
      l = length(outputBias_Kv);
      line2 = outputBias_Kv(1,a:l);
      outputBias_Kv = [line1; line2];

      length_line = length(hiddenBias_Kp)/2;
      line1 = hiddenBias_Kp(1,1:length_line);
      a = length_line+1;
      l = length(hiddenBias_Kp);
      line2 = hiddenBias_Kp(1,a:l);
      hiddenBias_Kp = [line1; line2];

      length_line = length(outputBias_Kp)/2;
      line1 = outputBias_Kp(1,1:length_line);
      a = length_line+1;
      l = length(outputBias_Kp);
      line2 = outputBias_Kp(1,a:l);
      outputBias_Kp = [line1; line2];
      % Define network architecture
      inputSize = size(inputToHiddenWeights_Kv, 2);
      outputSize = size(hiddenToOutputWeights_Kv, 1);
      hiddenSize = size(hiddenBias_Kv, 1);
      % Input data
      inputData = randn(inputSize,hiddenSize);
      % Call the calcGains function to obtain K_v and K_p
      %[K_v, K_p] = calcGains(inputData, inputToHiddenWeights_Kv, hiddenToOutputWeights_Kv, hiddenBias_Kv, outputBias_Kv, inputToHiddenWeights_Kp, hiddenToOutputWeights_Kp, hiddenBias_Kp, outputBias_Kp);
      [t, y] = ode45(@(t, y) arm(t, y, m1_sam, m2_sam, m2, L1_sam, L2_sam, c1_sam, c2_sam, g, power, m1_model, m2_model, m2_model_no_load, L1_model, L2_model, c1_model, c2_model, inputToHiddenWeights_Kv, hiddenToOutputWeights_Kv, inputToHiddenWeights_Kp, hiddenToOutputWeights_Kp, hiddenBias_Kv, outputBias_Kv, hiddenBias_Kp, outputBias_Kp,m_load,0,0.1), tspan, y0);
      for j = 1:numel(t)
          [~,theta_2dot_actual(j,:),tau] = arm(t(j), y(j,:), m1_sam, m2_sam, m2, L1_sam, L2_sam, c1_sam, c2_sam, g, power, m1_model, m2_model, m2_model_no_load, L1_model, L2_model, c1_model, c2_model,inputToHiddenWeights_Kv, hiddenToOutputWeights_Kv, inputToHiddenWeights_Kp, hiddenToOutputWeights_Kp, hiddenBias_Kv, outputBias_Kv, hiddenBias_Kp, outputBias_Kp,m_load,0,0.1);
      end
      [x_t_D, Vx_t_D, Ax_t_D, y_t_D, Vy_t_D, Ay_t_D] = trajectory_org(t);
      [x_actual, y_actual, x_dot_actual, y_dot_actual, x_2dot_actual, y_2dot_actual] = FOR_KIN(y(:,1), y(:,2), y(:,3), y(:,4), theta_2dot_actual(:,1), theta_2dot_actual(:,2), L1_model, L2_model);   
      R_a = sqrt((x_actual.^2)+(y_actual.^2));
      R_d = sqrt((x_t_D.^2)+(y_t_D.^2));
      R_a_dot = ((x_actual.*x_dot_actual)+(y_actual.*y_dot_actual))/sqrt((x_actual.^2)+(y_actual.^2));
      R_d_dot = ((x_t_D.*Vx_t_D)+(y_t_D.*Vy_t_D))/sqrt((x_t_D.^2)+(y_t_D.^2));
      R_a_2dot = (((x_dot_actual.^2)+x_actual.*x_2dot_actual+(y_dot_actual.^2)+y_actual.*y_2dot_actual).*R_a - ((((x_actual.*x_dot_actual)+(y_actual.*y_dot_actual)).^2)./R_a))./(2*R_a);
      R_d_2dot = (((Vx_t_D.^2)+Vx_t_D.*Ax_t_D+(Vy_t_D.^2)+y_t_D.*Ay_t_D).*R_d - ((((x_t_D.*Vx_t_D)+(y_t_D.*Vy_t_D)).^2)./R_d))./(2*R_d);
     constraint1_curr = ones(1,length(R_a));
     g1_curr=ones(1,length(R_a));
     for j = 1:length(R_a)
        g1 = abs(R_a(j) - R_d(j));
        g1_curr(j)=g1;
        constraint1_curr(j) = min(1,(g1<=0.2));
     end
     g1_tot=max(g1_curr);

     constraint2_curr = ones(1,length(R_a_dot));
     g2_curr=ones(1,length(R_a));
     for j = 1:length(R_a_dot)
        g2 = abs(R_a_dot(j) - R_d_dot(j));
        g2_curr(j)=g2;
        constraint2_curr(j) = min(1,(g2<=0.8));
     end
     g2_tot=max(g2_curr);

     constraint3_curr = ones(1,length(R_a_2dot));
     g3_curr=ones(1,length(R_a));
     for j = 1:length(R_a_2dot)
        g3 = abs(R_a_2dot(j) - R_d_2dot(j));
        g3_curr(j)=g3;
        constraint3_curr(j) = min(1,(g3<=0.8));
     end
     g3_tot=max(g3_curr);
      
     constraint1 = min(constraint1_curr);
     constraint2 = min(constraint2_curr);
     constraint3 = min(constraint3_curr);
  
     % constraint2_curr = [1,1];
     % g2_curr = [1,1];
     % for j = 1:length(tau)
     %     g2 = tau(j);
     %     g2_curr(j) = g2;
     %     %g2 = tau;
     %     %constraint2 = min(constraint2,(tau(j) <= 4.5*(10^11)));
     %     constraint2_curr(j) = min(1,(-5000<=g2 && g2 <= 20000));
     % end
     % g2_tot=max(g2_curr);
      
     %constraint2 = min(constraint2_curr);

    
    g4 = tau(1);
    %constraint2 = min(constraint2,(tau(j) <= 4.5*(10^11)));
    constraint4_curr = min(1,(-5000<=g4 && g4 <= 20000));
    g4_tot=g4;
    constraint4 = constraint4_curr;

    
    g5 = tau(2);
    %constraint2 = min(constraint2,(tau(j) <= 4.5*(10^11)));
    constraint5_curr = min(1,(-5000<=g5 && g5 <= 20000));
    g5_tot=g5;
    constraint5 = constraint5_curr;

     
     % constraint_p = min(constraint1,constraint2);
     % constraint_h = min(constraint3,constraint4);
     % constraint_i = min(constraint_h,constraint5);
     % constraint = min(constraint_p,constraint_i);
     %if i == 1
         %if constraint == 0
     if constraint1 == 0
       %if g1<=0
           %viol1 = abs(g4/4);
           viol1 = max(g1_tot,[],2);
       %else 
        %   viol1 = g1;
       %end
     %elseif constraint == 1
     elseif constraint1 == 1
       %if g1<=0
           %viol1 = g1;
           viol1 = 0;
       %else 
        %   viol1 = -g1;
       %end
     end
     %end
    
     %if i == 2
         %if constraint == 0
     if constraint2 == 0
       %if g1<=0
           %viol2 = abs(g1/2);
           %viol2 = abs(g4/2);
           viol2 =max(g2_tot,[],2);
           
       %else 
           %viol2 = g1/2;
           %viol2 = g1;
       %end
     elseif constraint2 == 1
       %if g1<=0
           %viol2 = g1/2;
           %viol2 = g1;
           viol2 = 0;
       %else 
           %viol2 = -g1/2;
           %viol2 = -g1;
       %end
     end
     %end
     
     if constraint3 == 0
     
           viol3 = max(g3_tot,[],2);
       
     elseif constraint3 == 1
  
           viol3 = 0;
      
     end

     if constraint4 == 0
     
           viol4 = max(g4_tot,[],2);
       
     elseif constraint4 == 1
  
           viol4 = 0;
      
     end

     if constraint5 == 0
     
           viol5 = max(g5_tot,[],2);
       
     elseif constraint5 == 1
  
           viol5 = 0;
      
     end



  end
  
  
 
  
  % if viol1<=0 && abs(viol1) >= 0.01
  %     viol2 = viol2 - 3;
  % end
  % if viol2<=0 && abs(viol2) >= 0.01
  %     viol2 = viol2 - 3;
  % end
  
  %violation = [max(constraint1, [], 2), constraint2];
  violation = [viol1, viol2, viol3, viol4,viol5];
  %disp(violation);
end

function var = Str2Fcn(var,type,useData,name,D)
% Convert a string into a variable or function and check its validity

    if ischar(var)
        try
            if ~isempty(regexp(var,'^<.+>$','once'))
                switch type
                    case 1      % For lower, upper, data
                        var = load(var(2:end-1));
                    otherwise   % For initFcn, evalFcn, decFcn, objFcn, conFcn, objGradFcn, objConFcn
                        [folder,file,ext] = fileparts(var(2:end-1));
                        if type ~= 4 || strcmp(ext,'.m')
                            addpath(folder);
                            var = str2func(file);
                        else
                            var = load(var(2:end-1));
                        end
                end
            else
                switch type
                    case 1      % For lower, upper, data
                        var = str2num(var);
                    case 2      % For initFcn
                        if useData
                            var = str2func(['@(N,data)',var]);
                        else
                            var = str2func(['@(N)',var]);
                        end
                    case {3,4}	% For evalFcn, decFcn, objFcn, conFcn, objGradFcn, objConFcn
                        if useData
                            var = str2func(['@(x,data)',var]);
                        else
                            var = str2func(['@(x)',var]);
                        end
                end
            end
        catch err
            err = addCause(err,MException('','Fail to define the %s',name));
            rethrow(err);
        end
    end
    if type == 1 && nargin > 4      % For lower, upper, data
        if isscalar(var)
            var = repmat(var,1,D);
        else
            assert(ismatrix(var)&&all(size(var)==[1,D]),'the %s should be a scalar or a 1*%d vector, while its current size is %d*%d.',name,D,size(var,1),size(var,2));
        end
    end
    if type == 4 && isnumeric(var)   % For objFcn, conFcn
        try
            fprintf('Fit the %s...\n',name);
            Model = fitrgp(var(:,1:end-1),var(:,end),'OptimizeHyperparameters','all','HyperparameterOptimizationOptions',struct('ShowPlots',false,'Verbose',0));
            var   = @(x)predict(Model,x);
        catch err
            err = addCause(err,MException('','Fail to fit the %s',name));
            rethrow(err);
        end
    end
end

function Var = Strs2Fcns(Var,type,useData,name)
% Convert multiple strings into functions

    if ~iscell(Var)
        Var = {Var};
    end
    Var(cellfun(@isempty,Var)) = [];
    for i = 1 : length(Var)
        Var{i} = Str2Fcn(Var{i},type,useData,[name,num2str(i)]);
    end
end

function varargout = CallFcn(func,input,data,name,varargin)
% Call a function and check the validity of its output

    try
        if isempty(data)
            [varargout{1:nargout}] = func(input);
        else
            [varargout{1:nargout}] = func(input,data);
        end
        for i = 1 : min(length(varargout),length(varargin))
            assert(ismatrix(varargout{i})&&all(size(varargout{i})==varargin{i}),'the size of its output #%d should be %d*%d, while its current size is %d*%d.',i,varargin{i}(1),varargin{i}(2),size(varargout{i},1),size(varargout{i},2));
        end
    catch err
        err = addCause(err,MException('','The %s is invalid',name));
        rethrow(err);
    end
end