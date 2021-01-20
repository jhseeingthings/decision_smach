% The function npg solves an n-person finite non-co-operative game to
% compute one sample Nash Equilibrium. It uses an optimization formulation
% of n-person non-co-operative games as described in the adjoining paper
% "An Optimization Formulation to Compute Nash Equilibrium in finite
% Games" presented by the author.
%
% The inputs to the function are:
% a) M : The row vector containing number of strategies of each player.
% b) U : The matrix containing the pay-offs to the players at various
%        pure strategy combinations.
%
% The outputs are:
% a) A : The matrix whose columns are mixed strategies to players at Nash
%        equilibrium.
% b) payoff : The row vector containing the pay-offs to the players at Nash
%        equilibrium.
% c) iterations : Number of iterations performed by the optimization
%        method.
% d) err : The absolute error in the objective function of the minimization
%        problem.
%
% For theory of the method the adjoining paper may be consulted. Here an
% example is given for explanantion. Consider a 3-person game where each
% player has 2 strategies each. So M = [2 2 2]. Suppose the pay-offs at the
% pure strategy combinations (1-1-1), (1-1-2) and so on, as described by
% the ranking in the theory, are given as the matrix U =
% [2,7.5,0;3,.2,.3;6,3.6,1.5;2,3.5,5;0,3.2,9;2,3.2,5;0,2,3.2;2.1,0,1]. Then
% after supplying M and U call [A,payoff,iterations,err] = npg(M,U).
%
% The method is capable of giving one sample Nash equilibrium out of
% probably many present in a given game. The screenshot showing GUI has 
% been developed on the code using it as dll and VB.Net. The GUI software 
% may be made available on request.
%
% Any error in the code may be reported to bhaskerchatterjee@gmail.com. Any
% suggestion/comment is greatly appreciated.

function [A] = npg(M,U)

n = 2;
Us = sum(U,2);
p = 6;
I = [1,4;1,5;2,4;2,5;3,4;3,5];
s = 5;
ub = [1;1;1;1;1;Inf;Inf];
lb = [0;0;0;0;0;-Inf;-Inf];
x0 = [0.333333;0.333333;0.333333;0.5;0.5;0.1388889;0.0925926];
Aeq = [1,1,1,0,0,0,0;0,0,0,1,1,0,0];
beq = [1;1];
pay = [6;6;6;7;7];


[x] = gamer(n,Us,p,I,s,ub,lb,x0,Aeq,beq,pay,U);

count = 0;
% M = [3,2];

for i = 1 : n
    for j = 1 : M(i)
        count = count + 1;
        A(j,i) = abs(x(count));
    end
end

%iterations = output.iterations;
%err = abs(fval);