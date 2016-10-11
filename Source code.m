% Part 1
% import the data
[train, tune, test]=getdata('wdbc.data',30);
% get the training set and seperate them into M and B
M=train(train(:,1)==77,[2:31]);
B=train(train(:,1)==66,[2:31]);
[m,n]=size(M);
[kb,n]=size(B);
miu=0.0001;
% forming the quadratic form to solve:
Q=[miu*eye(30) zeros(30, 370); zeros(370,400)];
c=[zeros(30,1); 1/m*ones(m,1); 1/kb*ones(kb,1); 0];
A=[-M -eye(m) zeros(m,kb) ones(m,1); 
    B zeros(kb,m) -eye(kb,kb) -ones(kb,1)];
b=[-ones(m+kb,1)];
lb=[-inf*ones(30,1); zeros(m+kb,1); -inf];
ub=[inf*ones(400,1)];
[x,obj]=cplexqp(Q,c,A,b,[],[],lb,ub);
w=x(1:30);
y=x(31:(30+m));
z=x((31+m):(30+m+kb));
gamma=x(400);
% the w is:
w
% the gamma is:
gamma
% the minimum value of the QP is:
obj

% Part 2
MTune=tune(tune(:,1)==77, [2:31]);
BTune=tune(tune(:,1)==66, [2:31]);
[mTune,nTune]=size(MTune);
[kbTune,nTune]=size(BTune);
mm=MTune*w-ones(mTune,1)*gamma;
mb=BTune*w-ones(kbTune,1)*gamma;
MisM=mm(mm<=0);
MisB=mb(mb>0);
[rowM, colM]=size(MisM);
[rowB, colB]=size(MisB);
rowB+rowM
% We can easily see that there are 3 points misclassified in total.
misclassifiedNum=[];
testingSetErr=[];
for miu = 0.00005:0.00005:0.0005;
    Q=[miu*eye(30) zeros(30, 370); zeros(370,400)];
    [x,obj]=cplexqp(Q,c,A,b,[],[],lb,ub);
    w=x(1:30);
    y=x(31:(30+m));
    z=x((31+m):(30+m+kb));
    gamma=x(400);
    MTune=tune(tune(:,1)==77, [2:31]);
    BTune=tune(tune(:,1)==66, [2:31]);
    [mTune,nTune]=size(MTune);
    [kbTune,nTune]=size(BTune);
    mm=MTune*w-ones(mTune,1)*gamma;
    mb=BTune*w-ones(kbTune,1)*gamma;
    % finding the misclassified points:
    MisM=mm(mm<=0);
    MisB=mb(mb>0);
    [rowM, colM]=size(MisM);
    [rowB, colB]=size(MisB);
    misclassifiedNum(end+1)=rowB+rowM;
    testingSetErr(end+1)=ones(1,m)*y+ones(1,kb)*z;
end
misclassifiedNum
testingSetErr
% in the above two vectors, we see that number of misclassified points
% remains the same for several values of mu, but the values of testing set 
% error keep increasing as the value of miu increases. Thus, I choose the 
% miu value as 0.00015 for fewer misclassified points and less testing set
% error. 
% Applying this value of miu to testing set:
miu=0.00015;
Q=[miu*eye(30) zeros(30, 370); zeros(370,400)];
[x,obj]=cplexqp(Q,c,A,b,[],[],lb,ub);
w=x(1:30);
y=x(31:(30+m));
z=x((31+m):(30+m+kb));
gamma=x(400);
MTest=test(test(:,1)==77, [2:31]);
BTest=test(test(:,1)==66, [2:31]);
[mTest,nTest]=size(MTest);
[kbTest,nTest]=size(BTest);
mm=MTest*w-ones(mTest,1)*gamma;
mb=BTest*w-ones(kbTest,1)*gamma;
% finding the misclassified points:
MisM=mm(mm<=0);
MisB=mb(mb>0);
[rowM, colM]=size(MisM);
[rowB, colB]=size(MisB);
TeSeErr=ones(1,m)*y+ones(1,kb)*z
NumMisClaPt=rowB+rowM
% the number of misclassified points is 7 and the testing set error is
% 6.4582.

% Part 3
twoAtts=combnk(1:30,2);
[np, col]=size(twoAtts);
MisCN=[];
TSE=[];
for i=1:1:np;
    Q3=[miu*eye(2) zeros(2, 370); zeros(370,372)];
    c3=[zeros(2,1); 1/m*ones(m,1); 1/kb*ones(kb,1); 0];
    A3=[-M(:,twoAtts(i,:)) -eye(m) zeros(m,kb) ones(m,1);
        B(:,twoAtts(i,:)) zeros(kb,m) -eye(kb,kb) -ones(kb,1)];
    b3=[-ones(m+kb,1)];
    lb3=[-inf*ones(2,1); zeros(m+kb,1); -inf];
    ub3=[inf*ones(372,1)];
    [x3,obj3]=cplexqp(Q3,c3,A3,b3,[],[],lb3,ub3);
    w3=x3(1:2);
    y3=x3(3:(2+m));
    z3=x3((3+m):(2+m+kb));
    gamma3=x3(372);
    MTune=tune(tune(:,1)==77, [2:31]);
    MTune3=MTune(:,twoAtts(i,:));
    BTune=tune(tune(:,1)==66, [2:31]);
    BTune3=BTune(:,twoAtts(i,:));
    [mTune,nTune3]=size(MTune3);
    [kbTune,nTune3]=size(BTune3);
    mm=MTune3*w3-ones(mTune,1)*gamma3;
    mb=BTune3*w3-ones(kbTune,1)*gamma3;
    % finding the misclassified points:
    MisM=mm(mm<=0);
    MisB=mb(mb>0);
    [rowM, colM]=size(MisM);
    [rowB, colB]=size(MisB);
    wrong=rowB+rowM;
    MisCN(end+1)=wrong;
    TSE(end+1)=ones(1,m)*y3+ones(1,kb)*z3;
    fprintf('atts %2d %2d: misclass %3d\n',twoAtts(i,1),twoAtts(i,2),wrong);
end
% the indexes of the pair of attributes that has the least number of
% misclassified points and corresponding tuning set error is:
index=find(MisCN==min(MisCN))
TSE(index)
% In the vectors above, we see that for index 188 and 415, we get the least
% number of misclassified points, and the tuning error for index 415 is
% significantly less. So I choose the index 415 as the best performing pair
% of attribute. 
twoAtts(415,:)
% That pair is attributes 24 and 25.

% Part 4
Q4=[miu*eye(2) zeros(2, 370); zeros(370,372)];
c4=[zeros(2,1); 1/m*ones(m,1); 1/kb*ones(kb,1); 0];
A4=[-M(:,twoAtts(415,:)) -eye(m) zeros(m,kb) ones(m,1);
    B(:,twoAtts(415,:)) zeros(kb,m) -eye(kb,kb) -ones(kb,1)];
b4=[-ones(m+kb,1)];
lb4=[-inf*ones(2,1); zeros(m+kb,1); -inf];
ub4=[inf*ones(372,1)];
[x4,obj4]=cplexqp(Q4,c4,A4,b4,[],[],lb4,ub4);
w4=x4(1:2);
gamma4=x4(372);
MTest=test(test(:,1)==77, [2:31]);
BTest=test(test(:,1)==66, [2:31]);
[mTest,nTest]=size(MTest);
[kbTest,nTest]=size(BTest);
mm4=MTest(:,twoAtts(415,:))*w4-ones(mTest,1)*gamma4;
mb4=BTest(:,twoAtts(415,:))*w4-ones(kbTest,1)*gamma4;
% finding the misclassified points:
MisM4=mm4(mm4<=0);
MisB4=mb4(mb4>0);
[rowM4, colM4]=size(MisM4);
[rowB4, colB4]=size(MisB4);
% number of misclassified points is:
rowB4+rowM4
hold on
axis([-1.5 4 -3 6])
plot(MTest(:,twoAtts(415,1)), MTest(:,twoAtts(415,2)), '+')
plot(BTest(:,twoAtts(415,1)), BTest(:,twoAtts(415,2)), 'o')
line(x, (gamma4-w4(1)*x)/w4(2))
hold off
% As I counted, the number of misclassified points agree with the plot.
% Most misclassified points are in set B and one misclassified point is in
% set M. The line using coefficients derived from tuning set performs well
% in the testing set. Thus, tuning set is a good representation for the
% testing set. 