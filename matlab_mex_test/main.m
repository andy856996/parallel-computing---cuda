mexfunction(ones(10,10),1,1,1,1,1)
s = 5; 
A = [1.5, 2, 9;1 1 1];

tic
a = 11;
a =ones(9,8);
b = ones(7,6);
conv2(a,b,"same")
B = mexfunction_test(a,b)
toc
