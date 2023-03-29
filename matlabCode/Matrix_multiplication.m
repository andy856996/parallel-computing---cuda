clc;clear all;

a = rand(3,3);
b = rand(3,5);

% a = ones(3,3);
% b = ones(3,5);

disp(a*b);

c = zeros(3,5);
for i = 1:size(a,1)
    for j = 1:size(b,2)
        for k = 1:size(a,2)
            c(i,j) = c(i,j) + a(i,k)*b(k,j);
        end
    end
end
disp(c)

disp(['equal:' num2str(isequal(a*b,c))])