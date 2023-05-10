a = [0 0 0 0 0 0 0 0 0;
     0 0 0 0 0 0 0 0 0;
     0 0 1 1 1 1 1 0 0;
     0 0 1 1 1 1 1 0 0;
     0 0 1 1 1 1 1 0 0;
     0 0 1 1 1 1 1 0 0;
     0 0 1 1 1 1 1 0 0;
     0 0 0 0 0 0 0 0 0;
     0 0 0 0 0 0 0 0 0];
b = [0 0 0 0 0 0 0 0 0;
     0 0 0 0 0 0 0 0 0;
     0 0 1 1 1 1 1 0 0;
     0 0 1 1 1 1 1 0 0;
     0 0 1 1 1 1 1 0 0;
     0 0 1 1 1 1 1 0 0;
     0 0 1 1 1 1 1 0 0;
     0 0 0 0 0 0 0 0 0;
     0 0 0 0 0 0 0 0 0];
conv2(a,b,"same")
conv2_for(a, b)

function P = conv2_for(mask, kernal)
[m_R,m_C] = size(mask);
[k_R,k_C] = size(kernal);
P = zeros(m_R,m_C);
kCenterX =  fix(k_C/ 2);
kCenterY = fix(k_R / 2);
for i = 1:m_R
    for j = 1:m_C
        for m = 1:k_R
            for n = 1:k_C
                ii = i + (m - kCenterY);
                jj = j + (n - kCenterX);
                if ii > 0 && ii <= m_R && jj > 0 && jj <= m_C
                    P(i,j) = P(i,j) + mask(ii,jj) * kernal(m,n);
                end
            end
        end
    end
end
end