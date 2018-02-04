% 将自己设计的采样和特征提取算法整合成一个函数以便于其他程序调用
% Tree-shaped sampling based LBP（简称TSLBP）
% 创建日期：2018年1月25日
% 作者： 马冬冬

function casd_TTC_SMCH = TSSHM(image,P1,P2,mapping1,mapping2,eight_spoins,R,r)

rectify = 0;
casd_TTC_SMCH = [];

imgExt = padarray(image,[1 1],'symmetric','both');      % 对称填补图像边缘,这里的补边[1 1]和下面的窗口大小[3 3]是确定的
imgblks = im2col(imgExt,[3 3],'sliding');               % 以3*3的块求取中值，而且是sliding滑动形式，imgblks有9行
a = mean(imgblks);
fimg = reshape(a,size(image));                          % 中值滤波后重新形成原图像大小的图像
%          fimg = image;

% 首先求取内圆和外圆的特征
for r_num = 1:2
    % 下面进行采样点数据计算以及特征提取，特征有S,M,C,D，S表示符号，M表示幅度，C表示灰度差，D表示邻域点与中心点的差
    % 求取每个点的中值，以3*3窗口

    [ysize,xsize] = size(fimg);
    spoints = eight_spoins{1,r_num};
    miny=min(spoints(:,1));
    maxy=max(spoints(:,1));
    minx=min(spoints(:,2));
    maxx=max(spoints(:,2));
    bsizey=ceil(max(maxy,0))-floor(min(miny,0))+1;
    bsizex=ceil(max(maxx,0))-floor(min(minx,0))+1;

    origy = 1-floor(min(miny,0));
    origx = 1-floor(min(minx,0));
    dx = xsize - bsizex;
    dy = ysize - bsizey;
    d_C = fimg(origy:origy+dy,origx:origx+dx);

    TTC_S=zeros(dy+1,dx+1);
    TTC_M=zeros(dy+1,dx+1);
    neighbor_cube = zeros(dy+1,dx+1,P1);
    Diff = zeros(dy+1,dx+1,P1);
    MeanDiff = zeros(1,P1);
    D_cube = zeros(size(neighbor_cube));
    for j = 1:P1
        y = spoints(j,1)+origy;
        x = spoints(j,2)+origx;
        fy = floor(y); cy = ceil(y); ry = round(y);
        fx = floor(x); cx = ceil(x); rx = round(x);
        if (abs(x - rx) < 1e-6) && (abs(y - ry) < 1e-6)
           N = fimg(ry:ry+dy,rx:rx+dx);
        else
           ty = y - fy;
           tx = x - fx;
           w1 = (1 - tx) * (1 - ty);
           w2 =      tx  * (1 - ty);
           w3 = (1 - tx) *      ty ;
           w4 =      tx  *      ty ;
           N = w1*fimg(fy:fy+dy,fx:fx+dx) + w2*fimg(fy:fy+dy,cx:cx+dx) + ...
               w3*fimg(cy:cy+dy,fx:fx+dx) + w4*fimg(cy:cy+dy,cx:cx+dx);
        end  
        D = N >= d_C;     
        Diff(:,:,j) = abs(N - d_C);    
        MeanDiff(j) = mean(mean(Diff(:,:,j)));
        v = 2^(j-1);
        TTC_S = TTC_S + v*D;

        neighbor_cube(:,:,j) = N;
        D_cube(:,:,j) = D;
    end

    %********************** 下面进行自适应阈值及修正 *******************
     if rectify == 1
        phi_t = zeros(size(neighbor_cube));
        phi_t(:,:,1) = (neighbor_cube(:,:,1)+neighbor_cube(:,:,P1))/2;
        for k = 2:P1
            phi_t(:,:,k) = (neighbor_cube(:,:,k)+neighbor_cube(:,:,k-1))/2;
        end
        % 下面判断是否是均匀模式，如果是非均匀模式再考虑应该恢复到均匀模式，自适应阈值
        phi_result = zeros(dy+1,dx+1,8);     % P is odd, so x-(P-1)/2=x-P+1=dy+1
        for k = 1:P1
            new_Centre = phi_t(:,:,k);
            for j = 1:P1
                D = neighbor_cube(:,:,j) >= new_Centre;           % 在LBP算法中，C是确定的，DLABP算法需要自适应地选择阈值C 
                v = 2^(j-1);
                phi_result(:,:,k) = phi_result(:,:,k) + v*D;        
            end
        end
        bins = mapping1.num;
        if isstruct(mapping1)
            for k = 1:size(TTC_S,1)
                for j = 1:size(TTC_S,2)
                    TTC_S(k,j) = mapping1.table(TTC_S(k,j)+1);  % result元素范围：0-9
                    if (TTC_S(k,j) == bins-1)        % 如果是非均匀模式，则用8个gm重新阈值化，在得到映射值
                        temp1 = squeeze(phi_result(k,j,1:8));
                        temp2 = squeeze(phi_t(k,j,1:8));
                        corrupted_num = mapping1.table(temp1 + 1) < bins-1;
                        temp3 = find(corrupted_num);
                        if sum(corrupted_num) > 1     % 如果可恢复那么就寻找最小差距新中心
                           new_old_Cdist = abs(temp2(corrupted_num)-d_C(k,j));
                           [~,min_ind] = min(new_old_Cdist);
                           min_Cdist_ind = temp3(min_ind);
                           TTC_S(k,j) = mapping1.table(temp1(min_Cdist_ind) + 1);
%                            modified_C(k,j) = temp2(min_Cdist_ind);
                        end
                    else
                         continue;         % 如果不是均匀模式，则继续
                    end
                end
            end
        end 
        %***********************自适应阈值及修正结束********************
    elseif rectify == 2           % 如果是2则采用自己设计的矫正方法
           % 下面判断是否是均匀模式，如果是非均匀模式再考虑应该恢复到均匀模式，自适应阈值
           bins = mapping1.num;
           if isstruct(mapping1)
               for k = 1:size(TTC_S,1)
                   for j = 1:size(TTC_S,2)
                       temp = mapping1.table(TTC_S(k,j)+1);  % result元素范围：0-9
                       if (temp == bins-1)
                           % 首先统计0,1跳变次数
                           bit_seq = squeeze(D_cube(k,j,:));
                           rot_bit_seq = bitset(bitshift(TTC_S(k,j),1,astype),1,bitget(TTC_S(k,j),P1)); 
                           numt = sum(bitget(bitxor(TTC_S(k,j),rot_bit_seq),1:P1));
                           isrect = 0;
                           if numt == 4
                              [isrect,rectf_value] = check_rectify(bit_seq,squeeze(neighbor_cube(k,j,:)),d_C(k,j)); 
                           end
                           if (isrect == 1)                  % 如果是非均匀模式
                               % 如果是跳变次数等于4并且有孤立杂点存在并且孤立点与中心处差值在阈值范围内，那么就进行修正
                               TTC_S(k,j) = mapping1.table(rectf_value+1);
                               continue;
                           end
                       end
                       TTC_S(k,j) = temp;
                   end
               end
           end    
    else
        sizarray = size(TTC_S);
        TTC_S = TTC_S(:);
        TTC_S = mapping1.table(TTC_S+1);
        TTC_S = reshape(TTC_S,sizarray);
    end

    % 计算TTC_S和TTC_M
    DiffThreshold = mean(MeanDiff);
    for j = 1:P1
        v = 2^(j-1);
%             TTC_M = TTC_M + v*(Diff(:,:,j) >= DiffThreshold);
        TTC_M = TTC_M + v*(Diff(:,:,j) >= MeanDiff(j));         % 1月14日下午，改为局部阈值试一下效果，结果是：
%             TTC_M = TTC_M + v*(Diff(:,:,j) >= mean(Diff(:,:,j),3)); 
    end
    % 计算TTC_C
    TTC_C = d_C >= mean(fimg(:));
    % 进行映射
    sizarray = size(TTC_M);
    TTC_M = TTC_M(:);
    TTC_M = mapping1.table(TTC_M+1);
    TTC_M = reshape(TTC_M,sizarray);
    % 特征直方图统计
    TTC_MCSum = TTC_M;
    idx = find(TTC_C);
    TTC_MCSum(idx) = TTC_MCSum(idx)+mapping1.num;
    TTC_SMC = [TTC_S(:),TTC_MCSum(:)];
    Hist3D = hist3(TTC_SMC,[mapping1.num,mapping1.num*2]);
    TTC_SMCH = reshape(Hist3D,1,numel(Hist3D));      
%         TTC_SMCH(i,:) = hist(TTC_S(:),0:mapping.num-1);
    casd_TTC_SMCH = [casd_TTC_SMCH,TTC_SMCH];
end


% 下面求取joint符号特征
% 下面是采用中间不规则的多边形作为阈值对内圆和外圆进行阈值化，然后得到8位二进制的数，最后再进行joint联合[delta_i,delta_j]
middle_pgon = [0,R;-(R+r)/2,(R+r)/2;-R,0;-(R+r)/2,-(R+r)/2;0,-R;(R+r)/2,-(R+r)/2;R,0;(R+r)/2,(R+r)/2];
inner_circ = [0,R-r;-(R-r)*sqrt(2)/2,(R-r)*sqrt(2)/2;-(R-r),0;-(R-r)*sqrt(2)/2,-(R-r)*sqrt(2)/2;0,-(R-r);(R-r)*sqrt(2)/2,-(R-r)*sqrt(2)/2;R-r,0;(R-r)*sqrt(2)/2,(R-r)*sqrt(2)/2];
outer_circ = [0,R+r;-(R+r)*sqrt(2)/2,(R+r)*sqrt(2)/2;-(R+r),0;-(R+r)*sqrt(2)/2,-(R+r)*sqrt(2)/2;0,-(R+r);(R+r)*sqrt(2)/2,-(R+r)*sqrt(2)/2;R+r,0;(R+r)*sqrt(2)/2,(R+r)*sqrt(2)/2];

[ysize,xsize] = size(fimg);
miny = -(R+r);
maxy = R+r;
minx = -(R+r);
maxx = R+r;
%     spoints = inner_circ;
%     miny=min(spoints(:,1));
%     maxy=max(spoints(:,1));
%     minx=min(spoints(:,2));
%     maxx=max(spoints(:,2));

bsizey=ceil(max(maxy,0))-floor(min(miny,0))+1;
bsizex=ceil(max(maxx,0))-floor(min(minx,0))+1;

origy = 1-floor(min(miny,0));
origx = 1-floor(min(minx,0));
dx = xsize - bsizex;
dy = ysize - bsizey;
d_C = fimg(origy:origy+dy,origx:origx+dx);

decimal_inner = zeros(dy+1,dx+1);
decimal_outer = zeros(dy+1,dx+1);
Diff2 = zeros(dy+1,dx+1,P1);
MeanDiff2 = zeros(1,P1);
Diff3 = zeros(dy+1,dx+1,P1);
MeanDiff3 = zeros(1,P1);
for j = 1:8
    y1 = middle_pgon(j,1)+origy; y2 = inner_circ(j,1)+origy; y3 = outer_circ(j,1)+origy;
    x1 = middle_pgon(j,2)+origx; x2 = inner_circ(j,2)+origx; x3 = outer_circ(j,2)+origx;
    fy1 = floor(y1); cy1 = ceil(y1); ry1 = round(y1);
    fx1 = floor(x1); cx1 = ceil(x1); rx1 = round(x1);
    fy2 = floor(y2); cy2 = ceil(y2); ry2 = round(y2);
    fx2 = floor(x2); cx2 = ceil(x2); rx2 = round(x2);
    fy3 = floor(y3); cy3 = ceil(y3); ry3 = round(y3);
    fx3 = floor(x3); cx3 = ceil(x3); rx3 = round(x3);
%         if (abs(x1 - rx1) < 1e-6) && (abs(y1 - ry1) < 1e-6)
%            N1 = fimg(ry1:ry1+dy,rx1:rx1+dx);
%         else
%            ty1 = y1 - fy1;
%            tx1 = x1 - fx1;
%            w1 = (1 - tx1) * (1 - ty1);
%            w2 =      tx1  * (1 - ty1);
%            w3 = (1 - tx1) *      ty1 ;
%            w4 =      tx1  *      ty1 ;
%            N1 = w1*fimg(fy1:fy1+dy,fx1:fx1+dx) + w2*fimg(fy1:fy1+dy,cx1:cx1+dx) + ...
%                 w3*fimg(cy1:cy1+dy,fx1:fx1+dx) + w4*fimg(cy1:cy1+dy,cx1:cx1+dx);
%         end
    if (abs(x2 - rx2) < 1e-6) && (abs(y2 - ry2) < 1e-6)
       N2 = fimg(ry2:ry2+dy,rx2:rx2+dx);
    else
       ty2 = y2 - fy2;
       tx2 = x2 - fx2;
       w1 = (1 - tx2) * (1 - ty2);
       w2 =      tx2  * (1 - ty2);
       w3 = (1 - tx2) *      ty2 ;
       w4 =      tx2  *      ty2 ;
       N2 = w1*fimg(fy2:fy2+dy,fx2:fx2+dx) + w2*fimg(fy2:fy2+dy,cx2:cx2+dx) + ...
            w3*fimg(cy2:cy2+dy,fx2:fx2+dx) + w4*fimg(cy2:cy2+dy,cx2:cx2+dx);
    end
    if (abs(x3 - rx3) < 1e-6) && (abs(y3 - ry3) < 1e-6)
       N3 = fimg(ry3:ry3+dy,rx3:rx3+dx);
    else
       ty3 = y3 - fy3;
       tx3 = x3 - fx3;
       w1 = (1 - tx3) * (1 - ty3);
       w2 =      tx3  * (1 - ty3);
       w3 = (1 - tx3) *      ty3 ;
       w4 =      tx3  *      ty3 ;
       N3 = w1*fimg(fy3:fy3+dy,fx3:fx3+dx) + w2*fimg(fy3:fy3+dy,cx3:cx3+dx) + ...
            w3*fimg(cy3:cy3+dy,fx3:fx3+dx) + w4*fimg(cy3:cy3+dy,cx3:cx3+dx);
    end       

    temp = N2 >= d_C;
    decimal_inner = decimal_inner + temp*2^(j-1);   
    temp = N3 >= d_C;
    decimal_outer = decimal_outer + temp*2^(j-1);  

    Diff2(:,:,j) = abs(N2 - d_C);    
    MeanDiff2(j) = mean(mean(Diff2(:,:,j)));
    Diff3(:,:,j) = abs(N3 - d_C);    
    MeanDiff3(j) = mean(mean(Diff3(:,:,j)));
end

sizarray = size(decimal_inner);
decimal_inner = decimal_inner(:);
decimal_inner = mapping1.table(decimal_inner+1);
decimal_inner = reshape(decimal_inner,sizarray);    
%     inner_feat = hist(decimal_inner(:),0:mapping.num-1);

sizarray = size(decimal_outer);
decimal_outer = decimal_outer(:);
decimal_outer = mapping1.table(decimal_outer+1);
decimal_outer = reshape(decimal_outer,sizarray);    
%     outer_feat = hist(decimal_outer(:),0:mapping.num-1);

temp = [decimal_inner(:),decimal_outer(:)];
Hist3D = hist3(temp,[mapping1.num,mapping1.num]);
s_feat = reshape(Hist3D,1,numel(Hist3D));
direct_hist = s_feat;
casd_TTC_SMCH = [casd_TTC_SMCH,direct_hist];


% 下面求取中间多边形的SMC特征
% 下面提取第4个特征
% plot(eight_spoins{1,4}(:,1),eight_spoins{1,4}(:,2),'*');
% P = P*2;
% if P == 8
%     astype = 'uint8';
% elseif P == 16
%     astype = 'uint16';
% elseif P == 32
%     astype = 'uint32';
% else
%     astype = 'uint64';  
% end
% mapping = getmapping(P,'riu2');           % 最多只能到P=24，再高内存不够，计算量太大
for r_num = 3:3

    [ysize,xsize] = size(fimg);
    spoints = eight_spoins{1,r_num};
    miny=min(spoints(:,1));
    maxy=max(spoints(:,1));
    minx=min(spoints(:,2));
    maxx=max(spoints(:,2));
    bsizey=ceil(max(maxy,0))-floor(min(miny,0))+1;
    bsizex=ceil(max(maxx,0))-floor(min(minx,0))+1;

    origy = 1-floor(min(miny,0));
    origx = 1-floor(min(minx,0));
    dx = xsize - bsizex;
    dy = ysize - bsizey;
    d_C = fimg(origy:origy+dy,origx:origx+dx);

    TTC_S = zeros(dy+1,dx+1);
    TTC_M = zeros(dy+1,dx+1);
    neighbor_cube = zeros(dy+1,dx+1,P2);
    Diff = zeros(dy+1,dx+1,P2);
    MeanDiff = zeros(1,P2);
    D_cube = zeros(size(neighbor_cube));
    for j = 1:P2
        y = spoints(j,1) + origy;
        x = spoints(j,2) + origx;
        fy = floor(y); cy = ceil(y); ry = round(y);
        fx = floor(x); cx = ceil(x); rx = round(x);
        if (abs(x - rx) < 1e-6) && (abs(y - ry) < 1e-6)
           N = fimg(ry:ry+dy,rx:rx+dx);
        else
           ty = y - fy;
           tx = x - fx;
           w1 = (1 - tx) * (1 - ty);
           w2 =      tx  * (1 - ty);
           w3 = (1 - tx) *      ty ;
           w4 =      tx  *      ty ;
           N = w1*fimg(fy:fy+dy,fx:fx+dx) + w2*fimg(fy:fy+dy,cx:cx+dx) + ...
               w3*fimg(cy:cy+dy,fx:fx+dx) + w4*fimg(cy:cy+dy,cx:cx+dx);
        end  
        D = N >= d_C;     
        Diff(:,:,j) = abs(N - d_C);    
        MeanDiff(j) = mean(mean(Diff(:,:,j)));
        v = 2^(j-1);
        TTC_S = TTC_S + v*D;
        neighbor_cube(:,:,j) = N;

        D_cube(:,:,j) = D;
    end

    %********************** 下面进行自适应阈值及修正 *******************
    if rectify == 1
       phi_t = zeros(size(neighbor_cube));
       phi_t(:,:,1) = (neighbor_cube(:,:,1)+neighbor_cube(:,:,P2))/2;
       for k = 2:P2
            phi_t(:,:,k) = (neighbor_cube(:,:,k)+neighbor_cube(:,:,k-1))/2;
        end
        % 下面判断是否是均匀模式，如果是非均匀模式再考虑应该恢复到均匀模式，自适应阈值
       phi_result = zeros(dy+1,dx+1,P2);     % P is odd, so x-(P-1)/2=x-P+1=dy+1
       for k = 1:P2
            new_Centre = phi_t(:,:,k);
            for j = 1:P2
                D = neighbor_cube(:,:,j) >= new_Centre;           % 在LBP算法中，C是确定的，DLABP算法需要自适应地选择阈值C 
                v = 2^(j-1);
                phi_result(:,:,k) = phi_result(:,:,k) + v*D;        
            end
        end
       bins = mapping2.num;
       if isstruct(mapping2)
          for k = 1:size(TTC_S,1)
              for j = 1:size(TTC_S,2)
                  TTC_S(k,j) = mapping2.table(TTC_S(k,j)+1);  % result元素范围：0-9
                  if (TTC_S(k,j) == bins-1)        % 如果是非均匀模式，则用8个gm重新阈值化，在得到映射值
                      temp1 = squeeze(phi_result(k,j,1:P2));
                      temp2 = squeeze(phi_t(k,j,1:P2));
                      corrupted_num = mapping2.table(temp1 + 1) < bins-1;
                      temp3 = find(corrupted_num);
                      if sum(corrupted_num) > 1     % 如果可恢复那么就寻找最小差距新中心
                         new_old_Cdist = abs(temp2(corrupted_num)-d_C(k,j));
                         [~,min_ind] = min(new_old_Cdist);
                         min_Cdist_ind = temp3(min_ind);
                         TTC_S(k,j) = mapping2.table(temp1(min_Cdist_ind) + 1);
%                            modified_C(k,j) = temp2(min_Cdist_ind);
                      end
                  else
                       continue;         % 如果不是均匀模式，则继续
                  end
              end
          end
       end 
        %***********************自适应阈值及修正结束********************
    elseif rectify == 2           % 如果是2则采用自己设计的矫正方法
           % 下面判断是否是均匀模式，如果是非均匀模式再考虑应该恢复到均匀模式，自适应阈值
           bins = mapping2.num;
           if isstruct(mapping2)
               for k = 1:size(TTC_S,1)
                   for j = 1:size(TTC_S,2)
                       temp = mapping2.table(TTC_S(k,j)+1);  % result元素范围：0-9
                       if (temp == bins-1)
                           % 首先统计0,1跳变次数
                           bit_seq = squeeze(D_cube(k,j,:));
                           rot_bit_seq = bitset(bitshift(TTC_S(k,j),1,astype),1,bitget(TTC_S(k,j),P2)); 
                           numt = sum(bitget(bitxor(TTC_S(k,j),rot_bit_seq),1:P2));
                           isrect = 0;
                           if numt == 4
                              [isrect,rectf_value] = check_rectify(bit_seq,squeeze(neighbor_cube(k,j,:)),d_C(k,j)); 
                           end
                           if (isrect == 1)                  % 如果是非均匀模式
                               % 如果是跳变次数等于4并且有孤立杂点存在并且孤立点与中心处差值在阈值范围内，那么就进行修正
                               TTC_S(k,j) = mapping2.table(rectf_value+1);
                               continue;
                           end
                       end
                       TTC_S(k,j) = temp;
                   end
               end
           end
    elseif rectify == 3              % 采用自适应阈值+添加新分类的方法
           phi_t = zeros(size(neighbor_cube));
           phi_t(:,:,1) = (neighbor_cube(:,:,1)+neighbor_cube(:,:,P2))/2;
           for k = 2:P2
                phi_t(:,:,k) = (neighbor_cube(:,:,k)+neighbor_cube(:,:,k-1))/2;
            end
            % 下面判断是否是均匀模式，如果是非均匀模式再考虑应该恢复到均匀模式，自适应阈值
           phi_result = zeros(dy+1,dx+1,P2);     % P is odd, so x-(P-1)/2=x-P+1=dy+1
           for k = 1:P2
                new_Centre = phi_t(:,:,k);
                for j = 1:P2
                    D = neighbor_cube(:,:,j) >= new_Centre;           % 在LBP算法中，C是确定的，DLABP算法需要自适应地选择阈值C 
                    v = 2^(j-1);
                    phi_result(:,:,k) = phi_result(:,:,k) + v*D;        
                end
            end
           bins = mapping2.num;
           if isstruct(mapping2)
              for k = 1:size(TTC_S,1)
                  for j = 1:size(TTC_S,2)
                      temp = mapping2.table(TTC_S(k,j)+1);  % result元素范围：0-9
                      if (temp == bins-1)        % 如果是非均匀模式，则用8个gm重新阈值化，在得到映射值
                          temp1 = squeeze(phi_result(k,j,1:P2));
                          temp2 = squeeze(phi_t(k,j,1:P2));
                          corrupted_num = mapping2.table(temp1 + 1) < bins-1;
                          temp3 = find(corrupted_num);
                          if sum(corrupted_num) > 1     % 如果可恢复那么就寻找最小差距新中心
                             new_old_Cdist = abs(temp2(corrupted_num)-d_C(k,j));
                             [~,min_ind] = min(new_old_Cdist);
                             min_Cdist_ind = temp3(min_ind);
                             TTC_S(k,j) = mapping2.table(temp1(min_Cdist_ind) + 1);
                             %  modified_C(k,j) = temp2(min_Cdist_ind);
                          else                       % 如果修正后还不是均匀模式，那么就根据跳变次数进行重新分类
%                                   % 首先统计0,1跳变次数
                              rot_bit_seq = bitset(bitshift(TTC_S(k,j),1,astype),1,bitget(TTC_S(k,j),P2)); 
                              numt = sum(bitget(bitxor(TTC_S(k,j),rot_bit_seq),1:P2));
                              if (numt <= 8)
                                 TTC_S(k,j) = mapping2.num - 2 + (numt/2-1);
                              else
                                  TTC_S(k,j) = mapping2.num - 1 + extend_pnum2;
                              end
%                                     TTC_S(k,j) = temp;
                          end
                      else
                          TTC_S(k,j) = temp;
                          continue;         % 如果不是均匀模式，则继续
                      end
                  end
              end
           end
            % 计算TTC_S和TTC_M
            DiffThreshold = mean(MeanDiff);
            for j = 1:P2
                v = 2^(j-1);
    %             TTC_M = TTC_M + v*(Diff(:,:,j) >= DiffThreshold);
                TTC_M = TTC_M + v*(Diff(:,:,j) >= MeanDiff(j));         % 1月14日下午，改为局部阈值试一下效果，结果是：
    %             TTC_M = TTC_M + v*(Diff(:,:,j) >= mean(Diff(:,:,j),3)); 
            end
            % 计算TTC_C
            TTC_C = d_C >= mean(fimg(:));
    %         TTC_C = d_C >= mean(fimg(:));  

            % 下面和TTC_S一样进行模式扩展
            for k = 1:size(TTC_M,1)
                  for j = 1:size(TTC_M,2)
                      temp = mapping2.table(TTC_M(k,j)+1); 
                      if (temp == bins-1)
                          rot_bit_seq = bitset(bitshift(TTC_M(k,j),1,astype),1,bitget(TTC_M(k,j),P2)); 
                          numt = sum(bitget(bitxor(TTC_M(k,j),rot_bit_seq),1:P2));
                          if (numt <= 8)
                             TTC_M(k,j) = mapping2.num - 2 + (numt/2-1);
                          else
                              TTC_M(k,j) = mapping2.num - 1 + extend_pnum2;
                          end 
                      else
                          TTC_M(k,j) = temp;
                      end
                  end
            end
            % 进行映射
%                 sizarray = size(TTC_M);
%                 TTC_M = TTC_M(:);
%                 TTC_M = mapping.table(TTC_M+1);
%                 TTC_M = reshape(TTC_M,sizarray);    

            % 特征直方图统计
            TTC_MCSum = TTC_M;
            idx = find(TTC_C);
            TTC_MCSum(idx) = TTC_MCSum(idx)+mapping2.num;
            TTC_SMC = [TTC_S(:),TTC_MCSum(:)];              % 下面hist3中后面的参数分别是TTC_S和TTC_MCSum的量化数目
            Hist3D = hist3(TTC_SMC,[mapping2.num+extend_pnum2,(mapping2.num+extend_pnum2)*2]);   % 横着有mapping.num个bar,侧着有mapping.num*2个bar
            TTC_SMCH(i,:) = reshape(Hist3D,1,numel(Hist3D));

%                 TTC_SMCH(i,:) = hist(TTC_S(:),0:(mapping.num-1+extend_pnum2));
%                 TTC_SMCH(i,:) = hist(TTC_M(:),0:(mapping.num-1+extend_pnum2));
%                 TTC_SMCH(i,:) = hist(TTC_M(:),0:(mapping.num-1));
            continue;

    else
        sizarray = size(TTC_S);
        TTC_S = TTC_S(:);
        TTC_S = mapping2.table(TTC_S+1);
        TTC_S = reshape(TTC_S,sizarray);
    end

    % 计算TTC_S和TTC_M
    DiffThreshold = mean(MeanDiff);
    for j = 1:P2
        v = 2^(j-1);
%             TTC_M = TTC_M + v*(Diff(:,:,j) >= DiffThreshold);
        TTC_M = TTC_M + v*(Diff(:,:,j) >= MeanDiff(j));         % 1月14日下午，改为局部阈值试一下效果，结果是：
%             TTC_M = TTC_M + v*(Diff(:,:,j) >= mean(Diff(:,:,j),3)); 
    end
    % 计算TTC_C
    TTC_C = d_C >= mean(fimg(:));
%         TTC_C = d_C >= mean(fimg(:));
    % 进行映射
    sizarray = size(TTC_M);
    TTC_M = TTC_M(:);
    TTC_M = mapping2.table(TTC_M+1);
    TTC_M = reshape(TTC_M,sizarray);
    % 特征直方图统计
    TTC_MCSum = TTC_M;
    idx = find(TTC_C);
    TTC_MCSum(idx) = TTC_MCSum(idx)+mapping2.num;
    TTC_SMC = [TTC_S(:),TTC_MCSum(:)];
    Hist3D = hist3(TTC_SMC,[mapping2.num,mapping2.num*2]);
    TTC_SMCH = reshape(Hist3D,1,numel(Hist3D));

    casd_TTC_SMCH = [casd_TTC_SMCH,TTC_SMCH];
end


% test_end = 1;





















