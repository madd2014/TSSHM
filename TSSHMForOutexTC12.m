% TSSHM for Outex_TC12 dataset
% Create date: 26/1/2018
% Author: Dongdong Ma


clc;
clear;

R = 4;                       % 在这种设计方式下r的值必须小于R的值
r = 2;
P1 = 8;
if P1 == 8
    astype1 = 'uint8';
elseif P1 == 16
    astype1 = 'uint16';
elseif P1 == 32
    astype1 = 'uint32';
else
    astype1 = 'uint64';  
end
mapping1 = getmapping(P1,'riu2');           % 最多只能到P=24，再高内存不够，计算量太大
P2 = P1*2;
if P2 == 8
    astype2 = 'uint8';
elseif P2 == 16
    astype2 = 'uint16';
elseif P2 == 32
    astype2 = 'uint32';
else
    astype2 = 'uint64';  
end
mapping2 = getmapping(P2,'riu2');           % 最多只能到P=24，再高内存不够，计算量太大

% calculate the position of the 21 poinits. format: [delta_i,delta_j]
eight_spoins{1,1} = [0,R-r;-(R-r)/2,(R-r)/2;-(R-r),0;-(R-r)/2,-(R-r)/2;0,-(R-r);(R-r)/2,-(R-r)/2;(R-r),0;(R-r)/2,(R-r)/2];
eight_spoins{1,2} = [0,R+r;-(R+r)/2,(R+r)/2;-(R+r),0;-(R+r)/2,-(R+r)/2;0,-(R+r);(R+r)/2,-(R+r)/2;(R+r),0;(R+r)/2,(R+r)/2];
eight_spoins{1,3} = [0,R;-r,R;-(R+r)/2,(R+r)/2;-R,r;-R,0;-R,-r;-(R+r)/2,-(R+r)/2;-r,-R; ...
                     0,-R;r,-R;(R+r)/2,-(R+r)/2;R,-r;R,0;R,r;(R+r)/2,(R+r)/2;r,R];
% calculate the rest 4 points
temp = [2,4,6,8];
eight_spoins{1,1}(temp,:) = eight_spoins{1,1}(temp,:)*sqrt(2);
eight_spoins{1,2}(temp,:) = eight_spoins{1,2}(temp,:)*sqrt(2);

% rootpic = '/home/ma/work/刘曦/LBP/dataset/Outex/Outex-TC-00010/';
rootpic = '/home/ma/work/madd/LBP/dataset/Outex/Outex_TC_00012/';

% picNum = 4320;
picNum = 9120;
casd_TSSHM = [];
for i = 1:picNum
%     filename = strcat(rootpic,'images/',num2str(i-1,'%06d'),'.bmp');
    filename = strcat(rootpic,'images/',num2str(i-1,'%06d'),'.ras');
    Gray = imread(filename);
    Gray = im2double(Gray);
%     imshow(Gray);
    
    % add the gaussian noise here
%     noiseMat = normrnd(0,0.01,size(Gray,1),size(Gray,2));
%     Gray = Gray + noiseMat;
    
    image = (Gray-mean(Gray(:)))/std(Gray(:)); % image normalization, to remove global intensity

    casd_TSSHM(i,:) = TSSHM(image,P1,P2,mapping1,mapping2,eight_spoins,R,r);
end


% trainTxt = strcat(rootpic,'000/train.txt');
% testTxt = strcat(rootpic,'000/test.txt');
trainTxt = strcat(rootpic,'000/train.txt');
testTxt = strcat(rootpic,'000/test.txt');

[trainIDs, trainClassIDs] = ReadOutexTxt(trainTxt);
[testIDs, testClassIDs] = ReadOutexTxt(testTxt);

% training set and testing set generation
trains = casd_TSSHM(trainIDs,:);
tests = casd_TSSHM(testIDs,:);
trainNum = size(trains,1);
testNum = size(tests,1);

DM = zeros(testNum,trainNum);
for i=1:testNum
    test = tests(i,:);        
    DM(i,:) = distMATChiSquare(trains,test)';
end
TSSHM_CP = ClassifyOnNN(DM,trainClassIDs,testClassIDs)
