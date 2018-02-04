% TSSHM for UIUC dataset
% Create date: 26/1/2018
% Author: Dongdong Ma

clc;
clear;

R = 4;       % radius of the tree   
r = 2;       % raiuds of the circles on the branches
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
mapping1 = getmapping(P1,'riu2');        
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
mapping2 = getmapping(P2,'riu2');          

% calculate the position of the 21 poinits. format: [delta_i,delta_j]
eight_spoins{1,1} = [0,R-r;-(R-r)/2,(R-r)/2;-(R-r),0;-(R-r)/2,-(R-r)/2;0,-(R-r);(R-r)/2,-(R-r)/2;(R-r),0;(R-r)/2,(R-r)/2];
eight_spoins{1,2} = [0,R+r;-(R+r)/2,(R+r)/2;-(R+r),0;-(R+r)/2,-(R+r)/2;0,-(R+r);(R+r)/2,-(R+r)/2;(R+r),0;(R+r)/2,(R+r)/2];
eight_spoins{1,3} = [0,R;-r,R;-(R+r)/2,(R+r)/2;-R,r;-R,0;-R,-r;-(R+r)/2,-(R+r)/2;-r,-R; ...
                     0,-R;r,-R;(R+r)/2,-(R+r)/2;R,-r;R,0;R,r;(R+r)/2,(R+r)/2;r,R];
% calculate the rest 4 points
temp = [2,4,6,8];
eight_spoins{1,1}(temp,:) = eight_spoins{1,1}(temp,:)*sqrt(2);
eight_spoins{1,2}(temp,:) = eight_spoins{1,2}(temp,:)*sqrt(2);

rootpic = '/home/ma/work/madd/LBP/dataset/UIUC/';
cls_name = {'T01_bark1','T02_bark2','T03_bark3','T04_wood1','T05_wood2','T06_wood3','T07_water','T08_granite','T09_marble',...
            'T10_floor1','T11_floor2','T12_pebbles','T13_wall','T14_brick1','T15_brick2','T16_glass1','T17_glass2','T18_carpet1',...
            'T19_carpet2','T20_upholstery','T21_wallpaper','T22_fur','T23_knit','T24_corduroy','T25_plaid'};
cls_num = 25;
train_cls_num = 1:4:40;
test_cls_num = setdiff(1:40,train_cls_num);

trains = [];
tests = [];
trainClassIDs = [];
testClassIDs = [];
counter1 = 1;
counter2 = 1;
for i = 1:cls_num
    base_path = strcat(rootpic,cls_name{1,i},'/T',num2str(i,'%02d'),'_');
    for j = 1:length(train_cls_num)
        filename = strcat(base_path,num2str(j,'%02d'),'.jpg');
        Gray = imread(filename);
        Gray = im2double(Gray);
        
        % add the gaussian noise here
%         noiseMat = normrnd(0,0.01,size(Gray,1),size(Gray,2));
%         Gray = Gray + noiseMat;
        
        image = (Gray-mean(Gray(:)))/std(Gray(:));                 % normalization
        
        trains(counter1,:) = TSSHM(image,P1,P2,mapping1,mapping2,eight_spoins,R,r);
        trainClassIDs = [trainClassIDs,i];
        counter1 = counter1 + 1;
    end
    for j = 1:length(test_cls_num)
        filename = strcat(base_path,num2str(j,'%02d'),'.jpg');
        Gray = imread(filename);
        Gray = im2double(Gray);
        
        % add the gaussian noise here
%         noiseMat = normrnd(0,0.01,size(Gray,1),size(Gray,2));
%         Gray = Gray + noiseMat;
        
        image = (Gray-mean(Gray(:)))/std(Gray(:));         % image normalization
        
        tests(counter2,:) = TSSHM(image,P1,P2,mapping1,mapping2,eight_spoins,R,r);
        testClassIDs = [testClassIDs,i];
        counter2 = counter2 + 1;
    end
end

trainNum = size(trains,1);
testNum = size(tests,1);

DM = zeros(testNum,trainNum);
for i=1:testNum
    test = tests(i,:);        
    DM(i,:) = distMATChiSquare(trains,test)';
end
TSSHM_CP = ClassifyOnNN(DM,trainClassIDs,testClassIDs)






