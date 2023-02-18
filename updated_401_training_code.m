%% Speech Command Recognition Using Deep Learning

%% Load Speech Commands Data Set

datafolder = '401 test';
ads = audioDatastore(datafolder, ...
    'IncludeSubfolders',true, ...
    'FileExtensions','.wav', ...
    'LabelSource','foldernames');
ads0 = copy(ads);

%% Choose Words to Recognize
commands = categorical(["one","two","three","four","five","six","seven","eight","nine","yes","no"]);

isCommand = ismember(ads.Labels,commands);
ads = subset(ads,isCommand);
countEachLabel(ads)

%% Split Data into Training, Validation, and Test Sets
c = fileread(fullfile(datafolder,'validation_list.txt'));
filesValidation = string(split(c));
filesValidation  = filesValidation(filesValidation ~= "");

files = ads.Files;
sf    = split(files,filesep);
isValidation = ismember(sf(:,end-1) + "/" + sf(:,end),filesValidation);

adsValidation = subset(ads,isValidation);
adsTrain = subset(ads,~isValidation);

%% Training and Validation spectrograms
segmentDuration = 1;
frameDuration = 0.025;
hopDuration = 0.010;
numBands = 40;
epsil = 1e-6;

XTrain = speechSpectrograms(adsTrain,segmentDuration,frameDuration,hopDuration,numBands);
XTrain = log10(XTrain + epsil);

XValidation = speechSpectrograms(adsValidation,segmentDuration,frameDuration,hopDuration,numBands);
XValidation = log10(XValidation + epsil);

YTrain = adsTrain.Labels;
YValidation = adsValidation.Labels;


%% Add Background Noise Data
adsBkg = subset(ads0,ads0.Labels=="_background_noise_");
numBkgClips = 4000;
volumeRange = [1e-4,1];

XBkg = backgroundSpectrograms(adsBkg,numBkgClips,volumeRange,segmentDuration,frameDuration,hopDuration,numBands);

XBkg = log10(XBkg + epsil);

%% Splitting Background noises into training and validation
numTrainBkg = floor(0.8*numBkgClips);
numValidationBkg = floor(0.2*numBkgClips);

XTrain(:,:,:,end+1:end+numTrainBkg) = XBkg(:,:,:,1:numTrainBkg);
XBkg(:,:,:,1:numTrainBkg) = [];
YTrain(end+1:end+numTrainBkg) = "background";

XValidation(:,:,:,end+1:end+numValidationBkg) = XBkg(:,:,:,1:numValidationBkg);
XBkg(:,:,:,1:numValidationBkg) = [];
YValidation(end+1:end+numValidationBkg) = "background";


YTrain = removecats(YTrain);
YValidation = removecats(YValidation);

%% Data Histogram

figure('Units','normalized','Position',[0.2 0.2 0.5 0.5]);
subplot(2,1,1)
histogram(YTrain)
title("Training Label Distribution")
subplot(2,1,2)
histogram(YValidation)
title("Validation Label Distribution")

%% Add Data Augmentation
sz = size(XTrain);
specSize = sz(1:2);
imageSize = [specSize 1];
augmenter = imageDataAugmenter( ...
    'RandXTranslation',[-10 10], ...
    'RandXScale',[0.8 1.2], ...
    'FillValue',log10(epsil));
augimdsTrain = augmentedImageDatastore(imageSize,XTrain,YTrain, ...
    'DataAugmentation',augmenter);

%% Define Neural Network Architecture
classWeights = 1./countcats(YTrain);
classWeights = classWeights'/mean(classWeights);
numClasses = numel(categories(YTrain));

timePoolSize = ceil(imageSize(2)/8);
dropoutProb = 0.25;
numF = 15;
layers = [
    imageInputLayer(imageSize)
    
    convolution2dLayer(8,numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(8,'Stride',2,'Padding','same')
    
    convolution2dLayer(8,2*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(8,'Stride',2,'Padding','same')
    
    convolution2dLayer(8,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(8,'Stride',2,'Padding','same')
    
    convolution2dLayer(8,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(8,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer([1 timePoolSize])
    
    dropoutLayer(dropoutProb)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    weightedClassificationLayer(classWeights)];

%% Train Network
miniBatchSize = 128;
validationFrequency = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions('adam', ...
    'InitialLearnRate',3e-4, ...
    'MaxEpochs',25, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ValidationData',{XValidation,YValidation}, ...
    'ValidationFrequency',validationFrequency, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20)


%% Training the data using the designed network layers
trainedNet_1 = trainNetwork(augimdsTrain,layers,options)
% load('trainedNet.mat');


%% Evaluate Trained Network
YValPred = classify(trainedNet_1,XValidation);
validationError = mean(YValPred ~= YValidation);
YTrainPred = classify(trainedNet_1,XTrain);
trainError = mean(YTrainPred ~= YTrain);
disp("Training error: " + trainError*100 + "%")
disp("Validation error: " + validationError*100 + "%")

%% Confusion matrix
figure('Units','normalized','Position',[0.2 0.2 0.5 0.5]);
cm = confusionchart(YValidation,YValPred);
cm.Title = 'Confusion Matrix for Validation Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
sortClasses(cm, [commands,"background"])


%% Single img prediction time

info = whos('trainedNet_1');
disp("Network size: " + info.bytes/1024 + " kB")

for i=1:100
    x = randn(imageSize);
    tic
    [YPredicted,probs] = classify(trainedNet_1,x,"ExecutionEnvironment",'cpu');
    time(i) = toc;
end
disp("Single-image prediction time on CPU: " + mean(time(11:end))*1000 + " ms")


