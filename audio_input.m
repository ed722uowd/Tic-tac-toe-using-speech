function choice = audio_input ()

% load ('specMin.mat');
% load ('specMax.mat');

load ('trainedNet.mat');
    


% Create a figure and detect commands as long as the created figure exists.
% To stop the live detection, simply close the figure.

segmentDuration = 1;
frameDuration = 0.025;
hopDuration = 0.010;
numBands = 40;

fs = 16e3;
classificationRate = 20;
audioIn = audioDeviceReader('SampleRate',fs, ...
    'SamplesPerFrame',floor(fs/classificationRate));

frameLength = floor(frameDuration*fs);
hopLength = floor(hopDuration*fs);
waveBuffer = zeros([fs,1]);

state = 0;


labels = trainedNet.Layers(end).Classes;
YBuffer(1:classificationRate/2) = categorical("background");
probBuffer = zeros([numel(labels),classificationRate/2]);
h = figure('Units','normalized','Position',[0.2 0.1 0.6 0.8]);

filterBank = designAuditoryFilterBank(fs,'FrequencyScale','bark',...
    'FFTLength',512,...
    'NumBands',numBands,...
    'FrequencyRange',[50,7000]);

while ishandle(h)
    state
    % Extract audio samples from the audio device and add the samples to
    % the buffer.
    x = audioIn();
    waveBuffer(1:end-numel(x)) = waveBuffer(numel(x)+1:end);
    waveBuffer(end-numel(x)+1:end) = x;
    
    % Compute the spectrogram of the latest audio samples.
    [~,~,~,spec] =  spectrogram(waveBuffer,hann(frameLength,'periodic'),frameLength - hopLength,512,'onesided');
    spec = filterBank * spec;
    spec = log10(spec + epsil);
    
    % Classify the current spectrogram, save the label to the label buffer,
    % and save the predicted probabilities to the probability buffer.
    [YPredicted,probs] = classify(trainedNet,spec,'ExecutionEnvironment','cpu');
    YBuffer(1:end-1)= YBuffer(2:end);
    YBuffer(end) = YPredicted;
    probBuffer(:,1:end-1) = probBuffer(:,2:end);
    probBuffer(:,end) = probs';
    
    %     figure('Name','Audio','NumberTitle','off');
    
    % Plot the current waveform and spectrogram.
    %     figure(2)
    subplot(2,1,1);
    plot(waveBuffer)
    axis tight
    ylim([-0.2,0.2])
    
    subplot(2,1,2);
    pcolor(spec)
    caxis([specMin+2 specMax])
    shading flat
    
    % Now do the actual command detection by performing a very simple
    % thresholding operation. Declare a detection and display it in the
    % figure title if all of the following hold:
    % 1) The most common label is not |background|.
    % 2) At least |countThreshold| of the latest frame labels agree.
    % 3) The maximum predicted probability of the predicted label is at
    % least |probThreshold|. Otherwise, do not declare a detection.
    [YMode,count] = mode(YBuffer);
    countThreshold = ceil(classificationRate*0.2);
    maxProb = max(probBuffer(labels == YMode,:));
    probThreshold = 0.8;
    subplot(2,1,1);
    if YMode == "background" || count<countThreshold || maxProb < probThreshold
        if state ==0
            %             subplot(3,1,2);
            title(" ")
        end
    else
        %         subplot(3,1,2);
        title(string(YMode),'FontSize',20)
        if state == 0 && strcmpi(char(YMode),'yes') == 0 && strcmpi(char(YMode),'no') == 0 && strcmpi(char(YMode),'unknown') == 0
            title(string(YMode),'FontSize',20)
            answer = char(YMode);
            state =1;
        end
        
    end
    
    
    if state ==1
        %         subplot(3,1,2);
        title("Selected "+ answer+ " Confirm with a yes to proceed or no to choose another value",'FontSize',15);
        if strcmpi(char(YMode),'yes')
            state =2;
            
        elseif strcmpi(char(YMode),'no')
            state=0;
        end
        
        
    end
    
    
    drawnow
    
    if state == 2
        choice = answer;
        close;
    end
    
end
end