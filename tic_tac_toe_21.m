clear
clc
%% Creating a Grid
I=imread('blank img.jpg');
I = rgb2gray(I);
BW = imbinarize(I);
BW = imresize(BW,[640 640]);
BW(201:220,:) =0;
BW(:,201:220) =0;

BW(421:440,:) =0;
BW(:,421:440) =0;
load ('trainedNet_1.mat');
% imshow(BW)
%%
x= imread('X.jpg');
o= imread('O.png');

x= imbinarize(x);
o= rgb2gray(o);
o= imbinarize(o);
%%

X = imresize(x,[200 200]);

o = imresize(o,[200 200]);
% imshow(X)

%% Game
m=randperm(2);
p1 = m(1)
p2 = m(2)

grid = zeros(3);

game = 1;
itr = 1;
hold =0;

h = figure('Units','normalized','Position',[0.2 0.1 0.6 0.8])

while (game ==1 && itr~=10)
    if hold == 0
        if mod(itr,2)
            p=1
        else
            p=2
        end
    end
    
    grid
    
    if hold == 0
        xlabel("Player "+num2str(m(1))+": X     Player"+num2str(m(2))+": O",'FontSize',15);
    end
    
    if itr == 1
%         figure(1)
        subplot(3,1,1)
        imshow(BW);
    end
    
    
    title("kindly select an empty box",'FontSize',15);
    %     choice= audi_input();
    %     choice =input("Enter a number");
    %     numbers = [1 2 3 4 5 6 7 8 9];
    
    
    % .................................................................................................
  %Getting input through speech
    
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
    
    
    labels = trainedNet_1.Layers(end).Classes;
    YBuffer(1:classificationRate/2) = categorical("background");
    probBuffer = zeros([numel(labels),classificationRate/2]);
    %h = figure('Units','normalized','Position',[0.2 0.1 0.6 0.8]);
    
    filterBank = designAuditoryFilterBank(fs,'FrequencyScale','bark',...
        'FFTLength',512,...
        'NumBands',numBands,...
        'FrequencyRange',[50,7000]);
    
    while (ishandle(h) && state ~= 3)
        
        
        x = audioIn();
        waveBuffer(1:end-numel(x)) = waveBuffer(numel(x)+1:end);
        waveBuffer(end-numel(x)+1:end) = x;
        
        [~,~,~,spec] =  spectrogram(waveBuffer,hann(frameLength,'periodic'),frameLength - hopLength,512,'onesided');
        spec = filterBank * spec;
        spec = log10(spec + epsil);
        
        [YPredicted,probs] = classify(trainedNet_1,spec,'ExecutionEnvironment','cpu');
        YBuffer(1:end-1)= YBuffer(2:end);
        YBuffer(end) = YPredicted;
        probBuffer(:,1:end-1) = probBuffer(:,2:end);
        probBuffer(:,end) = probs';
        
        
        subplot(3,1,2);
        plot(waveBuffer)
        axis tight
        ylim([-0.2,0.2])
        
        subplot(3,1,3);
        pcolor(spec)
        caxis([specMin+2 specMax])
        shading flat
        
        [YMode,count] = mode(YBuffer);
        countThreshold = ceil(classificationRate*0.2);
        maxProb = max(probBuffer(labels == YMode,:));
        probThreshold = 0.8;
        subplot(3,1,2);
        if YMode == "background" || count<countThreshold || maxProb < probThreshold
            if state ==0
                
                title(" ")
            end
        else
            
            title(string(YMode),'FontSize',20)
            if state == 0 && strcmpi(char(YMode),'yes') == 0 && strcmpi(char(YMode),'no') == 0 
                title(string(YMode),'FontSize',20)
                answer = char(YMode);
                state =1;
            end
            
        end
        
        
        if state ==1

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
            state =3;
        end
        
    end
    
    % .................................................................................................
    
    % Determining the row and column value
    numbers = ["one","two","three","four","five","six","seven","eight","nine"];
    
    num = find(numbers == choice);
    
    if mod(num, 3) == 0
        col = 3;
    else
        col = mod(num, 3);
    end
    
    if num < 4
        row = 1;
    elseif num < 7
        row = 2;
    else
        row = 3;
    end
    
    %Condition to check if the input location is being filled
    if grid(row,col) ~= 0
        subplot (3,1,1)
        xlabel("position has been taken.. try again",'FontSize',15);
        %flag to hold the iteration value
        hold = 1;
        
    else
        %determining the edge pixel coordinate of the desired cluster cell
        %of the grid
        hold = 0;
        grid(row,col) = p;
        pic_row = 200*(row-1)+ 20*(row-1)+1;
        pic_col = 200*(col-1)+ 20*(col-1)+1;
        
        if p ==1
            img_insrt = X;
        else
            img_insrt = o;
        end
        
        
        %Determining range of pixels to be overlapped 
        BW(pic_row:pic_row+199 , pic_col:pic_col+199) = img_insrt;
        subplot (3,1,1)
        imshow(BW);
        
        %Checking if the row or column values are the same
        %If values are same player has won
        rw_ln = grid(row, :);
        cl_ln = grid(: ,col);
        cl_ln = cl_ln';
        
        rw_repeats = find(rw_ln == p);
        rw_repeats = size(rw_repeats);
        rw_repeats = rw_repeats(2);
        
        cl_repeats = find(cl_ln == p);
        cl_repeats = size(cl_repeats);
        cl_repeats = cl_repeats(2);
        
        if cl_repeats == 3 || rw_repeats == 3
            game = 0;
            break;
            
        else
            %Checking whether the diagonal values are the same
            diag_1 = diag(grid);
            diag_1 = diag_1'
            diag_2 = [grid(1,3), grid(2,2), grid(3,1)];
            if ( row == col )
                if row == 2 && col == 2
                    diag_1_rpt = find(diag_1 == p);
                    diag_1_rpt= size(diag_1_rpt);
                    diag_1_rpt = diag_1_rpt(2)
                    
                    diag_2_rpt = find(diag_2 == p);
                    diag_2_rpt = size(diag_2_rpt);
                    diag_2_rpt = diag_2_rpt(2)
                    
                    if diag_2_rpt == 3 || diag_1_rpt == 3
                        game = 0;
                        break;
                    end
                else
                    diag_1_rpt = find(diag_1 == p);
                    diag_1_rpt= size(diag_1_rpt);
                    diag_1_rpt = diag_1_rpt(2)
                    
                    if diag_1_rpt == 3
                        game = 0;
                        break;
                    end
                end
                
            elseif ( row + col == 4)
                diag_2_rpt = find(diag_2 == p);
                diag_2_rpt = size(diag_2_rpt);
                diag_2_rpt = diag_2_rpt(2);
                
                if  diag_2_rpt == 3
                    game = 0;
                    break;
                end
                
            end
            
        end
        
    end
    if hold ==0
        itr = itr+1;
    end
    
end

if game == 0
    if find(m==p)==1
        winner = "Player 1"
    else
        winner = "Player 2"
    end
else
    winner = "No Winners Today!!"
end
subplot(3,1,1)
xlabel("Player "+num2str(m(1))+": X     Player"+num2str(m(2))+": O",'FontSize',15);
title("Winner: " + winner,'FontSize',15);

