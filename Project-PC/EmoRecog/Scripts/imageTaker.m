%---------------------------------------------------------------------------------------------------------%
%-----------------------------------User Interface to identify emotions-----------------------------------%
% Input: a category classifier
% Displays a UI that takes a snapshot on button click and identifies emotion

function output = imageTaker(categoryClassifier)

    %---------------------------------------------------------------------------------------------------------%
    %----------------------------------Create a UI and display a live webcam----------------------------------%
    % Use a figure to display a live webcam

    fig = figure();
    ax = axes('Parent', fig, 'Units', 'normalized', 'Position', [0 0 1 1]);
    cam = webcam(1);
    camsize = str2double(strsplit(cam.Resolution, 'x'));
    im = image(zeros(camsize), 'Parent', ax);
    preview(cam, im);
    axis image;

    %---------------------------------------------------------------------------------------------------------%
    %--------------------------------------Creating and binding a button--------------------------------------%
    % Create a button and display text into it
    % Bind the button click to the function evaluator

    ButtonH = uicontrol('Parent', fig, 'Style', 'pushbutton', 'String', 'Take image and predict', 'Units', 'normalized', 'Position', [0.0 0.0 1 0.2], 'Visible', 'on');
    ButtonH.Callback = {@evaluator, cam, categoryClassifier, ButtonH};
    set(0, 'DefaultUIControlFontSize', 28);

    %---------------------------------------------------------------------------------------------------------%
    %--------------------------------------------Evaluator function-------------------------------------------%
    % Take a snapshot and crop the image around the face
    % Predict the category and display the category on the button with additional text

    function evaluator(src, event, cam, categoryClassifier, ButtonH)
        frame = snapshot(cam);
        faceImg = faceDetectionLive(frame);
        [labelIndex, score] = predict(categoryClassifier, faceImg); % test it
        label = categoryClassifier.Labels(labelIndex)
        score
        set(ButtonH, 'String', sprintf('You are looking %s. Please press again to analyze your emotions.', label{:}));
    end

end
