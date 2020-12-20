%---------------------------------------------------------------------------------------------------------%
%-----------------------------------------The live face detection-----------------------------------------%
% As input give the path to an image
% Output: Cropped image of just the face with dimensions 256*256

function rightImg = faceDetection(inputImg)

    faceDetector = vision.CascadeObjectDetector();  % Create the face detector object
    faceDetector.MinSize = [100 100];
    img = imread(fullfile(inputImg));               % Read the inputted image to a
    % to a temporary variable
    bbox = step(faceDetector, img);                 % Create a box with 4 values that
    % are around the face
    croppedImg = imcrop(img, bbox);                 % Crop the image with the values
    rightImg = imresize(croppedImg, [256 NaN]);
    
end
