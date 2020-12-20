%---------------------------------------------------------------------------------------------------------%
%-----------------------------------------The live face detection-----------------------------------------%
% As input give an image object
% Output: Cropped image of just the face with dimensions 256*256

function rightImg = faceDetectionLive(inputImg)

    faceDetector = vision.CascadeObjectDetector();  % Create the face detector object
    faceDetector.MinSize = [100 100];
    img = inputImg;                                 % Read the inputted image to a
    bbox = step(faceDetector, img);                 % Create a box with 4 values that
    croppedImg = imcrop(img, bbox);                 % Crop the image with the values
    rightImg = imresize(croppedImg, [256 NaN]);

end
