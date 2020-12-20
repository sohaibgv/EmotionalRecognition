%---------------------------------------------------------------------------------------------------------%
%-----------------------------------------Training the classifier-----------------------------------------%
% First read our locally stored dataset and detect the faces

setDir = fullfile('../ImagesUncropped');
imds = imageDatastore(setDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imds.ReadFcn = @faceDetection;
imds = shuffle(imds);
[trainingSet, testSet] = splitEachLabel(imds, 0.7, 'randomize');

%---------------------------------------------------------------------------------------------------------%
%---------------------------Extracting the features and training the classifier---------------------------%
% Then create a bag of features out of 70% of the dataset
% And train the image category classifier using gaussian SVM

bag = bagOfFeatures(trainingSet, 'VocabularySize', 1000, 'GridStep', [8 8]);
options = templateSVM('KernelFunction', 'gaussian'); % 'gaussian', 'linear', 'polynomial'
categoryClassifier = trainImageCategoryClassifier(trainingSet, bag, 'LearnerOptions', options);

%---------------------------------------------------------------------------------------------------------%
%-----------------------------------------Displaying our accuracy-----------------------------------------%
% Create a confusion matrix by classifying our test dataset
% Calculate the average accuracy and call the UI to start the application.

confMatrix = evaluate(categoryClassifier, testSet);
mean(diag(confMatrix));
imageTaker(categoryClassifier);
