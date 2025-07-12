% Define folders
goodFolder = 'Good';
defectiveFolder = 'Defective';

% Create imageDatastore with custom read function
imds = imageDatastore({goodFolder, defectiveFolder}, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames', ...
    'ReadFcn', @(x) preprocessImage(x));

% Define function to resize images
function img = preprocessImage(filename)
    img = imread(filename);
    img = imresize(img, [224 224]); % Resize to AlexNet expected size
    % Convert grayscale images to RGB
    if size(img, 3) == 1
        img = cat(3, img, img, img);  % Duplicate channels to make RGB
    end

end

% Split dataset
[trainDS, testDS] = splitEachLabel(imds, 0.8, 'randomized');

imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-10, 10], ...
    'RandXReflection', true, ...
    'RandYReflection', true, ...
    'RandXTranslation', [-5, 5], ...
    'RandYTranslation', [-5, 5]);

augimdsTrain = augmentedImageDatastore([224 224 3], trainDS, ...
    'DataAugmentation', imageAugmenter);


net = resnet18;
lgraph = layerGraph(net);

% Replace the last fully connected layer and classification layer
% These are the default names in resnet18
newFCLayer = fullyConnectedLayer(2, 'Name', 'new_fc'); % 2 classes
newClassLayer = classificationLayer('Name', 'new_class');

% Replace layers by name
lgraph = replaceLayer(lgraph, 'fc1000', newFCLayer);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions', newClassLayer);


% Training options
options = trainingOptions('adam', ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 32, ...
    'ValidationData', testDS, ...
    'ValidationFrequency', 30, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% Train the modified ResNet
trainedNet = trainNetwork(augimdsTrain, lgraph, options);

%Save the model
save('tyreClassifier.mat', 'trainedNet');



function result = classifyTyre(imagePath, trainedNet)
    if exist(imagePath, 'file') ~= 2
        error('Image not found: %s', imagePath);
    end
    img = imread(imagePath);
    img = imresize(img, [224 224]); % Ensure correct input size
    if size(img, 3) == 1
        img = cat(3, img, img, img);  % Convert grayscale to RGB
    end
    label = classify(trainedNet, img);
    result = string(label);
end

% Example usage with correct path
imgPath = fullfile(pwd, 'Defective (28).jpg');
tyreCondition = classifyTyre(imgPath, trainedNet);
disp(['Tyre condition: ', tyreCondition]);


% Predict labels on validation/test set
predictedLabels = classify(trainedNet, testDS);  % Or use imdsValidation if using that
trueLabels = testDS.Labels;

% Confusion matrix
confMat = confusionmat(trueLabels, predictedLabels);
disp('Confusion Matrix:');
disp(confMat);

% Accuracy
accuracy = sum(predictedLabels == trueLabels) / numel(trueLabels);
disp(['Accuracy: ', num2str(accuracy * 100), '%']);

% Precision, Recall, F1 Score per class
classes = categories(trueLabels);
numClasses = numel(classes);

for i = 1:numClasses
    TP = confMat(i, i);
    FP = sum(confMat(:, i)) - TP;
    FN = sum(confMat(i, :)) - TP;

    precision = TP / (TP + FP + eps);
    recall = TP / (TP + FN + eps);
    f1 = 2 * (precision * recall) / (precision + recall + eps);

    fprintf('\nClass: %s\n', classes{i});
    fprintf('Precision: %.2f\n', precision);
    fprintf('Recall: %.2f\n', recall);
    fprintf('F1 Score: %.2f\n', f1);
end
