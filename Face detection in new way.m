%% Image Processing and Face Detection Demo
clc; clear; close all;
warning('off');

%% Load and Display Image
[file, path] = uigetfile({'*.jpg;*.png;*.jpeg','Image Files'}, 'Select an image');
if isequal(file,0)
    disp('No file selected. Exiting...');
    return;
end
imgPath = fullfile(path, file);
img = imread(imgPath);
figure, imshow(img), title('Original Image');

%% Convert to Grayscale and Binary
grayImg = rgb2gray(img);
figure, imshow(grayImg), title('Grayscale Image');

level = graythresh(grayImg);
bwImg = imbinarize(grayImg, level);
figure, imshow(bwImg), title('Binary Image');

%% Save Binary Image
imwrite(bwImg, 'binary_output.jpg');
disp('Binary image saved as binary_output.jpg');

%% Crop & Resize
cropRect = [100 100 400 400];
cropped = imcrop(img, cropRect);
figure, imshow(cropped), title('Cropped Image');

resized = imresize(img, [500 500]);
figure, imshow(resized), title('Resized Image (500x500)');

%% Flip & Rotate
flipped = flip(img, 1);
figure, imshow(flipped), title('Flipped Image');

rotated = imrotate(img, 30, 'crop');
figure, imshow(rotated), title('Rotated Image (30°)');

%% Face Detection
gray = rgb2gray(img);
faceDetector = vision.CascadeObjectDetector('FrontalFaceCART');
bbox = step(faceDetector, gray);
annotated = insertObjectAnnotation(img, 'rectangle', bbox, 'Face');
figure, imshow(annotated), title('Face Detection');

%% Facial Feature Detection
features = {'Mouth', 'Nose', 'RightEye', 'UpperBody'};
for i = 1:length(features)
    detector = vision.CascadeObjectDetector(features{i});
    if strcmp(features{i}, 'Mouth')
        detector.MergeThreshold = 110;
    elseif strcmp(features{i}, 'RightEye')
        detector.MergeThreshold = 40;
    elseif strcmp(features{i}, 'UpperBody')
        detector.MergeThreshold = 5;
    end
    bbox = step(detector, gray);
    annotated = insertObjectAnnotation(img, 'rectangle', bbox, features{i});
    figure, imshow(annotated), title([features{i}, ' Detection']);
end

%% Webcam Initialization
camList = webcamlist;
if isempty(camList)
    error('No webcam detected.');
end
web = webcam(camList{1});
disp(['Using webcam: ', camList{1}]);

% Uncomment to preview:
% preview(web); pause(3); closePreview(web);

%% Live View
disp('Press Ctrl+C to stop live view.');
figure;
while true
    frame = snapshot(web);
    imshow(frame);
    title('Live Camera Feed');
    pause(0.01);
end

%% Real-time Face Detection (Press Ctrl+C to Stop)
clc; close all;
cam = webcam;
faceDet = vision.CascadeObjectDetector('FrontalFaceCART');
disp('Real-time face detection started...');
figure;
while true
    frame = snapshot(cam);
    grayFrame = rgb2gray(frame);
    bboxes = step(faceDet, grayFrame);
    annotatedFrame = insertObjectAnnotation(frame, 'rectangle', bboxes, 'Face');
    imshow(annotatedFrame);
    title('Real-time Face Detection');
    pause(0.01);
end
