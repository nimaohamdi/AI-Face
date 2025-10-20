%% Real-Time Face Detection and Tracking
clc; clear; close all;

% Initialize webcam
cam = webcam();
disp('✅ Webcam initialized successfully');
cam.Resolution = '640x480';

% Read first frame
frame = snapshot(cam);

% Initialize video player
video_Player = vision.VideoPlayer('Position', [100 100 640 480]);

% Create face detector and point tracker
face_Detector = vision.CascadeObjectDetector();
point_Tracker = vision.PointTracker('MaxBidirectionalError', 2);

% Initialize variables
run_loop = true;
num_Points = 0;
frame_Count = 0;
max_Frames = 400;

disp('🎥 Starting real-time detection (Press X on video window to stop)');

while run_loop && frame_Count < max_Frames
    frame = snapshot(cam);
    gray_Frame = rgb2gray(frame);
    frame_Count = frame_Count + 1;

    if num_Points < 10
        % --- FACE DETECTION ---
        bbox = step(face_Detector, gray_Frame);

        if ~isempty(bbox)
            % Use the first detected face
            bbox = bbox(1, :);

            % Detect strong corner points inside face
            points = detectMinEigenFeatures(gray_Frame, 'ROI', bbox);

            if points.Count >= 10
                xy_Points = points.Location;
                num_Points = size(xy_Points, 1);

                % Initialize tracker with detected points
                release(point_Tracker);
                initialize(point_Tracker, xy_Points, gray_Frame);
                previous_Points = xy_Points;

                % Draw bounding box
                rectanglePts = bbox2points(bbox);
                face_Polygon = reshape(rectanglePts', 1, []);
                frame = insertShape(frame, 'Polygon', face_Polygon, 'LineWidth', 3, 'Color', 'green');
                frame = insertMarker(frame, xy_Points, '+', 'Color', 'white');
            end
        end

    else
        % --- TRACKING MODE ---
        [xy_Points, isFound] = step(point_Tracker, gray_Frame);
        new_Points = xy_Points(isFound, :);
        old_Points = previous_Points(isFound, :);
        num_Points = size(new_Points, 1);

        if num_Points >= 10
            % Estimate geometric transform
            [xform, old_Points, new_Points] = estimateGeometricTransform2D(...
                old_Points, new_Points, 'similarity', 'MaxDistance', 4);

            % Transform bounding box
            rectanglePts = transformPointsForward(xform, rectanglePts);
            face_Polygon = reshape(rectanglePts', 1, []);
            frame = insertShape(frame, 'Polygon', face_Polygon, 'LineWidth', 3, 'Color', 'green');
            frame = insertMarker(frame, new_Points, '+', 'Color', 'white');

            % Update tracker
            previous_Points = new_Points;
            setPoints(point_Tracker, previous_Points);
        else
            % Lost track — reset
            num_Points = 0;
            release(point_Tracker);
        end
    end

    % Display frame
    step(video_Player, frame);
    run_loop = isOpen(video_Player);
end

% Cleanup
clear cam;
release(video_Player);
release(point_Tracker);
release(face_Detector);
disp('✅ Tracking ended and resources released.');
