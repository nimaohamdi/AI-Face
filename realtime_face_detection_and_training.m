

clear all

% install the webcam package
cam = webcam();
% I m just Entering my Available Resolution
% For Checkeing "cam.AvailableResolutions"
cam.Resolution = '640x480';
%  Now, to read the frames one by one from the ‘cam’ object
video.Frame = snapshot(cam);
% initiate the video player object
video_Player = vision.VideoPlayer('Position', [100 100 432 246]);

% detect the face
face_Detection = vision.CascadeObjectDetector();
point_Tracker = vision.PointTracker('MaxBidirectionalError', 2);

run_loop = true;
number_of_Points = 0;
frame_Count = 0;

% while loop. It will keep looping as long as fg is true and frame_Count is less than 400 frames. If I want to run the webcam for longer, I can increase the number of frames. Then end this loop. Inside this loop, we are going to do the detection and tracking
while run_loop && frame_Count <400

    
    video_Frame = snapshot(cam);
    %video_Reader = readFrame(video_Frame);
    gray_Frame = rgb2gray(video_Frame);
    frame_Count = frame_Count+1;
    if number_of_Points < 10
        face_Rectangle = face_Detection.step(gray_Frame);
        
        if ~isempty(face_Rectangle)
            points = detectMinEigenFeatures(gray_Frame, 'ROI', face_Rectangle(1, :));
            xy_Points = points.Location;
            number_of_points = size(xy_Points, 1);
            release(point_Tracker);
            initialize(point_Tracker, xy_Points, gray_Frame);
            
            previous_Points = xy_Points;
            
            rectangle = bbox2points(face_Rectangle(1, :));
            face_Polygon = reshape(rectangle', 1, []);
            
            video_Frame = insertShape(video_Frame, 'Polygon', face_Polygon, 'LineWidth', 3);
            video_Frame = insertMarker(video_Frame, xy_Points, '+', 'Color', 'white');
        end
        
    else
        [xy_points, isFound] = step(point_Tracker, gray_Frame);
        new_Points = previous_Points(isFound, :);
        number_of_points = size(new_Points, 1);
        
        if number_of_points >=10
            
            [xform, old_Points, new_Points] = estimateGeometricTransform(...
                old_Points, new_Points, 'similarity', 'MaxDistance', 4);
            rectangle = transformPointsForward(xform, rectangle);
            face_Polygon = reshape(rectangle', 1, []);
            video_Frame = insertShape(video_Frame, 'Polygon', face_Polygon, 'LineWidth', 3);
            video_Frame = insertMarker(video_Frame, new_Points, '+', 'Color', 'white');
            previous_Points = new_Points;
            %Point Tracker
            setPoints(point_Tracker, previous_Points);
        end
    end
    step(video_Player, video_Frame);
    run_loop = isOpen(video_Player);
end
% Closing
clear cam;

release(video_Player);
release(point_Tracker);
release(face_Detection);
