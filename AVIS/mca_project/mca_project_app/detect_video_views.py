import os
import tempfile
import cv2
import numpy as np
from django.core.files.base import ContentFile
from django.shortcuts import render, redirect, get_object_or_404
from .models import DetectVideo
from ultralytics import YOLO
from .sort.sort import Sort
import easyocr
from . import util  # Import your existing util.py file
from django.urls import reverse
from mca_project import settings

# Initialize the SORT tracker
mot_tracker = Sort()

# Load YOLOv8 model for vehicle detection (coco_model) and license plate detection (license_plate_detector)
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('mca_project_app/models/license_plate_detection_best_training4_70epochs.pt')

# Initialize EasyOCR reader.
# ['en'] specifies that we want to recognize text in English only.
reader = easyocr.Reader(['en'], gpu=False)

# Output video settings
output_height = 720  # Desired height
output_width = 1280  # Desired width


def upload_video(request):
    if request.method == 'POST' and request.FILES['video']:
        uploaded_video = request.FILES['video']

        # Process the uploaded video
        detection_results, output_video_path = process_uploaded_video(uploaded_video)

        # Get the video names for the processed and cropped license plate images
        video_name = uploaded_video.name
        processed_video_name = os.path.basename(output_video_path)
        license_plate_image_name = processed_video_name.replace(".avi", "_license.png")

        # Save the results to the database
        save_results_to_database(video_name, processed_video_name, license_plate_image_name, detection_results)

        # Check if any DetectVideo records exist. Allows to avoid the error where if no data was saved to the database after uploading a video and the there was no existing data in the database
        if DetectVideo.objects.exists():
            # Get the ID of the latest uploaded video
            video_id = DetectVideo.objects.latest('id').id
            # Redirect to the detection_results view with the video ID as an argument
            # return redirect(reverse('detection_results_video', args=[video_id]))
            return redirect('showIndexHomepage')
        else:
            # Handle the case where there are no DetectVideo records
            print("No DetectVideo records found.")
            return render(request, 'index.html')  # Redirect to the index page or another appropriate page

    return render(request, 'index.html')


def process_uploaded_video(uploaded_video):
    try:
        # Create the output folder for storing processed videspos and cropped grayscale license images, if it doesn't exist
        # output_folder = 'output_results_videos'
        # os.makedirs(output_folder, exist_ok=True)

        # Extract the base filename from the video
        video_filename = uploaded_video.name
        output_video_filename = f'{video_filename.replace(".", "_processed.")}'
        output_video_path = os.path.join("media", output_video_filename)
        # output_video_path = os.path.join("media/output_results_videos", output_video_filename)

        # Create a temporary file to save the uploaded video
        with tempfile.NamedTemporaryFile(delete=False) as temp_video:
            for chunk in uploaded_video.chunks():
                temp_video.write(chunk)

        # Open the temporary video file with cv2.VideoCapture
        cap = cv2.VideoCapture(temp_video.name)

        if not cap.isOpened():
            print("Error: Couldn't open video file.")
            return

        # Create the video writer for the output video
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                            (output_width, output_height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame for display
            frame = cv2.resize(frame, (output_width, output_height))

            # Detect vehicles using YOLOv8
            detections = coco_model(frame)[0]
            detections_ = []

            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection

                if int(class_id) in [2, 3, 5, 7]:
                    detections_.append([x1, y1, x2, y2, score])

            # Track vehicles
            track_ids = mot_tracker.update(np.asarray(detections_))

            # Detect license plates using YOLOv8
            license_plates = license_plate_detector(frame)[0]

            results = {}

            # Process license plates
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                xcar1, ycar1, xcar2, ycar2, car_id = util.get_car(license_plate, track_ids)

                # Crop the license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                # Convert license plate image to grayscale
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # Read license plate number using OCR
                license_plate_text, license_plate_text_score = util.read_license_plate(license_plate_crop_thresh)

                if license_plate_text is not None:
                    # Modify this line to get the base filename from the video_filename
                    base_filename = os.path.splitext(os.path.basename(video_filename))[0]

                    # Save the grayscale license plate image as a PNG file
                    license_plate_image_filename = f'{base_filename}_{car_id}_license.png'
                    license_plate_image_path = os.path.join("output_results_videos", license_plate_image_filename)
                    cv2.imwrite(os.path.join(settings.MEDIA_ROOT, license_plate_image_path), license_plate_crop_gray)

                    results[car_id] = {
                        'vehicle_bbox': [xcar1, ycar1, xcar2, ycar2],
                        'license_plate_bbox': [x1, y1, x2, y2],
                        'license_plate_text': license_plate_text,
                        'license_plate_text_score': license_plate_text_score,
                        'license_plate_cropped_image': license_plate_image_path
                    }

                    # Draw green bounding box for the license plate
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0),
                                  4)  # Green bounding box for license plates

                    # Display OCR license plate text and its background above the license plate bounding box
                    license_plate_display_text = f"License Plate: {license_plate_text}"
                    text_size, _ = cv2.getTextSize(license_plate_display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    text_x = int((x1 + x2 - text_size[0]) / 2)  # Center the OCR license plate text horizontally
                    text_y = int(y1) - text_size[
                        1] - 10  # Position OCR license plate text above the license plate bounding box

                    # Calculate the background rectangle dimensions for the OCR license plate text in red
                    rect_x1 = text_x - 5
                    rect_x2 = text_x + text_size[0] + 5
                    rect_y1 = text_y - text_size[1] - 5
                    rect_y2 = text_y + 5

                    # Draw background rectangle for the OCR license plate text in red
                    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), -1)

                    # Display OCR license plate text in green with smaller font
                    cv2.putText(frame, license_plate_display_text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Display confidence text and its background below the OCR license plate text
                    confidence_text = f"Confidence: {license_plate_text_score:.2f}"
                    text_size, _ = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    text_x = int((x1 + x2 - text_size[0]) / 2)  # Center the confidence text horizontally
                    text_y = int(y2) + text_size[1] + 10  # Position confidence score below the OCR license plate text

                    # Calculate the background rectangle dimensions for the confidence score in red
                    rect_x1 = text_x - 5
                    rect_x2 = text_x + text_size[0] + 5
                    rect_y1 = text_y - text_size[1] - 5
                    rect_y2 = text_y + 5

                    # Draw background rectangle for the confidence score in red
                    cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), -1)

                    # Display confidence score in green with smaller font
                    cv2.putText(frame, confidence_text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Draw green bounding box for the license plate
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

                # Draw green bounding box for the vehicle
                cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0),
                              4)  # Green bounding box for vehicles

            # Add the frame with detections to the output video
            out.write(frame)

            # Display the frame with detections
            cv2.imshow('Video with Detections. Press Q to Exit.', frame)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture and writer objects
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        # os.remove('temp_video.mp4')

        return results, output_video_filename

    finally:
        # Close the temporary video file if it's not already closed
        if not temp_video.closed:
            temp_video.close()
        if not uploaded_video.closed:
            uploaded_video.close()


def save_results_to_database(video_name, processed_video_name, license_plate_image_name, results):
    try:
        # Check if the results dictionary is empty
        if not results:
            print("No results to save.")
            return None

        # Print the contents of the results dictionary for debugging
        print("Results Dictionary Contents:")
        print(results)

        # Define a default value for detect_video (outside the loop)
        detect_video = None

        # Create a list to store all created DetectVideo objects (if there are multiple detections in a single video)
        detect_videos = []

        # Iterate over the results dictionary
        for car_id, data in results.items():
            if car_id == 'license_plates':
                continue  # Skip the 'license_plates' entry
            print(f"Processing result for car_id {car_id}:")
            print(f"Data: {data}")

            detect_video = DetectVideo(
                original_video=video_name,
                processed_video=processed_video_name,
                license_plate_ocr_text=data['license_plate_text'],
                license_plate_conf_score=data['license_plate_text_score'],
                vehicle_bbox=data['vehicle_bbox'],
                license_plate_bbox=data['license_plate_bbox'],
                license_plate_cropped_image=data['license_plate_cropped_image'],
            )
            detect_video.save()

            # Append the created DetectVideo object to the list
            detect_videos.append(detect_video)

        # Check if detect_video is still None (no valid entries in results)
        if detect_video is None:
            print("No valid detections were made. No DetectVideo object was created.")

        # Return the created DetectVideo object (if it was created)
        return detect_videos

    except Exception as e:
        # Handle any exceptions or errors here
        # You can log the error or take appropriate action as needed
        print(f"Error saving DetectVideo: {str(e)}")
        return None


def detection_results_video(request, video_id):
    # Retrieve the DetectImage instance based on image_id
    detect_video = get_object_or_404(DetectVideo, id=video_id)

    # Retrieve the detection results from the database based on video_id
    det_results = DetectVideo.objects.filter(original_video=detect_video.original_video)

    return render(request, 'detection_results_video.html', {'results': det_results})


