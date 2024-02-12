import os
import tempfile
import cv2
import numpy as np
from django.core.files.base import ContentFile
from django.shortcuts import render, redirect, get_object_or_404
from .models import DetectCamera
from ultralytics import YOLO
from .sort.sort import Sort
import easyocr
from . import util  # Import your existing util.py file
from django.urls import reverse
from mca_project import settings
import datetime


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


def detect_camera(request):
    # Open the laptop's camera
    cap = cv2.VideoCapture(0)  # Use the default camera (0)

    if not cap.isOpened():
        return render(request, 'camera_error_msg.html', {'error_message': "Couldn't access the camera."})

    # Generate a unique processed video name based on the current date and time
    current_datetime = datetime.datetime.now()
    processed_video_name = f'processed_camera_{current_datetime.strftime("%Y-%m-%d_%H-%M-%S")}.avi'

    # Define the VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    processed_video_path = os.path.join(settings.MEDIA_ROOT, 'output_results_camera', processed_video_name)
    out = cv2.VideoWriter(processed_video_path, fourcc, 20.0, (output_width, output_height))

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

        # Check if there are valid detections
        if detections_:
            # Track vehicles
            track_ids = mot_tracker.update(np.asarray(detections_))
        else:
            track_ids = []

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
                # Save the grayscale license plate image as a PNG file in the media folder and database
                base_filename = "camera_capture"  # Change this as needed
                # Generate a unique license plate name based on the current date and time
                license_plate_image_filename = f'{base_filename}_{car_id}_license_{current_datetime.strftime("%Y-%m-%d_%H-%M-%S")}.png'
                license_plate_image_path = os.path.join(settings.MEDIA_ROOT, 'output_results_camera',
                                                        license_plate_image_filename)
                cv2.imwrite(license_plate_image_path, license_plate_crop_gray)


                results[car_id] = {
                    'vehicle_bbox': [xcar1, ycar1, xcar2, ycar2],
                    'license_plate_bbox': [x1, y1, x2, y2],
                    'license_plate_text': license_plate_text,
                    'license_plate_text_score': license_plate_text_score,
                    'license_plate_cropped_image': os.path.join('output_results_camera', license_plate_image_filename),
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

                # Write the frame to the video
                out.write(frame)

                # Save the results to the database
                save_results_to_database(None, processed_video_name, license_plate_image_filename, results)

            # Draw green bounding box for the license plate
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)

            # Draw green bounding box for the vehicle
            cv2.rectangle(frame, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 255, 0),
                          4)  # Green bounding box for vehicles

        # Display the frame with detections
        cv2.imshow('Camera Detection. Press Q to Exit.', frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and the VideoWriter
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # return render(request, 'detection_results_camera.html', {'results': results})
    # return redirect('manage_detect_camera')
    return redirect('showIndexHomepage')


def save_results_to_database(video_name, processed_video_name, license_plate_image_name, results):
    try:
        # Check if the results dictionary is empty
        if not results:
            print("No results to save.")
            return None

        # Print the contents of the results dictionary for debugging
        print("Results Dictionary Contents:")
        print(results)

        # Define a default value for detect_camera (outside the loop)
        detect_camera = None

        # Create a list to store all created DetectCamera objects (if there are multiple detections in a single camera live-feed session)
        detect_cameras = []

        # Iterate over the results dictionary
        for car_id, data in results.items():
            if car_id == 'license_plates':
                continue  # Skip the 'license_plates' entry
            print(f"Processing result for car_id {car_id}:")
            print(f"Data: {data}")

            detect_camera = DetectCamera(
                original_video=video_name,
                processed_video=processed_video_name,
                license_plate_ocr_text=data['license_plate_text'],
                license_plate_conf_score=data['license_plate_text_score'],
                vehicle_bbox=data['vehicle_bbox'],
                license_plate_bbox=data['license_plate_bbox'],
                license_plate_cropped_image=data['license_plate_cropped_image'],
            )
            detect_camera.save()

            # Append the created DetectCamera object to the list
            detect_cameras.append(detect_camera)

        # Check if detect_camera is still None (no valid entries in results)
        if detect_camera is None:
            print("No valid detections were made. No DetectCamera object was created.")

        # Return the created DetectCamera object (if it was created)
        return detect_cameras

    except Exception as e:
        # Handle any exceptions or errors here
        # You can log the error or take appropriate action as needed
        print(f"Error saving DetectCamera: {str(e)}")
        return None


def detection_results_camera(request, camera_id):
    # Retrieve the DetectCamera instance based on camera_id
    detect_camera = get_object_or_404(DetectCamera, id=camera_id)

    # Retrieve the detection results from the database based on camera_id. original_video is defined in DetectCamera table of DB.
    det_results = DetectCamera.objects.filter(original_video=detect_camera.original_video)

    return render(request, 'detection_results_camera.html', {'results': det_results})


