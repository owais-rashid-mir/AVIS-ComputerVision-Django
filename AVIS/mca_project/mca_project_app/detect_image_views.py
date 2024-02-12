import tempfile
import cv2
from django.core.files.base import ContentFile
from ultralytics import YOLO
import numpy as np
from django.shortcuts import render, redirect, get_object_or_404
from .models import DetectImage
import easyocr
from .sort.sort import Sort
from django.core.files import File
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from . import util  # Import your existing util.py file
import os
from django.urls import reverse


# Initialize the SORT tracker
mot_tracker = Sort()

# Load YOLOv8 model for vehicle detection (coco_model) and license plate detection (license_plate_detector)
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('mca_project_app/models/license_plate_detection_best_training4_70epochs.pt')

# Initialize EasyOCR reader.
# ['en'] specifies that we want to recognize text in English only.
reader = easyocr.Reader(['en'], gpu=False)


# Define the view for uploading an image
def upload_image(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']

        # Create a new instance of the SORT tracker for each image
        mot_tracker = Sort()

        # Process the uploaded image
        detection_results = process_uploaded_image(uploaded_image, mot_tracker)

        # Get the image names for the processed and cropped license plate images
        image_name = uploaded_image.name
        processed_image_name = image_name.replace(".", "_processed.")
        license_plate_image_name = image_name.replace(".", "_license.")

        # Save the results to the database
        save_results_to_database(image_name, processed_image_name, license_plate_image_name, detection_results)

        # Get the ID of the uploaded image
        image_id = DetectImage.objects.latest('id').id  # Assuming 'id' is the primary key field

        # Redirect to the detection_results view with the image ID as an argument
        return redirect(reverse('detection_results', args=[image_id]))

    return render(request, 'index.html')


def process_uploaded_image(uploaded_image, mot_tracker):
    try:
        # Read the image using OpenCV
        image_bytes = uploaded_image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Generate a unique image name for the processed and cropped license plate images
        image_name = uploaded_image.name
        processed_image_name = image_name.replace(".", "_processed.")
        # license_plate_image_name = image_name.replace(".", "_license.")

        # Save the original image with its original name. Images will be saved in the 'media' folder.
        original_image_path = os.path.join("media", image_name)
        with open(original_image_path, 'wb') as original_image_file:
            original_image_file.write(image_bytes)

        # Detect vehicles using YOLOv8
        detections = coco_model(frame)[0]
        detections_ = []

        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection

            # Filter by class IDs for vehicles (you can modify this based on your class IDs). 2, 3, 5, and 7 are the IDs of different vehicles trained on the Coco dataset.
            if int(class_id) in [2, 3, 5, 7]:
                detections_.append([x1, y1, x2, y2, score])

        # Initialize an empty dictionary for results
        results = {}

        # Track vehicles only if there are detections
        if detections_:
            # Track vehicles using the provided tracker instance
            track_ids = mot_tracker.update(np.asarray(detections_))

        if not detections_:
            # No vehicles detected, set an empty list of license plates in results
            results['license_plates'] = []

        # Detect license plates using YOLOv8
        license_plates = license_plate_detector(frame)[0]

        any_detection = False  # Flag to check if any detections were made

        # Process license plates
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            if detections_:
                # Get the corresponding vehicle information. get_car is defined in util.py
                xcar1, ycar1, xcar2, ycar2, car_id = util.get_car(license_plate, track_ids)
            else:
                # Set default values for tracking information when no vehicles are detected
                xcar1, ycar1, xcar2, ycar2, car_id = 0, 0, 0, 0, -1

            # Crop the license plate
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

            # Convert license plate image to grayscale
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

            # Read license plate number using OCR
            license_plate_text, license_plate_text_score = util.read_license_plate(license_plate_crop_thresh)

            if license_plate_text:
                results[car_id] = {
                    'vehicle_bbox': [xcar1, ycar1, xcar2, ycar2],
                    'license_plate_bbox': [x1, y1, x2, y2],
                    'license_plate_text': license_plate_text,
                    'license_plate_text_score': license_plate_text_score
                }

                any_detection = True  # Set the flag to True if any detections were made

                # Save the cropped license plate image with the new name
                license_plate_image_name = f'{image_name}_{car_id}_license.png'
                license_plate_image_path = os.path.join("media", license_plate_image_name)
                cv2.imwrite(license_plate_image_path, license_plate_crop_gray)

                # Draw bounding box for the vehicle
                xcar1, ycar1, xcar2, ycar2 = map(int, results[car_id]['vehicle_bbox'])
                xcar1 = max(0, xcar1)
                ycar1 = max(0, ycar1)
                xcar2 = min(frame.shape[1], xcar2)
                ycar2 = min(frame.shape[0], ycar2)

                # Increase bounding box thickness
                cv2.rectangle(frame, (xcar1, ycar1), (xcar2, ycar2), (0, 255, 0), 4)

                # Draw bounding box for the license plate
                x1, y1, x2, y2 = map(int, results[car_id]['license_plate_bbox'])
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                # Define bounding box color and thickness
                bounding_box_color = (0, 255, 0)  # Green color
                bounding_box_thickness = 6

                # Draw bounding box for the license plate
                cv2.rectangle(frame, (x1, y1), (x2, y2), bounding_box_color, bounding_box_thickness)

                # Display OCR license plate text
                license_plate_display_text = f"License Plate: {license_plate_text}"
                text_size, _ = cv2.getTextSize(license_plate_display_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                text_x = int((x1 + x2 - text_size[0]) / 2)  # Center the OCR license plate text horizontally
                text_y = int(y1) - text_size[
                    1] - 20  # Position OCR license plate text above the license plate bounding box

                # Calculate the background rectangle dimensions for the OCR license plate text
                rect_x1 = text_x - 10
                rect_x2 = text_x + text_size[0] + 10
                rect_y1 = text_y - text_size[1] - 10
                rect_y2 = text_y + 10

                # Draw background rectangle for the OCR license plate text
                cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), -1)  # Filled background

                # Display OCR license plate text
                cv2.putText(frame, license_plate_display_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)  # Display OCR license plate text in green

                # Display confidence score
                confidence_text = f"Confidence: {license_plate_text_score:.2f}"
                text_size, _ = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                text_x = int((x1 + x2 - text_size[0]) / 2)  # Center the confidence text horizontally
                text_y = int(y2) + text_size[1] + 20  # Position confidence score below the OCR license plate text

                # Calculate the background rectangle dimensions for the confidence score
                rect_x1 = text_x - 10
                rect_x2 = text_x + text_size[0] + 10
                rect_y1 = text_y - text_size[1] - 10
                rect_y2 = text_y + 10

                # Draw background rectangle for the confidence score
                cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), -1)  # Filled background

                # Display confidence score
                cv2.putText(frame, confidence_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)  # Display confidence score in green

        if not any_detection:
            # No license plates detected, set an empty list of license plates in results
            results['license_plates'] = []

        # Save the processed image with the new name
        processed_image_path = os.path.join("media", processed_image_name)
        cv2.imwrite(processed_image_path, frame)

        return results

    finally:
        # Close the uploaded file if it's not already closed
        if not uploaded_image.closed:
            uploaded_image.close()


# Function to save results to the database
def save_results_to_database(image_name, processed_image_name, license_plate_image_name, results):
    try:
        # Check if the results dictionary is empty
        if not results:
            print("No results to save.")
            return None

        # Print the contents of the results dictionary for debugging
        print("Results Dictionary Contents:")
        print(results)

        # Define a default value for detect_image (outside the loop)
        detect_image = None

        # Create a list to store all created DetectImage objects (if there are multiple detections in a single image)
        detect_images = []

        # Iterate over the results dictionary
        for car_id, data in results.items():
            if car_id == 'license_plates':
                continue  # Skip the 'license_plates' entry
            print(f"Processing result for car_id {car_id}:")
            print(f"Data: {data}")

            # Appending the car_id with license plates in the same way they are stored in "media" folder so that they can be displayed on the detection_results.html
            license_plate_image_name = f'{image_name}_{car_id}_license.png'

            detect_image = DetectImage(
                original_image=image_name,
                processed_image=processed_image_name,
                license_plate_ocr_text=data['license_plate_text'],
                license_plate_conf_score=data['license_plate_text_score'],
                vehicle_bbox=data['vehicle_bbox'],
                license_plate_bbox=data['license_plate_bbox'],
                license_plate_cropped_image=license_plate_image_name
            )
            detect_image.save()

            # Append the created DetectImage object to the list
            detect_images.append(detect_image)

        # Check if detect_image is still None (no valid entries in results)
        if detect_image is None:
            print("No valid detections were made. No DetectImage object was created.")

        # Return the list of created DetectImage objects
        return detect_images

    except Exception as e:
        # Handle any exceptions or errors here
        # You can log the error or take appropriate action as needed
        print(f"Error saving DetectImage: {str(e)}")
        return None


# Define a view for displaying detection results
def detection_results(request, image_id):
    # Retrieve the DetectImage instance based on image_id
    detect_image = get_object_or_404(DetectImage, id=image_id)

    # Retrieve all relevant detection results for the given image ID
    detection_results = DetectImage.objects.filter(original_image=detect_image.original_image)

    return render(request, 'detection_results.html', {'results': detection_results})

