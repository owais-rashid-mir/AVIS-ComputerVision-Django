import os
from django.conf import settings
from django.http import Http404
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages


# Create your views here.
from mca_project_app.models import DetectImage, DetectVideo, DetectCamera


def showIndexHomepage(request):
    return render(request, "index.html")


# Manage Image Detections
def manage_detect_image(request):
    # Reading all DetectImage table data by calling DetectImage.objects.all()
    results = DetectImage.objects.all()
    return render(request, "manage_detect_image_template.html", {"results": results})


# Delete detect_image instances
def delete_detect_image(request, id):
    if request.method == "POST":
        try:
            results = DetectImage.objects.get(id=id)
        except DetectImage.DoesNotExist:
            raise Http404("Data not found")

        try:
            # Get the path to the original_image
            original_image_path = os.path.join(settings.MEDIA_ROOT, str(results.original_image))
            # Check if the original_image file exists and is a file (not a directory)
            if os.path.isfile(original_image_path):
                os.remove(original_image_path)

            # Get the path to the processed_image
            processed_image_path = os.path.join(settings.MEDIA_ROOT, str(results.processed_image))
            # Check if the processed_image file exists and is a file (not a directory)
            if os.path.isfile(processed_image_path):
                os.remove(processed_image_path)

            # Get the path to the license_plate_cropped_image
            license_plate_cropped_image_path = os.path.join(settings.MEDIA_ROOT, str(results.license_plate_cropped_image))
            # Check if the license_plate_cropped_image file exists and is a file (not a directory)
            if os.path.isfile(license_plate_cropped_image_path):
                os.remove(license_plate_cropped_image_path)

            # Delete the employee instance
            results.delete()

            messages.success(request, "Image Detection Data And Associated Details Deleted Successfully.")
        except Exception as e:
            messages.error(request, f"Failed To Delete Image Detection Data: {str(e)}")

        return redirect("manage_detect_image")  # Redirect to the list of employees page


# View image details of DetectImage table instance
def view_image_details(request, id):
    # Fetch the specific instance using the id parameter
    # get_object_or_404 : used to retrieve a single object from the database based on certain criteria, and if the object doesn't exist, it raises an Http404 exception.
    results = get_object_or_404(DetectImage, id=id)

    context = {
        'results': results,
    }
    return render(request, 'view_image_details.html', context)


# ------------------------------------------------------------------------
# Manage Video Detections
def manage_detect_video(request):
    # Reading all DetectVideo table data by calling DetectVideo.objects.all()
    results = DetectVideo.objects.all()
    return render(request, "manage_detect_video_template.html", {"results": results})


# Delete detect_video instances
def delete_detect_video(request, id):
    if request.method == "POST":
        try:
            results = DetectVideo.objects.get(id=id)
        except DetectVideo.DoesNotExist:
            raise Http404("Data not found")

        try:
            # Get the path to the processed_video
            processed_video_path = os.path.join(settings.MEDIA_ROOT, str(results.processed_video))
            # Check if the processed_video file exists and is a file (not a directory)
            if os.path.isfile(processed_video_path):
                os.remove(processed_video_path)

            # Get the path to the license_plate_cropped_image
            license_plate_cropped_image_path = os.path.join(settings.MEDIA_ROOT, str(results.license_plate_cropped_image))
            # Check if the license_plate_cropped_image file exists and is a file (not a directory)
            if os.path.isfile(license_plate_cropped_image_path):
                os.remove(license_plate_cropped_image_path)

            # Delete the instance
            results.delete()

            messages.success(request, "Video Detection Data And Associated Details Deleted Successfully.")
        except Exception as e:
            messages.error(request, f"Failed To Delete Video Detection Data: {str(e)}")

        return redirect("manage_detect_video")


# View image details of DetectVideo table instance
def view_video_details(request, id):
    # Fetch the specific instance using the id parameter
    # get_object_or_404 : used to retrieve a single object from the database based on certain criteria, and if the object doesn't exist, it raises an Http404 exception.
    results = get_object_or_404(DetectVideo, id=id)

    context = {
        'results': results,
    }
    return render(request, 'view_video_details.html', context)


# ------------------------------------------------------------------------
# Manage Video Detections
def manage_detect_camera(request):
    # Reading all DetectCamera table data by calling DetectCamera.objects.all()
    results = DetectCamera.objects.all()
    return render(request, "manage_detect_camera_template.html", {"results": results})


# Delete detect_video instances
def delete_detect_camera(request, id):
    if request.method == "POST":
        try:
            results = DetectCamera.objects.get(id=id)
        except DetectCamera.DoesNotExist:
            raise Http404("Data not found")

        try:
            # Get the path to the processed_video
            processed_video_path = os.path.join(settings.MEDIA_ROOT, str(results.processed_video))
            # Check if the processed_video file exists and is a file (not a directory)
            if os.path.isfile(processed_video_path):
                os.remove(processed_video_path)

            # Get the path to the license_plate_cropped_image
            license_plate_cropped_image_path = os.path.join(settings.MEDIA_ROOT, str(results.license_plate_cropped_image))
            # Check if the license_plate_cropped_image file exists and is a file (not a directory)
            if os.path.isfile(license_plate_cropped_image_path):
                os.remove(license_plate_cropped_image_path)

            # Delete the instance
            results.delete()

            messages.success(request, "Camera Detection Data And Associated Details Deleted Successfully.")
        except Exception as e:
            messages.error(request, f"Failed To Delete Camera Detection Data: {str(e)}")

        return redirect("manage_detect_camera")


# View image details of DetectVideo table instance
def view_camera_details(request, id):
    # Fetch the specific instance using the id parameter
    results = get_object_or_404(DetectCamera, id=id)

    context = {
        'results': results,
    }
    return render(request, 'view_camera_details.html', context)