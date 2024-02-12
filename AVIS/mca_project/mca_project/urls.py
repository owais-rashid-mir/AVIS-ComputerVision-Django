"""
URL configuration for mca_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.conf.urls.static import static
from django.contrib import admin
from django.conf.urls.static import static
from django.urls import path

from mca_project_app import views, detect_image_views
from mca_project import settings

# from mca_project.mca_project import settings
# from mca_project.mca_project_app import views, detect_image_views
from mca_project_app import detect_video_views, detect_camera_views

urlpatterns = [
    # Setting showIndexHomepage(index.html) as my homepage
    path('', views.showIndexHomepage, name="showIndexHomepage"),
    path('index_homepage', views.showIndexHomepage),

    path('admin/', admin.site.urls),

    # Image Detection URLs
    # Add a URL for uploading and processing images
    path('upload_image/', detect_image_views.upload_image, name='upload_image'),

    # URL pattern for displaying detection results
    # path('detection_results/', detect_image_views.detection_results, name='detection_results'),
    path('detection_results/<int:image_id>/', detect_image_views.detection_results, name='detection_results'),

    path('manage_detect_image', views.manage_detect_image, name="manage_detect_image"),

    path('delete_detect_image/<int:id>/', views.delete_detect_image, name='delete_detect_image'),

    path('view_image_details/<int:id>/', views.view_image_details, name="view_image_details"),


    # Video Detection URLs
    path('upload_video/', detect_video_views.upload_video, name='upload_video'),

    # URL pattern for displaying video detection results
    path('detection_results_video/<int:video_id>/', detect_video_views.detection_results_video, name='detection_results_video'),

    path('manage_detect_video', views.manage_detect_video, name="manage_detect_video"),

    path('delete_detect_video/<int:id>/', views.delete_detect_video, name='delete_detect_video'),

    path('view_video_details/<int:id>/', views.view_video_details, name="view_video_details"),


    # Camera/webcam Detection URLs
    path('detect_camera/', detect_camera_views.detect_camera, name='detect_camera'),
    path('detection_results_camera/<int:camera_id>/', detect_camera_views.detection_results_camera, name='detection_results_camera'),

    path('manage_detect_camera', views.manage_detect_camera, name="manage_detect_camera"),

    path('delete_detect_camera/<int:id>/', views.delete_detect_camera, name='delete_detect_camera'),

    path('view_camera_details/<int:id>/', views.view_camera_details, name="view_camera_details"),



] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) + static(settings.STATIC_URL,
                                                                                         document_root=settings.STATIC_ROOT)
