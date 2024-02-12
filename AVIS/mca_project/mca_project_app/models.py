from django.db import models


class DetectImage(models.Model):
    # id will be automatically added by Django, no need to specify it here.
    original_image = models.ImageField(upload_to='image_output')
    processed_image = models.ImageField(upload_to='image_output')
    license_plate_ocr_text = models.CharField(max_length=255)
    license_plate_conf_score = models.FloatField()
    vehicle_bbox = models.CharField(max_length=255)
    license_plate_bbox = models.CharField(max_length=255)
    license_plate_cropped_image = models.ImageField(upload_to='image_output')
    created_at = models.DateTimeField(auto_now_add=True)
    objects = models.Manager()


class DetectVideo(models.Model):
    original_video = models.FileField(upload_to='output_results_videos', verbose_name='Original Video File')
    processed_video = models.FileField(upload_to='output_results_videos', verbose_name='Processed Video File')
    license_plate_ocr_text = models.CharField(max_length=255, verbose_name='License Plate OCR Text')
    license_plate_conf_score = models.FloatField(verbose_name='License Plate Confidence Score')
    vehicle_bbox = models.CharField(max_length=255, verbose_name='Vehicle Bounding Box')
    license_plate_bbox = models.CharField(max_length=255, verbose_name='License Plate Bounding Box')
    license_plate_cropped_image = models.ImageField(upload_to='output_results_videos', verbose_name='Cropped License Plate Image')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Created At')
    objects = models.Manager()

    def __str__(self):
        return f"Original Video: {self.original_video.name}, Processed Video: {self.processed_video.name}, Created at: {self.created_at}"

    class Meta:
        verbose_name = 'Video Detection Result'
        verbose_name_plural = 'Video Detection Results'


class DetectCamera(models.Model):
    original_video = models.FileField(upload_to='output_results_camera', verbose_name='Original Video File')
    processed_video = models.FileField(upload_to='output_results_camera', verbose_name='Processed Video File')
    license_plate_ocr_text = models.CharField(max_length=255, verbose_name='License Plate OCR Text')
    license_plate_conf_score = models.FloatField(verbose_name='License Plate Confidence Score')
    vehicle_bbox = models.CharField(max_length=255, verbose_name='Vehicle Bounding Box')
    license_plate_bbox = models.CharField(max_length=255, verbose_name='License Plate Bounding Box')
    license_plate_cropped_image = models.ImageField(upload_to='output_results_camera', verbose_name='Cropped License Plate Image')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Created At')
    objects = models.Manager()

