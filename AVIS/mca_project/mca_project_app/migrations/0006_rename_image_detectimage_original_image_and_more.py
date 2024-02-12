# Generated by Django 4.2.4 on 2023-10-05 12:14

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('mca_project_app', '0005_remove_detectimage_detected_image_and_more'),
    ]

    operations = [
        migrations.RenameField(
            model_name='detectimage',
            old_name='image',
            new_name='original_image',
        ),
        migrations.AddField(
            model_name='detectimage',
            name='processed_image',
            field=models.ImageField(default=django.utils.timezone.now, upload_to='image_output'),
            preserve_default=False,
        ),
    ]
