# Generated by Django 5.1.2 on 2024-10-24 02:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('book_app', '0004_interaction'),
    ]

    operations = [
        migrations.AddField(
            model_name='document',
            name='point_1',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
        migrations.AddField(
            model_name='document',
            name='point_2',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
        migrations.AddField(
            model_name='document',
            name='point_3',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]
