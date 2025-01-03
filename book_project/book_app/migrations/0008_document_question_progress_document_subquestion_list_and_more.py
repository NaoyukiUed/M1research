# Generated by Django 5.1.2 on 2024-10-29 06:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('book_app', '0007_document_question_num_alter_document_question_list'),
    ]

    operations = [
        migrations.AddField(
            model_name='document',
            name='question_progress',
            field=models.IntegerField(blank=True, default=0),
        ),
        migrations.AddField(
            model_name='document',
            name='subquestion_list',
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name='document',
            name='subquestion_num',
            field=models.IntegerField(blank=True, default=0),
        ),
    ]
