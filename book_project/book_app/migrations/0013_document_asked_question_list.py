# Generated by Django 5.1.2 on 2024-11-07 04:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('book_app', '0012_remove_document_toc_document_structed_toc'),
    ]

    operations = [
        migrations.AddField(
            model_name='document',
            name='asked_question_list',
            field=models.JSONField(blank=True, default=list),
        ),
    ]