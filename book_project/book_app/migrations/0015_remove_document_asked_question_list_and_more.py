# Generated by Django 5.1.4 on 2024-12-10 10:13

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('book_app', '0014_document_question_stack'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='document',
            name='asked_question_list',
        ),
        migrations.RemoveField(
            model_name='document',
            name='question_list',
        ),
        migrations.RemoveField(
            model_name='document',
            name='question_num',
        ),
        migrations.RemoveField(
            model_name='document',
            name='question_progress',
        ),
        migrations.RemoveField(
            model_name='document',
            name='question_stack',
        ),
        migrations.RemoveField(
            model_name='document',
            name='subquestion_list',
        ),
        migrations.RemoveField(
            model_name='document',
            name='subquestion_num',
        ),
        migrations.RemoveField(
            model_name='document',
            name='text_summary',
        ),
    ]
