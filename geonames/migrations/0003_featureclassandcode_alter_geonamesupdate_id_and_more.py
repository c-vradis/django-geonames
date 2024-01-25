# Generated by Django 5.0 on 2024-01-25 22:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('geonames', '0002_slugs_postcodes'),
    ]

    operations = [
        migrations.CreateModel(
            name='FeatureClassAndCode',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('f_class', models.CharField(blank=True, max_length=1, null=True, verbose_name='feature class')),
                ('f_code', models.CharField(blank=True, max_length=10, null=True, verbose_name='feature code')),
                ('f_class_and_code', models.CharField(blank=True, max_length=11, null=True, verbose_name='full feature class and code')),
                ('name_en', models.CharField(blank=True, max_length=255, null=True, verbose_name='feature code name')),
                ('description_en', models.CharField(blank=True, max_length=255, null=True, verbose_name='feature code description')),
            ],
            options={
                'verbose_name': 'geonames feature code',
                'verbose_name_plural': 'geonames feature codes',
                'ordering': ['f_class', 'f_code'],
            },
        ),
        migrations.AlterField(
            model_name='geonamesupdate',
            name='id',
            field=models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID'),
        ),
        migrations.AlterField(
            model_name='postcode',
            name='id',
            field=models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID'),
        ),
    ]
