# Generated by Django 5.0 on 2024-01-29 03:11

import django.contrib.gis.db.models.fields
import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Admin1Code',
            fields=[
                ('status', models.IntegerField(choices=[(0, 'Disabled'), (100, 'Enabled'), (500, 'Archived')], default=100)),
                ('geonameid', models.PositiveIntegerField(primary_key=True, serialize=False)),
                ('code', models.CharField(max_length=20)),
                ('name', models.CharField(max_length=200)),
            ],
            options={
                'ordering': ['country', 'name'],
            },
        ),
        migrations.CreateModel(
            name='Continent',
            fields=[
                ('status', models.IntegerField(choices=[(0, 'Disabled'), (100, 'Enabled'), (500, 'Archived')], default=100)),
                ('code', models.CharField(max_length=2, primary_key=True, serialize=False)),
                ('name_en', models.CharField(blank=True, max_length=255, null=True, verbose_name='continent name')),
                ('geonameid', models.PositiveIntegerField()),
            ],
            options={
                'verbose_name': 'continent',
                'verbose_name_plural': 'continents',
                'ordering': ['code'],
            },
        ),
        migrations.CreateModel(
            name='Currency',
            fields=[
                ('status', models.IntegerField(choices=[(0, 'Disabled'), (100, 'Enabled'), (500, 'Archived')], default=100)),
                ('code', models.CharField(max_length=3, primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=200)),
            ],
            options={
                'verbose_name_plural': 'Currencies',
                'ordering': ['name'],
            },
        ),
        migrations.CreateModel(
            name='FeatureClass',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('f_class', models.CharField(blank=True, max_length=1, null=True, unique=True, verbose_name='feature class')),
                ('name_en', models.CharField(blank=True, max_length=255, null=True, verbose_name='feature class name')),
                ('description_en', models.CharField(blank=True, max_length=255, null=True, verbose_name='feature class description')),
            ],
            options={
                'verbose_name': 'geonames feature class',
                'verbose_name_plural': 'geonames feature classes',
                'ordering': ['f_class'],
            },
        ),
        migrations.CreateModel(
            name='GeonamesUpdate',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('update_date', models.DateField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='Language',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('status', models.IntegerField(choices=[(0, 'Disabled'), (100, 'Enabled'), (500, 'Archived')], default=100)),
                ('name', models.CharField(max_length=255)),
                ('iso_639_1', models.CharField(blank=True, max_length=2, null=True, verbose_name='ISO 639-1 code')),
                ('iso_639_2', models.CharField(blank=True, max_length=10, null=True, verbose_name='ISO 639-2 code')),
                ('iso_639_3', models.CharField(blank=True, max_length=3, null=True, verbose_name='ISO 639-3 code')),
            ],
            options={
                'ordering': ['name'],
            },
        ),
        migrations.CreateModel(
            name='Country',
            fields=[
                ('status', models.IntegerField(choices=[(0, 'Disabled'), (100, 'Enabled'), (500, 'Archived')], default=100)),
                ('code', models.CharField(max_length=2, primary_key=True, serialize=False, verbose_name='ISO code')),
                ('name', models.CharField(db_index=True, max_length=200, unique=True)),
                ('code_iso3', models.CharField(blank=True, max_length=3, null=True, verbose_name='ISO 3 code')),
                ('iso_numeric', models.CharField(blank=True, max_length=3, null=True, verbose_name='ISO Numeric code')),
                ('fips', models.CharField(blank=True, max_length=2, null=True, verbose_name='FIPS code')),
                ('capital', models.CharField(blank=True, max_length=255, null=True, verbose_name='Country capital')),
                ('area_sq_km', models.FloatField(blank=True, null=True, verbose_name='Area in square kilometers')),
                ('population', models.PositiveIntegerField(blank=True, null=True, verbose_name='Population')),
                ('tld', models.CharField(blank=True, max_length=3, null=True, verbose_name='Top-level domain')),
                ('currency_code', models.CharField(blank=True, max_length=3, null=True, verbose_name='Currency code')),
                ('currency_name', models.CharField(blank=True, max_length=255, null=True, verbose_name='Currency name')),
                ('phone', models.CharField(blank=True, max_length=255, null=True, verbose_name='Phone code')),
                ('postal_code_format', models.CharField(blank=True, max_length=255, null=True, verbose_name='Postal code format')),
                ('postal_code_regex', models.CharField(blank=True, max_length=255, null=True, verbose_name='Postal code regular expression (regex)')),
                ('geonameid', models.CharField(blank=True, max_length=10, null=True, verbose_name='Geonames ID')),
                ('neighbours_codes', models.CharField(blank=True, max_length=255, null=True, verbose_name='neighbours codes')),
                ('equivalent_fips_code', models.CharField(blank=True, max_length=255, null=True, verbose_name='Equivalent FIPS code')),
                ('continent', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='geonames.continent', verbose_name='Continent')),
                ('neighbours', models.ManyToManyField(blank=True, db_index=True, to='geonames.country', verbose_name='neighbours')),
                ('currency', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='country_set', to='geonames.currency')),
                ('languages', models.ManyToManyField(related_name='country_set', to='geonames.language')),
            ],
            options={
                'verbose_name_plural': 'Countries',
                'ordering': ['name'],
            },
        ),
        migrations.CreateModel(
            name='Admin2Code',
            fields=[
                ('status', models.IntegerField(choices=[(0, 'Disabled'), (100, 'Enabled'), (500, 'Archived')], default=100)),
                ('geonameid', models.PositiveIntegerField(primary_key=True, serialize=False)),
                ('code', models.CharField(max_length=30)),
                ('name', models.CharField(max_length=200)),
                ('slug', models.CharField(blank=True, db_index=True, max_length=35, null=True)),
                ('admin1', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='admin2_set', to='geonames.admin1code')),
                ('country', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='admin2_set', to='geonames.country')),
            ],
            options={
                'ordering': ['country', 'admin1', 'name'],
                'unique_together': {('country', 'admin1', 'name')},
            },
        ),
        migrations.AddField(
            model_name='admin1code',
            name='country',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='admin1_set', to='geonames.country'),
        ),
        migrations.CreateModel(
            name='FeatureClassAndCode',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('f_code', models.CharField(blank=True, max_length=10, null=True, unique=True, verbose_name='feature code')),
                ('f_class_and_code', models.CharField(blank=True, max_length=11, null=True, verbose_name='full feature class and code')),
                ('name_en', models.CharField(blank=True, max_length=255, null=True, verbose_name='feature code name')),
                ('description_en', models.CharField(blank=True, max_length=255, null=True, verbose_name='feature code description')),
                ('f_class', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='includes_feature_codes', to='geonames.featureclass', to_field='f_class', verbose_name='feature class')),
            ],
            options={
                'verbose_name': 'geonames feature code',
                'verbose_name_plural': 'geonames feature codes',
                'ordering': ['f_class', 'f_code'],
            },
        ),
        migrations.CreateModel(
            name='Locality',
            fields=[
                ('status', models.IntegerField(choices=[(0, 'Disabled'), (100, 'Enabled'), (500, 'Archived')], default=100)),
                ('geonameid', models.PositiveIntegerField(primary_key=True, serialize=False)),
                ('name', models.CharField(db_index=True, max_length=200)),
                ('name_ascii', models.CharField(db_index=True, max_length=200)),
                ('long_name', models.CharField(max_length=200)),
                ('admin3', models.CharField(blank=True, max_length=20, null=True, verbose_name='code for third level administrative division')),
                ('admin4', models.CharField(blank=True, max_length=20, null=True, verbose_name='code for fourth level administrative division')),
                ('population', models.PositiveIntegerField(blank=True, null=True, verbose_name='population')),
                ('elevation', models.IntegerField(blank=True, null=True, verbose_name='elevation in meters')),
                ('lat', models.DecimalField(decimal_places=6, max_digits=9, null=True)),
                ('lon', models.DecimalField(decimal_places=6, max_digits=9, null=True)),
                ('point', django.contrib.gis.db.models.fields.PointField(srid=4326)),
                ('modification_date', models.DateField()),
                ('slug', models.CharField(blank=True, db_index=True, max_length=35, null=True)),
                ('admin1', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='locality_set', to='geonames.admin1code')),
                ('admin2', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='locality_set', to='geonames.admin2code')),
                ('alt_country', models.ManyToManyField(blank=True, related_name='locality_alt_set', to='geonames.country', verbose_name='alternate country codes')),
                ('country', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='locality_set', to='geonames.country')),
                ('feature_class', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='geonames.featureclass', to_field='f_class', verbose_name='feature class')),
                ('feature_code', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='geonames.featureclassandcode', to_field='f_code', verbose_name='feature code')),
            ],
            options={
                'verbose_name_plural': 'Localities',
                'ordering': ['country', 'admin1', 'admin2', 'long_name'],
            },
        ),
        migrations.CreateModel(
            name='AlternateName',
            fields=[
                ('status', models.IntegerField(choices=[(0, 'Disabled'), (100, 'Enabled'), (500, 'Archived')], default=100)),
                ('alternatenameid', models.PositiveIntegerField(primary_key=True, serialize=False)),
                ('name', models.CharField(db_index=True, max_length=200)),
                ('isolanguage', models.CharField(blank=True, help_text='iso 639 language code 2- or 3-characters, optionally followed by a hyphen and a countrycode for country specific variants (ex:zh-CN) or by a variant name (ex: zh-Hant); 4-characters "post" for postal codes and "iata","icao" and "faac" for airport codes, fr_1793 for French Revolution names,  abbr for abbreviation, link to a website (mostly to wikipedia), wkdt for the wikidataid', max_length=10, null=True, verbose_name='ISO 639 language code (or other, see below')),
                ('is_preferred_name', models.BooleanField(help_text='is this alternate name is an official/preferred name?', null=True, verbose_name='is preferred name?')),
                ('is_short_name', models.BooleanField(help_text='is this is a short name like California for State of California?', null=True, verbose_name='is short name?')),
                ('is_colloquial', models.BooleanField(help_text='is this alternate name a colloquial or slang term? Example: Big Apple for New York.', null=True, verbose_name='is preferred name?')),
                ('is_historic', models.BooleanField(help_text='is this alternate name historic and was used in the past? Example Bombay for Mumbai', null=True, verbose_name='is historic name?')),
                ('from_period', models.CharField(blank=True, max_length=100, null=True, verbose_name='from period when the name was used')),
                ('to_period', models.CharField(blank=True, max_length=10, null=True, verbose_name='to period when the name was used')),
                ('locality', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='alternatename_set', to='geonames.locality')),
            ],
            options={
                'ordering': ['locality__pk', 'name'],
            },
        ),
        migrations.CreateModel(
            name='LocalityHierarchy',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('hierarchy_type', models.CharField(blank=True, max_length=255, null=True, verbose_name='hierarchy type')),
                ('child', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='has_child', to='geonames.locality')),
                ('parent', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='is_parent_of', to='geonames.locality')),
            ],
            options={
                'verbose_name': 'geonames hierarchy',
                'verbose_name_plural': 'geonames hierarchies',
                'unique_together': {('parent', 'child', 'hierarchy_type')},
            },
        ),
        migrations.AddField(
            model_name='locality',
            name='children',
            field=models.ManyToManyField(blank=True, through='geonames.LocalityHierarchy', to='geonames.locality'),
        ),
        migrations.CreateModel(
            name='Postcode',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('postal_code', models.CharField(db_index=True, max_length=20)),
                ('place_name', models.CharField(max_length=180)),
                ('admin_name1', models.CharField(blank=True, max_length=100, null=True, verbose_name='state')),
                ('admin_code1', models.CharField(blank=True, max_length=20, null=True, verbose_name='state')),
                ('admin_name2', models.CharField(blank=True, max_length=100, null=True, verbose_name='county/province')),
                ('admin_code2', models.CharField(blank=True, max_length=20, null=True, verbose_name='county/province')),
                ('admin_name3', models.CharField(blank=True, max_length=100, null=True, verbose_name='community')),
                ('admin_code3', models.CharField(blank=True, max_length=20, null=True, verbose_name='community')),
                ('lat', models.DecimalField(decimal_places=6, max_digits=9, null=True)),
                ('lon', models.DecimalField(decimal_places=6, max_digits=9, null=True)),
                ('point', django.contrib.gis.db.models.fields.PointField(srid=4326)),
                ('accuracy', models.IntegerField(blank=True, null=True)),
                ('country', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='postcode_set', to='geonames.country')),
            ],
        ),
        migrations.CreateModel(
            name='Timezone',
            fields=[
                ('status', models.IntegerField(choices=[(0, 'Disabled'), (100, 'Enabled'), (500, 'Archived')], default=100)),
                ('name', models.CharField(max_length=200, primary_key=True, serialize=False)),
                ('gmt_offset', models.DecimalField(decimal_places=2, max_digits=4)),
                ('dst_offset', models.DecimalField(decimal_places=2, max_digits=4)),
                ('raw_offset', models.DecimalField(decimal_places=2, max_digits=4)),
                ('country', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='geonames.country', verbose_name='Country code')),
            ],
            options={
                'ordering': ['gmt_offset', 'name'],
            },
        ),
        migrations.AddField(
            model_name='locality',
            name='timezone',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='locality_set', to='geonames.timezone'),
        ),
        migrations.AlterUniqueTogether(
            name='admin1code',
            unique_together={('country', 'name')},
        ),
    ]
