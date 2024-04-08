import datetime
import glob
import os
import shutil
import sys
import tempfile
import traceback

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction
from django.db.models import Count

from geonames.models import (Admin1Code, Admin2Code, AlternateName, Continent, Country,
                             Currency, GeonamesUpdate, Language, Locality, LocalityHierarchy,
                             Postcode, Timezone, FeatureClass, FeatureClassAndCode)
from geonames.models import GIS_LIBRARIES

if GIS_LIBRARIES:
    from django.contrib.gis.geos import Point

FILES = [
    'https://download.geonames.org/export/dump/featureCodes_en.txt',
    'https://download.geonames.org/export/dump/timeZones.txt',
    'https://download.geonames.org/export/dump/iso-languagecodes.txt',
    'https://download.geonames.org/export/dump/countryInfo.txt',
    'https://download.geonames.org/export/dump/admin1CodesASCII.txt',
    'https://download.geonames.org/export/dump/admin2Codes.txt',
    'https://download.geonames.org/export/dump/cities500.zip',
    #'https://download.geonames.org/export/dump/alternateNames.zip',
    #'https://download.geonames.org/export/dump/alternatenames/CY.zip',
    # postcodes
    'https://download.geonames.org/export/dump/hierarchy.zip',
    #'https://download.geonames.org/export/zip/allCountries.zip',
    #'https://download.geonames.org/export/dump/CY.zip',
    'https://download.geonames.org/export/zip/GB_full.csv.zip',
]

# See http://www.geonames.org/export/codes.html
city_types = ['PPL', 'PPLA', 'PPLC', 'PPLA2', 'PPLA3', 'PPLA4', 'PPLG']
geo_models = [
    Timezone, 
    Language, 
    Continent, 
    Country, 
    Currency,
    Admin1Code, 
    Admin2Code, 
    Locality, 
    LocalityHierarchy, 
    AlternateName, 
    Postcode, 
    FeatureClass, 
    FeatureClassAndCode
    ]

class Command(BaseCommand):
    help = "Geonames import command."
    if settings.DEBUG:
        download_dir = os.path.join(os.getcwd(), 'downloads')
    else:
        download_dir = os.path.join(tempfile.gettempdir(), 'django-geonames-downloads')

    countries = {}
    continents = {}
    localities = set()

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Delete data first and redownload',
        )

    def handle(self, *args, **options):
        start_time = datetime.datetime.now()

        if options['force']:
            self.load(clear=True)
        else:
            self.load()
        print(f'\nCompleted in {datetime.datetime.now() - start_time}')

    @transaction.atomic
    def load(self, clear=False):
        if clear:
            # self.cleanup_files()
            print("Deleting data")
            for member in geo_models:
                print(f" - {member._meta.verbose_name}")
                member.objects.all().delete()

        for member in geo_models:
            if member.objects.all().count() != 0:
                print(f'ERROR: there are {member._meta.verbose_name_plural} in the database')
                #sys.exit(1)

        #self.download_files()
        #self.unzip_files()

        self.load_continents()
        self.load_featureclasses()
        self.load_featurecodes()
        self.load_timezones()
        self.load_languagecodes()
        self.load_countries()
        
        #self.load_postcodes()
        #self.load_postcodes('GB_full.txt')

        self.load_admin1()
        self.load_admin2()
        self.load_localities()
        
        #self.cleanup()

        self.load_altnames()
        #self.load_hierarchies()
        
        #self.check_errors()

        # Save the time when the load happened
        GeonamesUpdate.objects.create()

    def download_files(self):
        # make the temp folder if it doesn't exist
        try:
            os.mkdir(self.download_dir)
        except OSError:
            pass
        os.chdir(self.download_dir)
        print()
        print(f'Downloading files to {self.download_dir}')
        print()
        for f in FILES:
            # --timestamping (-N) will overwrite files rather then appending .1, .2 ...
            # see http://stackoverflow.com/a/16840827/913223
            if os.system(f'wget --timestamping {f}') != 0:
                print(f"ERROR fetching {os.path.basename(f)}. Perhaps you are missing the 'wget' utility.")
                sys.exit(1)

    def unzip_files(self):
        os.chdir(self.download_dir)
        print("Unzipping downloaded files as needed: ''." % glob.glob('*.zip'))
        for f in glob.glob('*.zip'):
            if os.system(f'unzip -o {f}') != 0:
                print(f"ERROR unzipping {f}. Perhaps you are missing the 'unzip' utility.")
                sys.exit(1)

    def cleanup_files(self):
        try:
            print('Deleting files')
            shutil.rmtree(self.download_dir)
        except FileNotFoundError:
            print('Files not present')
            pass

    def load_continents(self):
        # TODO: [    ] Put this dictionary in an external TSV file. The info comes from this page: https://download.geonames.org/export/dump/
        continents = {
            "AF": { "name_en": "Africa", "geonameId": 6255146},
            "AS": { "name_en": "Asia", "geonameId": 6255147},
            "EU": { "name_en": "Europe", "geonameId": 6255148},
            "NA": { "name_en": "North America", "geonameId": 6255149},
            "OC": { "name_en": "Oceania", "geonameId": 6255151},
            "SA": { "name_en": "South America", "geonameId": 6255150},
            "AN": { "name_en": "Antarctica", "geonameId": 6255152},
        }
        objects = []
        for code, data in continents.items():
            objects.append(Continent(code=code, name_en=data["name_en"], geonameid=data["geonameId"]))
        Continent.objects.bulk_create(objects)
        print(f'{Continent.objects.all().count():8d} Continent objects created')

    def load_featureclasses(self):
        print('Adding feature classes')
        # The name_en field comes from this page: http://www.geonames.org/statistics/united-states.html (same for all countries)
        # TODO [    ] Put this dictionary in an external TSV file.
        feature_classes = {
            "A": {"name_en": "Administrative Boundary Features", "description_en": "country, state, region,..."},
            "H": {"name_en": "Hydrographic Features", "description_en": "stream, lake, ..."},
            "L": {"name_en": "Area Features", "description_en": "parks,area, ..."},
            "P": {"name_en": "Populated Place Features", "description_en": "city, village,..."},
            "R": {"name_en": "Road / Railroad Features", "description_en": "road, railroad"},
            "S": {"name_en": "Spot Features", "description_en": "spot, building, farm"},
            "T": {"name_en": "Hypsographic Features", "description_en": "mountain,hill,rock,..."}, 
            "U": {"name_en": "Undersea Features", "description_en": "undersea"},
            "V": {"name_en": "Vegetation Features", "description_en": "forest,heath,..."},
        }
        objects = []
        for f_class, data in feature_classes.items():
            objects.append(FeatureClass(f_class=f_class, name_en=data["name_en"], description_en=data["description_en"]))
        FeatureClass.objects.bulk_create(objects)
        print(f'{FeatureClass.objects.all().count():8d} FeatureClass objects created')

    def load_featurecodes(self):
        print('Loading feature codes')
        objects = []
        os.chdir(self.download_dir)
        with open('featureCodes_en.txt', 'r', encoding="utf8") as fd:
            try:
                #fd.readline()
                for line in fd:
                    fields = [field.strip() for field in line[:-1].split('\t')]
                    full_code, name, description = fields[0:3]
                    if full_code != "null":
                        if "." in full_code:
                            f_class = full_code.split(".")[0]
                            #feature_class = FeatureClass.objects.get(f_class=f_class)
                            f_code = full_code.split(".")[1]
                        else:
                            #feature_class=None
                            f_code=None
                        objects.append(
                            FeatureClassAndCode(
                                f_class_id=f_class, 
                                f_code=f_code, 
                                f_class_and_code=full_code, 
                                name_en=name, 
                                description_en=description
                            )
                        )
            except Exception as inst:
                traceback.print_exc(inst)
                raise Exception(f"ERROR parsing:\n {line}\n The error was: {inst}")
        FeatureClassAndCode.objects.bulk_create(objects)
        print(f'{FeatureClassAndCode.objects.all().count():8d} FeatureClassAndCode objects loaded')

    def load_timezones(self):
        print('Loading Timezones')
        objects = []
        os.chdir(self.download_dir)
        with open('timeZones.txt', 'r', encoding="utf8") as fd:
            try:
                fd.readline()
                for line in fd:
                    fields = [field.strip() for field in line[:-1].split('\t')]
                    name, gmt_offset, dst_offset, raw_offset = fields[1:5]
                    objects.append(Timezone(name=name, gmt_offset=gmt_offset, dst_offset=dst_offset, raw_offset=raw_offset))
            except Exception as inst:
                traceback.print_exc(inst)
                raise Exception(f"ERROR parsing:\n {line}\n The error was: {inst}")

        Timezone.objects.bulk_create(objects)
        print(f'{Timezone.objects.all().count():8d} Timezones loaded')

    def load_languagecodes(self):
        print('Loading Languages')
        objects = []
        os.chdir(self.download_dir)
        with open('iso-languagecodes.txt', 'r', encoding="utf8") as fd:
            try:
                fd.readline()  # skip the head
                for line in fd:
                    fields = [field.strip() for field in line.split('\t')]
                    iso_639_3=fields[0]
                    iso_639_2=fields[1]
                    iso_639_1=fields[2]
                    name = fields[3]
                    #if iso_639_1 != '':
                    objects.append(
                        Language(
                            iso_639_3=iso_639_3,
                            iso_639_2=iso_639_2,
                            iso_639_1=iso_639_1,
                            name=name)
                            )
            except Exception as inst:
                traceback.print_exc(inst)
                raise Exception(f"ERROR parsing:\n {line}\n The error was: {inst}")

        Language.objects.bulk_create(objects)
        print(f'{Timezone.objects.all().count():8d} Languages loaded')
        # TODO: [    ] Do we really need to "fix" the language names? We could add a shorter name for these cases.
        #self.fix_languagecodes()

    def fix_languagecodes(self):
        print('Fixing Language codes')
        corrections = {
            'km': 'Khmer',
            'ia': 'Interlingua',
            'ms': 'Malay',
            'el': 'Greek',
            'se': 'Sami',
            'oc': 'Occitan',
            'st': 'Sotho',
            'sw': 'Swahili',
            'to': 'Tonga',
            'fy': 'Frisian',
        }
        for iso_code, name in corrections.items():
            Language.objects.filter(iso_639_1=iso_code).update(name=name)

    def load_countries(self):
        print('Loading Countries')
        objects = []
        langs_dic = {}
        neighbours_dict = {}
        dollar = Currency.objects.create(code='USD', name='Dollar')
        os.chdir(self.download_dir)
        with open('countryInfo.txt', encoding="utf8") as fd:
            try:
                for line in fd:
                    if line[0] == '#':
                        continue
                    fields = [field.strip() for field in line[:-1].split('\t')]
                    code = fields[0]
                    code_iso3 = fields[1]
                    iso_numeric = fields[2]
                    fips = fields[3]
                    name = fields[4]
                    capital = fields[5]
                    area_sq_km = fields[6]
                    population = fields[7]
                    continent = fields[8] #Continent.objects.get(code=fields[8]) or None
                    tld = fields[9]
                    currency_code = fields[10]
                    currency_name = fields[11]
                    phone = fields[12]
                    postal_code_format = fields[13]
                    postal_code_regex = fields[14]
                    langs_dic[code] = fields[15]
                    geonameid = fields[16]
                    neighbours_dict[code] = fields[17]
                    neighbours_codes = fields[17]
                    equivalent_fips_code = fields[18]
                    if currency_code == '':
                        currency = dollar
                    else:
                        currency, created = Currency.objects.get_or_create(
                                code=currency_code, defaults={'name': currency_name})
                    self.countries[code] = {}
                    objects.append(
                        Country(
                            code=code,
                            code_iso3=code_iso3,
                            iso_numeric=iso_numeric,
                            fips=fips,
                            name=name,
                            capital=capital,
                            area_sq_km=area_sq_km,
                            population=population,
                            continent_id=continent,
                            tld=tld,
                            currency=currency,
                            currency_code=currency_code,
                            currency_name=currency_name,
                            phone=phone,
                            postal_code_format=postal_code_format,
                            postal_code_regex=postal_code_regex,
                            geonameid=geonameid,
                            neighbours_codes=neighbours_codes,
                            equivalent_fips_code=equivalent_fips_code
                        )   
                    )
            except Exception as inst:
                traceback.print_exc(inst)
                raise Exception(f"ERROR parsing:\n {line}\n The error was: {inst}")

        Country.objects.bulk_create(objects)
        print(f'{Country.objects.all().count():8d} Countries loaded')

        print('Adding Languages to Countries')
        default_lang = Language.objects.get(iso_639_1='en')
        for country in Country.objects.all():
            for code in langs_dic[country.code].split(','):
                iso_639_1 = code.split("-")[0]
                if len(iso_639_1) < 2:
                    continue

                languages = Language.objects.filter(iso_639_1=iso_639_1)
                if languages.count() == 1:
                    country.languages.add(languages[0])

            if country.languages.count() == 0:
                country.languages.add(default_lang)
        
        print('Adding neighbours to Countries')
        for country in Country.objects.all():
            #print(f"Neighbours for country {country} are {country.neighbours_codes}")
            for n in country.neighbours_codes.split(','):
                if n and n != '':
                    if Country.objects.filter(code=n).exists():
                        neighbour = Country.objects.get(code=n)
                        if neighbour:
                            #print(f"Adding neighbour for country {country}: {n} - {neighbour}")
                            country.neighbours.add(neighbour)
                    else:
                        print(f"Oops, country code {n} not found, skipping.")

    def load_admin1(self):
        print('Loading Admin1Codes')
        objects = []
        os.chdir(self.download_dir)
        with open('admin1CodesASCII.txt', encoding="utf8") as fd:
            for line in fd:
                fields = [field.strip() for field in line[:-1].split('\t')]
                codes, name = fields[0:2]
                country_code, admin1_code = codes.split('.')
                geonameid = fields[3]
                self.countries[country_code][admin1_code] = {'geonameid': geonameid, 'admins2': {}}
                objects.append(
                    Admin1Code(
                        geonameid=geonameid,
                        code=admin1_code,
                        name=name,
                        country_id=country_code
                    )
                )
        Admin1Code.objects.bulk_create(objects)
        print(f'{Admin1Code.objects.all().count():8d} Admin1Codes loaded')

    def load_admin2(self):
        print('Loading Admin2Codes')
        objects = []
        admin2_list = []  # to find duplicated
        skipped_duplicated = 0
        os.chdir(self.download_dir)
        with open('admin2Codes.txt', encoding="utf8") as fd:
            try:
                for line in fd:
                    fields = [field.strip() for field in line[:-1].split('\t')]
                    codes, name = fields[0:2]
                    country_code, admin1_code, admin2_code = codes.split('.')

                    # if there is a duplicated
                    long_code = f"{country_code}.{admin1_code}.{name}"
                    if long_code in admin2_list:
                        skipped_duplicated += 1
                        continue

                    admin2_list.append(long_code)

                    geonameid = fields[3]
                    admin1_dic = self.countries[country_code].get(admin1_code)

                    # if there is not admin1 level we save it but we don't keep it for the localities
                    if admin1_dic is None:
                        admin1_id = None
                    else:
                        # If not, we get the id of admin1 and we save geonameid for filling in Localities later
                        admin1_id = admin1_dic['geonameid']
                        admin1_dic['admins2'][admin2_code] = geonameid

                    name = name  # unicode(name, 'utf-8')
                    objects.append(
                        Admin2Code(
                            geonameid=geonameid,
                            code=admin2_code,
                            name=name,
                            country_id=country_code,
                            admin1_id=admin1_id
                        )
                    )
            except Exception as inst:
                traceback.print_exc(inst)
                raise Exception(f"ERROR parsing:\n {line}\n The error was: {inst}")
        Admin2Code.objects.bulk_create(objects, ignore_conflicts=True)
        print(f'{Admin2Code.objects.all().count():8d} Admin2Codes loaded')
        print(f'{skipped_duplicated:8d} Admin2Codes skipped because duplicated')

    def load_localities(self, fn='CY_Localities/CY.txt'):
        print('Loading Localities')
        objects = []
        batch = 1000
        processed = 0
        os.chdir(self.download_dir)
        with open(fn, 'r', encoding="utf8") as fd:
            for line in fd:
                #print(f'Processing line: ', line)
                try:
                    fields = [field.strip() for field in line[:-1].split('\t')]
                    geonameid = fields[0]
                    name = fields[1]
                    # ascii_name = fields[2]
                    # alternate_names = field[3]
                    lat = float(fields[4])
                    lon = float(fields[5])
                    
                    feature_class = fields[6] #FeatureClass.objects.filter(f_class=fields[6])
                    """
                    if feature_class.exists():
                        feature_class = feature_class[0]
                    else:
                        feature_class = None
                    """
                    feature_code = fields[7] #FeatureClassAndCode.objects.filter(f_class=feature_class, f_code=fields[7])
                    """
                    if feature_code.exists():
                        feature_code = feature_code[0]
                    else:
                        feature_code = None
                    """
                    country_code = fields[8]
                    #country = Country.objects.filter(code=fields[8])[0]
                    # alternative_countries = field[9]
                    admin1_code = fields[10]
                    admin2_code = fields[11]
                    admin3 = fields[12]
                    admin4 = fields[13]
                    admin1_dic = self.countries[country_code].get(admin1_code)
                    if admin1_dic:
                        admin1_id = admin1_dic['geonameid']
                        admin2_id = admin1_dic['admins2'].get(admin2_code)
                    else:
                        admin1_id = None
                        admin2_id = None
                    #type = fields[7]
                    #if type not in city_types:
                    #    continue
                    population = int(fields[14]) if fields[14] != '' else None
                    elevation = int(fields[15]) if fields[15] != '' else None
                    # dem = fields[16]
                    timezone_name = fields[17]
                    timezone = Timezone.objects.filter(name=timezone_name)
                    if timezone.exists():
                        timezone = timezone[0]
                    else:
                        timezone = None
                    modification_date = fields[18]
                    locality = Locality(
                        geonameid=geonameid,
                        name=name,
                        country_id=country_code if country_code != '' else None,
                        feature_class_id=feature_class if feature_class != '' else None,
                        feature_code_id=feature_code if feature_code != '' else None,
                        admin1_id=admin1_id,
                        admin2_id=admin2_id,
                        admin3=admin3,
                        admin4=admin4,
                        lat=lat,
                        lon=lon,
                        timezone=timezone,
                        population=population,
                        elevation=elevation,
                        modification_date=modification_date
                    )
                    if GIS_LIBRARIES:
                        locality.point = Point(lon, lat)
                    locality.long_name = locality.generate_long_name()
                    objects.append(locality)
                    processed += 1
                    self.localities.add(geonameid)
                except Exception as inst:
                    print(f'Error parsing line: {line}')
                    traceback.print_exc(inst)
                    raise Exception(f"ERROR parsing:\n {line}\n The error was: {inst}")

                if processed % batch == 0:
                    Locality.objects.bulk_create(objects, ignore_conflicts=True)
                    print(f"{processed:8d} Localities loaded")
                    objects = []

        Locality.objects.bulk_create(objects)
        print(f"{processed:8d} Localities lines processed.")
        print(f"{Locality.objects.count():8d} Localities loaded")

        print('Filling missed timezones in localities')
        # Try to find the missing timezones
        for locality in Locality.objects.filter(timezone__isnull=True):
            # We assign the time zone of the most populated locality in the same admin2
            near_localities = Locality.objects.filter(admin2=locality.admin2)
            near_localities = near_localities.exclude(timezone__isnull=True)
            if not near_localities.exists():
                # We assign the time zone of the most populated locality in the same admin1
                near_localities = Locality.objects.filter(admin1=locality.admin1)
                near_localities = near_localities.exclude(timezone__isnull=True)

            if not near_localities.exists():
                # We assign the time zone of the most populated locality in the same country
                near_localities = Locality.objects.filter(country=locality.country)
                near_localities = near_localities.exclude(timezone__isnull=True)

            if near_localities.exists():
                near_localities = near_localities.order_by('-population')
                locality.timezone = near_localities[0].timezone
                locality.save()
            else:
                print(f"ERROR locality with no timezone {locality}")
                raise Exception()

    def cleanup(self):
        # We do not need this, we need all countries ENABLED
        # self.delete_empty_countries()
        self.delete_duplicated_localities()

    def delete_empty_countries(self):
        print('Setting as deleted empty Countries')
        # Countries
        countries = Country.objects.annotate(Count("locality_set")).filter(locality_set__count=0)
        for c in countries:
            c.status = Country.objects.STATUS_DISABLED
            c.save()

        print(f" {countries.count():8d} Countries set status 'STATUS_DISABLED'")

    def delete_duplicated_localities(self):
        print("Setting as deleted duplicated localities")
        total = 0
        for c in Country.objects.all():
            prev_name = ""
            for loc in c.locality_set.order_by("long_name", "-population"):
                if loc.long_name == prev_name:
                    loc.status = Locality.objects.STATUS_DISABLED
                    loc.save(check_duplicated_longname=False)
                    total += 1

                prev_name = loc.long_name

        print(f" {total:8d} localities set as 'STATUS_DISABLED'")

    def load_altnames(self, fn='CY_Altnames/CY.txt'):
        print('Loading alternate names')
        objects = []
        allobjects = {}
        batch = 1000
        processed = 0
        os.chdir(self.download_dir)
        with open(fn, 'r', encoding="utf8") as fd:
            for line in fd:
                try:
                    fields = [field.strip() for field in line.split('\t')]
                    alternatenameid = fields[0]
                    locality_geonameid = fields[1]
                    """
                    locality_geoname = Locality.objects.filter(geonameid=locality_geonameid)
                    if locality_geoname.exists():
                        locality_geoname = locality_geoname[0]
                    else:
                        locality_geoname = None
                    """
                    isolanguage = fields[2]
                    #if locality_geonameid not in self.localities:
                    #    continue
                    name = fields[3]
                    is_preferred_name = bool(fields[4])
                    is_short_name = bool(fields[5])
                    is_colloquial = bool(fields[6])
                    is_historic = bool(fields[7])
                    from_period = fields[8]
                    to_period = fields[9]
                    """
                    if locality_geonameid in allobjects:
                        if name in allobjects[locality_geonameid]:
                            continue
                    else:
                        allobjects[locality_geonameid] = set()

                    allobjects[locality_geonameid].add(name)
                    """
                    objects.append(
                        AlternateName(
                            alternatenameid=alternatenameid,
                            locality_id=locality_geonameid,
                            isolanguage=isolanguage,
                            name=name,
                            is_preferred_name=is_preferred_name,
                            is_short_name=is_short_name,
                            is_colloquial=is_colloquial,
                            is_historic=is_historic,
                            from_period=from_period,
                            to_period=to_period,
                        )
                    )
                    processed += 1
                except Exception as inst:
                    print(f'Error parsing line: {line}')
                    traceback.print_exc(inst)
                    raise Exception(f"ERROR parsing:\n {line}\n The error was: {inst}")

                if processed % batch == 0:
                    AlternateName.objects.bulk_create(objects, ignore_conflicts=True)
                    print(f"{processed:8d} AlternateNames loaded")
                    objects = []

        AlternateName.objects.bulk_create(objects, ignore_conflicts=True)
        print(f"{processed:8d} AlternateNames loaded")

    def load_hierarchies(self, fn='hierarchy.txt'):
        print('Loading hierarchies')
        objects = []
        allobjects = {}
        batch = 1000
        processed = 0
        os.chdir(self.download_dir)
        with open(fn, 'r', encoding="utf8") as fd:
            for line in fd:
                try:
                    fields = [field.strip() for field in line.split('\t')]
                    parent = Locality.objects.filter(geonameid=int(fields[0]))
                    child = Locality.objects.filter(geonameid=int(fields[1]))
                    if parent.exists() and child.exists():
                        parent=parent[0]
                        child=child[0]
                        hierarchy_type = fields[2]
                        objects.append(
                            LocalityHierarchy(
                                parent=parent,
                                child=child,
                                hierarchy_type=hierarchy_type,
                            )
                        )
                    processed += 1
                except Exception as inst:
                    print(f'Error parsing line: {line}')
                    traceback.print_exc(inst)
                    raise Exception(f"ERROR parsing:\n {line}\n The error was: {inst}")

                if processed % batch == 0:
                    LocalityHierarchy.objects.bulk_create(objects, ignore_conflicts=True)
                    print(f"{processed:8d} LocalityHierarchy loaded")
                    objects = []

        LocalityHierarchy.objects.bulk_create(objects, ignore_conflicts=True)
        print(f"{processed:8d} LocalityHierarchy loaded")

    def load_postcodes(self, fn='allCountries.txt'):
        """Load postcode files: allCountries.txt and GB_full.txt"""
        print(f"Loading postcodes from {fn}.")
        objects = []
        batch = 1000
        processed = 0
        os.chdir(self.download_dir)

        with open(fn, 'r', encoding="utf8") as fd:
            for line in fd:
                fields = [field.strip() for field in line.split('\t')]
                (
                    country_code, postal_code, place_name,
                    admin_name1, admin_code1,
                    admin_name2, admin_code2,
                    admin_name3, admin_code3,
                    lat, lon, accuracy
                ) = fields

                lat = float(lat)
                lon = float(lon)

                postcode = Postcode(
                    country_id=country_code,
                    postal_code=postal_code,
                    place_name=place_name,
                    admin_name1=admin_name1,
                    admin_code1=admin_code1,
                    admin_name2=admin_name2,
                    admin_code2=admin_code2,
                    admin_name3=admin_name3,
                    admin_code3=admin_code3,
                    lat=lat,
                    lon=lon,
                    accuracy=accuracy or None
                )
                if GIS_LIBRARIES:
                    postcode.point = Point(lon, lat)
                objects.append(postcode)
                processed += 1

                if processed % batch == 0:
                    Postcode.objects.bulk_create(objects, ignore_conflicts=True)
                    print(f"{processed:8d} Postcodes loaded")
                    objects = []

    def check_errors(self):
        print('Checking errors')

        print('Checking empty country')
        if Country.objects.public().annotate(Count("locality_set")).filter(locality_set__count=0):
            print("Possible error: there are Countries with no localities")
            #raise Exception()

        print('Checking if Localities have a timezone')
        if Locality.objects.filter(timezone__isnull=True):
            print("ERROR Localities with no timezone")
            #raise Exception()

        print('Checking duplicated localities per country')
        for country in Country.objects.all():
            duplicated = country.locality_set.public().values('long_name')\
                .annotate(Count('long_name')).filter(long_name__count__gt=1)
            if len(duplicated) != 0:
                print(f"ERROR Duplicated localities in {country}: {duplicated}")
                print(duplicated)
                # raise Exception()
