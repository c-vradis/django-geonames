from django.contrib import admin
from geonames.models import (
    GeonamesUpdate, Timezone, Language, Currency, Country,
    Admin1Code, Admin2Code, Locality, AlternateName, Postcode, FeatureClass, FeatureClassAndCode
)


class PostcodeAdmin(admin.ModelAdmin):
    search_fields = ['postal_code']


class LocalityAdmin(admin.ModelAdmin):
    search_fields = ['name']
    raw_id_fields = ['admin1', 'admin2']


class AlternateNameAdmin(admin.ModelAdmin):
    search_fields = ['name']
    raw_id_fields = ['locality']


class Admin1CodeAdmin(admin.ModelAdmin):
    search_fields = ['name']
    raw_id_fields = ['locality']

class FeatureClassAdmin(admin.ModelAdmin):
    search_fields = ['f_class', 'name_en', 'description_en',]
    list_display = ['f_class', 'name_en', 'description_en']

class FeatureClassAndCodeCodeAdmin(admin.ModelAdmin):
    search_fields = ['name_en', 'description_en', 'f_class_and_code']
    list_display = ['f_class__fclass', 'f_code', 'name_en', 'description_en']

admin.site.register(GeonamesUpdate)
admin.site.register(Timezone)
admin.site.register(Language)
admin.site.register(Currency)
admin.site.register(Country)
admin.site.register(Admin1Code)
admin.site.register(Admin2Code)
admin.site.register(Locality, LocalityAdmin)
admin.site.register(AlternateName, AlternateNameAdmin)
admin.site.register(Postcode, PostcodeAdmin)
admin.site.register(FeatureClass, FeatureClassAdmin)
admin.site.register(FeatureClassAndCode, FeatureClassAndCodeCodeAdmin)
