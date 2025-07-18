from django.db import models

class WorkPost(models.Model):
    farmer_name = models.CharField(max_length=100)
    contact_number = models.CharField(max_length=15)
    village = models.CharField(max_length=100)
    crop_type = models.CharField(max_length=50)
    work_description = models.TextField()
    date_posted = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.crop_type} work in {self.village} by {self.farmer_name}"

