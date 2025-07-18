
from django.urls import path
from . import views

urlpatterns = [
    path('post-work/', views.create_work_post, name='create-work-post'),
    path('view-work/', views.view_work_posts, name='view-work-posts'),
]


