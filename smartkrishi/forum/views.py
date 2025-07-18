from django.shortcuts import render, redirect
from .models import WorkPost
from .forms import WorkPostForm

def create_work_post(request):
    if request.method == 'POST':
        form = WorkPostForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('view-work-posts')
    else:
        form = WorkPostForm()
    return render(request, 'forum/create_post.html', {'form': form})

def view_work_posts(request):
    posts = WorkPost.objects.all().order_by('-date_posted')
    return render(request, 'forum/view_posts.html', {'posts': posts})

