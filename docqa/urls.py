from django.urls import include, path
from rest_framework.authtoken.views import obtain_auth_token
from rest_framework.routers import DefaultRouter

from docqa import views

router = DefaultRouter()
router.register("documents", views.DocumentViewSet, basename="document")

urlpatterns = [
    path("auth/register/", views.RegisterView.as_view()),
    path("auth/token/", obtain_auth_token),
    path("ask/", views.AskView.as_view()),
    path("ask/stream/", views.AskStreamView.as_view()),
    path("", include(router.urls)),
]
