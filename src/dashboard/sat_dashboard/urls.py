from django.contrib import admin
from django.urls import include, path, re_path
from django.views.generic import TemplateView
from simulation.proxy_views import proxy_to_sim


urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("simulation.urls")),
    # Transparent proxy to SimSat simulator for /sim/* routes
    re_path(r"^sim/(?P<path>.*)$", proxy_to_sim, name="sim-proxy"),
    # Catch-all: serve the React SPA entrypoint
    re_path(r"^.*$", TemplateView.as_view(template_name="index.html"), name="spa-entry"),
]
