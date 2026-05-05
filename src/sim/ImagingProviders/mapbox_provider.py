from math import acos, radians, sin, cos, sqrt, atan2, log2
import os
import requests

import numpy as np
EARTH_RADIUS_KM = 6371.0
EPS = 1e-12

class MapboxProvider:

    def __init__(self):
        self.api_token = os.environ.get("MAPBOX_ACCESS_TOKEN")
        if self.api_token is None:
            raise ValueError("MAPBOX_ACCESS_TOKEN environment variable not set")



    def get_target_image(self, sat_lon, sat_lat, sat_alt, target_lon, target_lat):
        metadata = self.probe_target_geometry(sat_lon, sat_lat, sat_alt, target_lon, target_lat)
        print(f"get target input vars: sat_lon={sat_lon}, sat_lat={sat_lat}, sat_alt={sat_alt}, target_lon={target_lon}, target_lat={target_lat}")

        if not metadata["target_visible"]:
            return {"image": None, "metadata": metadata}
        
        # get image from mapbox
        url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{target_lon},{target_lat},{metadata['zoom_factor']},{metadata['bearing']},{metadata['pitch']}/1280x1280@2x?access_token={self.api_token}"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error fetching image: {response.status_code} - {response.text}")
            return {
                "image": None,
                "metadata": metadata
            }

        metadata["image_available"] = True
        return {
            "image": response.content,
            "metadata": metadata
        }

    def probe_target_geometry(self, sat_lon, sat_lat, sat_alt, target_lon, target_lat):
        cartesian_sat = self._spherical_to_cartesian(sat_lon, sat_lat, EARTH_RADIUS_KM + sat_alt)
        cartesian_target = self._spherical_to_cartesian(target_lon, target_lat, EARTH_RADIUS_KM)

        target_to_sat_vector = cartesian_sat - cartesian_target
        distance = np.linalg.norm(target_to_sat_vector)
        target_to_sat_unit_vector = target_to_sat_vector / distance
        target_unit_vector = cartesian_target / np.linalg.norm(cartesian_target)

        # Elevation angle: theta = angle between surface normal and sat-to-target vector.
        theta = acos(np.clip(np.dot(target_unit_vector, target_to_sat_unit_vector), -1.0, 1.0))
        elevation_degrees = 90 - np.degrees(theta)
        pitch = np.degrees(theta)  # mapbox pitch
        target_visible = elevation_degrees >= 30
        zoom_factor = 13.92 + log2(560 / distance)

        sat_to_target_unit_vector = (cartesian_target - cartesian_sat) / distance
        nadir_unit_vector = -cartesian_sat / np.linalg.norm(cartesian_sat)
        off_nadir_degrees = np.degrees(
            acos(np.clip(np.dot(nadir_unit_vector, sat_to_target_unit_vector), -1.0, 1.0))
        )

        # Bearing: project sat vector and Z-axis (north pole) onto the target's tangent plane.
        earth_center_to_south_vector = np.array([0, 0, 1])  # Z-axis points to North Pole
        target_to_sat_vec_projection_to_earth_surface = target_to_sat_unit_vector - np.dot(target_to_sat_unit_vector, target_unit_vector) * target_unit_vector
        target_proj_norm = np.linalg.norm(target_to_sat_vec_projection_to_earth_surface)
        if target_proj_norm < EPS:
            # Near nadir view: bearing is undefined; use a stable default.
            bearing = 0.0
        else:
            target_to_sat_vec_projection_to_earth_surface_unit_vector = target_to_sat_vec_projection_to_earth_surface / target_proj_norm
            south_vec_projection_to_earth_surface = earth_center_to_south_vector - np.dot(earth_center_to_south_vector, target_unit_vector) * target_unit_vector
            south_proj_norm = np.linalg.norm(south_vec_projection_to_earth_surface)
            if south_proj_norm < EPS:
                # Degenerate case close to poles; use the same stable default.
                bearing = 0.0
            else:
                south_vec_projection_to_earth_surface_unit_vector = south_vec_projection_to_earth_surface / south_proj_norm
                bearing = 180 - np.degrees(acos(np.clip(np.dot(south_vec_projection_to_earth_surface_unit_vector, target_to_sat_vec_projection_to_earth_surface_unit_vector), -1.0, 1.0)))
                bearing_cross = np.cross(south_vec_projection_to_earth_surface_unit_vector, target_to_sat_vec_projection_to_earth_surface_unit_vector)
                if np.dot(bearing_cross, target_unit_vector) < 0:
                    bearing = -bearing
                if bearing < 0:
                    bearing += 360

        return {
            "target_visible": target_visible,
            "image_available": False,
            "elevation_degrees": elevation_degrees if target_visible else None,
            "zoom_factor": zoom_factor if target_visible else None,
            "bearing": bearing if target_visible else None,
            "pitch": pitch if target_visible else None,
            "slant_range_km": distance,
            "off_nadir_degrees": off_nadir_degrees,
        }

    # --------------------------------------------------------
    # Helper functions
    # --------------------------------------------------------

    def _spherical_to_cartesian(self, lon, lat, radius):
        # Convert degrees to radians
        lon_rad = radians(lon)
        lat_rad = radians(lat)

        x = radius * cos(lat_rad) * cos(lon_rad)
        y = radius * cos(lat_rad) * sin(lon_rad)
        z = radius * sin(lat_rad)

        return np.array([x, y, z])


if __name__ == "__main__":
    provider = MapboxProvider()
    lausanne = {'lon': 6.6322734, 'lat': 46.5218266}
    lausanne_north = {'lon': 6.6322734, 'lat': 46.5318266}
    paris = {'lon': 2.3522219, 'lat': 48.856614}
    stuttgart = {'lon': 9.1829321, 'lat': 48.7758459}
    p1 = {'lon': 6.6322734-1, 'lat': 46.5218266}

    h = 500  # km

    sat = stuttgart
    target = lausanne

    provider.get_target_image(sat['lon'], sat['lat'], h, target['lon'], target['lat'])

        # def get_target_image(self, sat_lon, sat_lat, sat_alt, target_lon, target_lat):
