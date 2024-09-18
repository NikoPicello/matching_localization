import google_streetview.api
import requests
import polyline
import math
import csv
import os
import numpy as np
from geopy.distance import geodesic

KEY = 'AIzaSyAibkslal3KC-_PgONK85PKV5dDTS0H-XQ'

def interpolate_path(path_coords, interval=10):
    interpolated_points = []
    for i in range(len(path_coords) - 1):
        start = path_coords[i]
        end = path_coords[i + 1]
        distance = geodesic(start, end).meters
        num_points = int(distance // interval)
        lats = np.linspace(start[0], end[0], num_points)
        lngs = np.linspace(start[1], end[1], num_points)
        interpolated_points.extend(zip(lats, lngs))
    return interpolated_points

import requests

def snap_to_road(points):
    max_points_per_request = 100  # The Roads API allows a maximum of 100 points per request
    snapped_points = []

    # Split points into batches of 100 or fewer
    for i in range(0, len(points), max_points_per_request):
        batch = points[i:i + max_points_per_request]
        path = '|'.join([f"{lat},{lng}" for lat, lng in batch])

        url = f"https://roads.googleapis.com/v1/snapToRoads?path={path}&interpolate=true&key={KEY}"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            if 'snappedPoints' in data:
                # Extract snapped points from the response
                batch_snapped_points = [(p['location']['latitude'], p['location']['longitude']) for p in data['snappedPoints']]
                snapped_points.extend(batch_snapped_points)
            else:
                print("No snapped points in the response.")
        else:
            print(f"Error in Roads API request: {response.status_code}")
            print(response.text)

    return snapped_points


def calculate_bearing(pointA, pointB):
    lat1 = math.radians(pointA[0])
    lon1 = math.radians(pointA[1])
    lat2 = math.radians(pointB[0])
    lon2 = math.radians(pointB[1])
    dLon = lon2 - lon1
    x = math.sin(dLon) * math.cos(lat2)
    y = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dLon)
    bearing = (math.degrees(math.atan2(x, y)) + 360) % 360
    return bearing

def get_headings(points):
    headings = []
    for i in range(len(points) - 1):
        bearing = calculate_bearing(points[i], points[i + 1])
        headings.append(bearing)
    headings.append(headings[-1])  # Assume last heading same as second last
    return headings


def download_street_view_images(points, headings, output_dir='street_view_images'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create CSV file to store image metadata (coordinates and filenames)
    metadata_file = os.path.join(output_dir, 'image_metadata.csv')

    with open(metadata_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Filename', 'Latitude', 'Longitude', 'Heading'])  # CSV header

        for idx, (point, heading) in enumerate(zip(points, headings)):
            lat, lng = point
            image_filename = f"image_{idx:03d}.jpg"
            image_path = os.path.join(output_dir, image_filename)

            # Construct the Street View API request URL
            url = f"https://maps.googleapis.com/maps/api/streetview?size=640x640&location={lat},{lng}&heading={heading}&pitch=0&key={KEY}"
            response = requests.get(url)

            if response.status_code == 200:
                # Save the image
                with open(image_path, 'wb') as img_file:
                    img_file.write(response.content)

                # Save metadata (coordinates and heading) in CSV
                writer.writerow([image_filename, lat, lng, heading])
                print(f"Downloaded and saved: {image_filename} (Lat: {lat}, Lng: {lng}, Heading: {heading})")
            else:
                print(f"Failed to download image at index {idx} (Lat: {lat}, Lng: {lng}). Status code: {response.status_code}")


def get_route_coordinates(origin, destination):
    url = f"https://maps.googleapis.com/maps/api/directions/json?origin={origin[0]},{origin[1]}&destination={destination[0]},{destination[1]}&key={KEY}"
    response = requests.get(url)
    routes = response.json()['routes']
    if routes:
        overview_polyline = routes[0]['overview_polyline']['points']
        path_coords = polyline.decode(overview_polyline)
        return path_coords
    else:
        print("No routes found")
        return []

def main():
    # Define your start and end coordinates
    origin = (41.383541, 2.181542)      # Empire State Building
    destination = (41.383902, 2.178899) # Times Square

    # origin = (40.748817, -73.985428)      # Empire State Building
    # destination = (40.758896, -73.985130)

    # Get route coordinates from Directions API
    path_coords = get_route_coordinates(origin, destination)

    if path_coords:
        # Interpolate points along the path
        interpolated_points = interpolate_path(path_coords, interval=10)

        # Snap points to the road network
        snapped_points = snap_to_road(interpolated_points)

        # Calculate headings at each point
        headings = get_headings(snapped_points)

        # Download Street View images
        download_street_view_images(snapped_points, headings)
    else:
        print("Could not retrieve route coordinates.")

if __name__ == '__main__':
  main()
