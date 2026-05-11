"""
Heatmap Visualization Module for AI Accident Risk Prediction System.

This module provides functionality to generate interactive heatmaps
visualizing accident hotspots based on geographic coordinates and risk levels.

Author: AI Road Detection Team
"""

import folium
from folium.plugins import HeatMap
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the Haversine distance between two geographic points.
    
    Parameters:
    -----------
    lat1 : float
        Latitude of the first point in decimal degrees
    lon1 : float
        Longitude of the first point in decimal degrees
    lat2 : float
        Latitude of the second point in decimal degrees
    lon2 : float
        Longitude of the second point in decimal degrees
    
    Returns:
    --------
    float
        Distance between the two points in kilometers
    """
    # Earth's radius in kilometers
    R = 6371.0
    
    # Convert latitude and longitude from degrees to radians
    lat1_rad = radians(lat1)
    lon1_rad = radians(lon1)
    lat2_rad = radians(lat2)
    lon2_rad = radians(lon2)
    
    # Differences in coordinates
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # Haversine formula
    a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    # Calculate distance
    distance = R * c
    
    return distance


def generate_heatmap(
    df: pd.DataFrame,
    selected_hour: int = None,
    center_lat: float = None,
    center_lon: float = None,
    radius_km: float = 10,
    radius: int = 15,
    blur: int = 10
) -> folium.Map:
    """
    Generate an interactive heatmap visualization of accident hotspots.
    
    This function creates a Folium map with a heatmap layer showing accident
    locations. It supports filtering by hour and geographic radius, and
    weights accidents by their risk level.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing accident data with columns:
        - Hour: Hour of the accident (0-23)
        - Latitude: Geographic latitude
        - Longitude: Geographic longitude
        - High_Risk: Risk indicator (1 for high risk, 0 for low risk)
    selected_hour : int, optional
        Filter accidents by specific hour (0-23). If provided, only accidents
        occurring during this hour will be included.
    center_lat : float, optional
        Latitude of the center point for radius filtering. If provided,
        accidents within radius_km of this location will be shown.
    center_lon : float, optional
        Longitude of the center point for radius filtering. If provided,
        accidents within radius_km of this location will be shown.
    radius_km : float, optional
        Radius in kilometers for filtering accidents around center point
        (default: 10). Only used when center_lat and center_lon are provided.
    radius : int, optional
        Radius of each heat point in the heatmap (default: 15)
    blur : int, optional
        Blur amount for heat points in the heatmap (default: 10)
    
    Returns:
    --------
    folium.Map
        Interactive Folium map object with heatmap layer
    
    Raises:
    -------
    ValueError
        If no valid data remains after filtering (empty DataFrame)
    
    Example:
    --------
    >>> import pandas as pd
    >>> from visualization.heatmap import generate_heatmap
    >>> df = pd.read_csv('accidents_clean.csv')
    >>> 
    >>> # Generate heatmap with all accidents
    >>> m = generate_heatmap(df)
    >>> 
    >>> # Generate heatmap filtered by hour
    >>> m = generate_heatmap(df, selected_hour=8)
    >>> 
    >>> # Generate heatmap centered on a specific location
    >>> m = generate_heatmap(df, center_lat=40.7128, center_lon=-74.0060, radius_km=15)
    """
    
    # Start with a copy of the input DataFrame
    df_filtered = df.copy()
    
    # Filter by selected_hour if provided
    if selected_hour is not None:
        if 'Hour' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['Hour'] == selected_hour]
    
    # Drop missing Latitude and Longitude values
    required_cols = ['Latitude', 'Longitude', 'High_Risk']
    df_filtered = df_filtered.dropna(subset=required_cols)
    
    # If center coordinates are provided, filter by radius
    if center_lat is not None and center_lon is not None:
        # Calculate distance from center for each accident
        distances = []
        for _, row in df_filtered.iterrows():
            dist = haversine_distance(
                center_lat, center_lon,
                row['Latitude'], row['Longitude']
            )
            distances.append(dist)
        
        df_filtered = df_filtered[np.array(distances) <= radius_km]
    
    # Check if we have valid data after filtering
    if df_filtered.empty:
        raise ValueError(
            "No valid data available after filtering. "
            "Please check your filter criteria (selected_hour, radius_km) "
            "and ensure the DataFrame contains valid latitude/longitude data."
        )
    
    # Calculate center of the map
    if center_lat is not None and center_lon is not None:
        # Use the provided center coordinates
        map_center_lat = center_lat
        map_center_lon = center_lon
    else:
        # Use average latitude and longitude from the data
        map_center_lat = df_filtered['Latitude'].mean()
        map_center_lon = df_filtered['Longitude'].mean()
    
    # Create a Folium map centered on the data
    m = folium.Map(
        location=[map_center_lat, map_center_lon],
        zoom_start=6,
        tiles='OpenStreetMap'
    )
    
    # Add a marker for the center location if provided
    if center_lat is not None and center_lon is not None:
        folium.Marker(
            location=[center_lat, center_lon],
            popup=f"Center Location<br>Radius: {radius_km} km",
            icon=folium.Icon(color='red', icon='info-sign'),
            tooltip="Center Point"
        ).add_to(m)
    
    # Prepare heatmap data with High_Risk as intensity weight
    # High_Risk == 1 → intensity = 1.0
    # High_Risk == 0 → intensity = 0.3
    heat_data = []
    for _, row in df_filtered.iterrows():
        if row['High_Risk'] == 1:
            intensity = 1.0
        else:
            intensity = 0.3
        
        heat_data.append([row['Latitude'], row['Longitude'], intensity])
    
    # Define the gradient: blue → lime → yellow → orange → red
    gradient = {
        0.2: 'blue',
        0.4: 'lime',
        0.6: 'yellow',
        0.8: 'orange',
        1.0: 'red'
    }
    
    # Add HeatMap layer to the map
    HeatMap(
        heat_data,
        radius=radius,
        blur=blur,
        max_zoom=13,
        gradient=gradient
    ).add_to(m)
    
    return m
