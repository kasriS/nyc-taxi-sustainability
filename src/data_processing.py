import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import requests
import json

class DataProcessor:
    def __init__(self):
        self.data_dir = 'data'
        self.raw_dir = os.path.join(self.data_dir, 'raw')
        self.processed_dir = os.path.join(self.data_dir, 'processed')
        self.external_dir = os.path.join(self.data_dir, 'external')
        
        # Create directories if they don't exist
        for dir_path in [self.raw_dir, self.processed_dir, self.external_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def load_taxi_data(self, file_path):
        """Load and preprocess taxi data"""
        try:
            df = pd.read_csv(file_path)
            df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
            if 'dropoff_datetime' in df.columns:
                df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])
            return df
        except Exception as e:
            print(f"Error loading taxi data: {e}")
            return None
    
    def create_weather_data(self):
        """Create synthetic weather data for NYC"""
        # Generate weather data for the taxi trip period
        date_range = pd.date_range(start='2016-01-01', end='2016-06-30', freq='D')
        
        np.random.seed(42)
        weather_data = []
        
        for date in date_range:
            # Seasonal patterns
            day_of_year = date.timetuple().tm_yday
            temp_base = 20 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            
            weather_data.append({
                'date': date.date(),
                'temperature': temp_base + np.random.normal(0, 5),
                'humidity': max(20, min(100, 60 + np.random.normal(0, 15))),
                'wind_speed': max(0, np.random.exponential(5)),
                'precipitation': max(0, np.random.exponential(2) if np.random.random() < 0.2 else 0),
                'snow_fall': max(0, np.random.exponential(1) if (date.month in [12, 1, 2] and np.random.random() < 0.1) else 0),
                'visibility': max(1, min(10, 8 + np.random.normal(0, 2))),
                'weather_condition': np.random.choice(['Clear', 'Cloudy', 'Rainy', 'Snowy'], p=[0.5, 0.3, 0.15, 0.05])
            })
        
        weather_df = pd.DataFrame(weather_data)
        weather_df['precip_intensity'] = weather_df['precipitation'] + weather_df['snow_fall']
        
        # Save weather data
        weather_path = os.path.join(self.external_dir, 'weather_data.csv')
        weather_df.to_csv(weather_path, index=False)
        print(f"Weather data created and saved to {weather_path}")
        
        return weather_df
    
    def create_holiday_data(self):
        """Create holiday data for NYC"""
        import holidays
        
        # Get US holidays for the relevant period
        us_holidays = holidays.US(years=[2016])
        
        holiday_data = []
        for date, name in us_holidays.items():
            holiday_data.append({
                'date': date,
                'holiday': name,
                'is_holiday': 1
            })
        
        # Add some NYC specific events
        nyc_events = [
            ('2016-03-17', 'St. Patrick\'s Day Parade'),
            ('2016-11-24', 'Thanksgiving Day Parade'),
            ('2016-12-31', 'New Year\'s Eve Times Square')
        ]
        
        for date_str, event_name in nyc_events:
            holiday_data.append({
                'date': pd.to_datetime(date_str).date(),
                'holiday': event_name,
                'is_holiday': 1
            })
        
        holiday_df = pd.DataFrame(holiday_data)
        
        # Save holiday data
        holiday_path = os.path.join(self.external_dir, 'holiday_data.csv')
        holiday_df.to_csv(holiday_path, index=False)
        print(f"Holiday data created and saved to {holiday_path}")
        
        return holiday_df
    
    def create_osrm_data(self, taxi_df, sample_size=10000):
        """Create synthetic OSRM routing data"""
        # Sample data for OSRM features
        sample_df = taxi_df.sample(min(sample_size, len(taxi_df)), random_state=42)
        
        osrm_data = []
        for _, row in sample_df.iterrows():
            # Calculate Haversine distance for baseline
            lat1, lon1 = row['pickup_latitude'], row['pickup_longitude']
            lat2, lon2 = row['dropoff_latitude'], row['dropoff_longitude']
            
            # Haversine distance calculation
            R = 6371  # Earth's radius in km
            lat1_rad, lat2_rad = np.radians([lat1, lat2])
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            
            a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
            haversine_dist = 2 * R * np.arcsin(np.sqrt(a))
            
            # Synthetic OSRM features based on haversine distance
            total_distance = haversine_dist * (1 + np.random.normal(0.2, 0.1))  # Roads are longer than straight line
            total_travel_time = total_distance / 25 * 3600 + np.random.normal(0, 300)  # ~25 km/h average
            number_of_steps = max(1, int(total_distance * 2 + np.random.normal(0, 5)))
            
            osrm_data.append({
                'id': row['id'],
                'total_distance': max(0.1, total_distance * 1000),  # Convert to meters
                'total_travel_time': max(60, total_travel_time),  # Minimum 1 minute
                'number_of_steps': number_of_steps
            })
        
        osrm_df = pd.DataFrame(osrm_data)
        
        # Save OSRM data
        osrm_path = os.path.join(self.external_dir, 'osrm_data.csv')
        osrm_df.to_csv(osrm_path, index=False)
        print(f"OSRM data created and saved to {osrm_path}")
        
        return osrm_df
    
    def clean_and_validate_data(self, df):
        """Clean and validate taxi data"""
        print(f"Initial data shape: {df.shape}")
        
        # Remove invalid coordinates (outside NYC area)
        df = df[
            (df['pickup_longitude'].between(-74.3, -73.7)) &
            (df['pickup_latitude'].between(40.5, 41.0)) &
            (df['dropoff_longitude'].between(-74.3, -73.7)) &
            (df['dropoff_latitude'].between(40.5, 41.0))
        ].copy()
        
        # Remove invalid trip durations (if available)
        if 'trip_duration' in df.columns:
            df = df[df['trip_duration'].between(30, 7200)].copy()  # 30 seconds to 2 hours
        
        # Remove invalid passenger counts
        df = df[df['passenger_count'].between(1, 6)].copy()
        
        print(f"Cleaned data shape: {df.shape}")
        return df
    
    def save_processed_data(self, df, filename):
        """Save processed data"""
        file_path = os.path.join(self.processed_dir, filename)
        df.to_csv(file_path, index=False)
        print(f"Processed data saved to {file_path}")
        
    def load_all_external_data(self):
        """Load or create all external data sources"""
        external_data = {}
        
        # Weather data
        weather_path = os.path.join(self.external_dir, 'weather_data.csv')
        if os.path.exists(weather_path):
            external_data['weather'] = pd.read_csv(weather_path)
            external_data['weather']['date'] = pd.to_datetime(external_data['weather']['date']).dt.date
        else:
            external_data['weather'] = self.create_weather_data()
        
        # Holiday data
        holiday_path = os.path.join(self.external_dir, 'holiday_data.csv')
        if os.path.exists(holiday_path):
            external_data['holidays'] = pd.read_csv(holiday_path)
            external_data['holidays']['date'] = pd.to_datetime(external_data['holidays']['date']).dt.date
        else:
            external_data['holidays'] = self.create_holiday_data()
        
        return external_data

if __name__ == "__main__":
    processor = DataProcessor()
    
    # Create sample external data
    weather_data = processor.create_weather_data()
    holiday_data = processor.create_holiday_data()
    
    print("Data processing setup complete!")
