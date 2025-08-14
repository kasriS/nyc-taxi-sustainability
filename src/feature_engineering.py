import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pandas.tseries.holiday import USFederalHolidayCalendar
import holidays

class FeatureEngine:
    def __init__(self):
        self.kmeans_pickup = None
        self.kmeans_dropoff = None
        self.zone_hour_avg = {}
        self.zone_median = {}
        self.pickup_freq = {}
        
    def create_temporal_features(self, df):
        """Create time-based features"""
        df = df.copy()
        
        # Basic time features
        df['hour'] = df['pickup_datetime'].dt.hour
        df['weekday'] = df['pickup_datetime'].dt.weekday
        df['month'] = df['pickup_datetime'].dt.month
        df['day'] = df['pickup_datetime'].dt.day
        df['year'] = df['pickup_datetime'].dt.year
        
        # Cyclical encoding for temporal features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Rush hour indicators
        df['rush_hour'] = (
            ((df['hour'] >= 7) & (df['hour'] <= 9)) |
            ((df['hour'] >= 16) & (df['hour'] <= 19))
        ).astype(int)
        
        # Weekend indicator
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        
        # Late night indicator
        df['late_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        
        return df
    
    def create_distance_features(self, df):
        """Create distance-based features"""
        df = df.copy()
        
        # Manhattan distance
        df['manhattan_dist'] = (
            np.abs(df['pickup_latitude'] - df['dropoff_latitude']) +
            np.abs(df['pickup_longitude'] - df['dropoff_longitude'])
        )
        
        # Haversine distance
        R = 6371  # Earth's radius in kilometers
        lat1 = np.radians(df['pickup_latitude'])
        lat2 = np.radians(df['dropoff_latitude'])
        dlat = lat2 - lat1
        dlon = np.radians(df['dropoff_longitude'] - df['pickup_longitude'])
        
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        df['haversine_dist'] = 2 * R * np.arcsin(np.sqrt(a))
        
        # Bearing (direction of travel)
        df['bearing'] = np.arctan2(
            np.sin(np.radians(df['dropoff_longitude'] - df['pickup_longitude'])) *
            np.cos(np.radians(df['dropoff_latitude'])),
            np.cos(np.radians(df['pickup_latitude'])) * np.sin(np.radians(df['dropoff_latitude'])) -
            np.sin(np.radians(df['pickup_latitude'])) * np.cos(np.radians(df['dropoff_latitude'])) *
            np.cos(np.radians(df['dropoff_longitude'] - df['pickup_longitude']))
        )
        
        # Convert bearing to degrees and create direction bins
        df['bearing_deg'] = np.degrees(df['bearing']) % 360
        df['direction_bin'] = pd.cut(
            df['bearing_deg'], 
            bins=[0, 45, 135, 225, 315, 360], 
            labels=['E', 'N', 'W', 'S', 'E2']
        )
        df['direction_bin'] = df['direction_bin'].map({'N': 0, 'E': 1, 'S': 2, 'W': 3, 'E2': 1})
        
        # Distance bins
        df['distance_bin'] = pd.cut(
            df['haversine_dist'], 
            bins=[0, 1, 2, 5, 10, 20, 50], 
            labels=False
        )
        
        return df
    
    def create_location_features(self, df):
        """Create location-based features"""
        df = df.copy()
        
        # NYC landmarks (approximate coordinates)
        landmarks = {
            'times_square': (40.7580, -73.9855),
            'central_park': (40.7829, -73.9654),
            'jfk_airport': (40.6413, -73.7781),
            'lga_airport': (40.7769, -73.8740),
            'penn_station': (40.7505, -73.9934),
            'grand_central': (40.7527, -73.9772)
        }
        
        # Distance to landmarks
        for landmark, (lat, lon) in landmarks.items():
            # Pickup distance to landmark
            df[f'pickup_dist_{landmark}'] = np.sqrt(
                (df['pickup_latitude'] - lat)**2 + 
                (df['pickup_longitude'] - lon)**2
            )
            # Dropoff distance to landmark
            df[f'dropoff_dist_{landmark}'] = np.sqrt(
                (df['dropoff_latitude'] - lat)**2 + 
                (df['dropoff_longitude'] - lon)**2
            )
        
        # NYC boroughs (simplified)
        def get_borough(lat, lon):
            if lat > 40.8 and lon > -73.9:
                return 'Bronx'
            elif lat > 40.75 and lon < -73.95:
                return 'Manhattan'
            elif lat < 40.68 and lon < -74.0:
                return 'Staten_Island'
            elif lat < 40.72 and lon > -73.9:
                return 'Brooklyn'
            else:
                return 'Queens'
        
        df['pickup_borough'] = df.apply(
            lambda row: get_borough(row['pickup_latitude'], row['pickup_longitude']), 
            axis=1
        )
        df['dropoff_borough'] = df.apply(
            lambda row: get_borough(row['dropoff_latitude'], row['dropoff_longitude']), 
            axis=1
        )
        
        # Same borough indicator
        df['same_borough'] = (df['pickup_borough'] == df['dropoff_borough']).astype(int)
        
        # One-hot encode boroughs
        pickup_borough_dummies = pd.get_dummies(df['pickup_borough'], prefix='pickup')
        dropoff_borough_dummies = pd.get_dummies(df['dropoff_borough'], prefix='dropoff')
        
        df = pd.concat([df, pickup_borough_dummies, dropoff_borough_dummies], axis=1)
        
        return df
    
    def create_clustering_features(self, df):
        """Create clustering-based features"""
        df = df.copy()
        
        # Pickup location clustering
        pickup_coords = df[['pickup_latitude', 'pickup_longitude']]
        if self.kmeans_pickup is None:
            self.kmeans_pickup = KMeans(n_clusters=30, random_state=42)
            self.kmeans_pickup.fit(pickup_coords)
        
        df['pickup_zone'] = self.kmeans_pickup.predict(pickup_coords)
        
        # Dropoff location clustering
        dropoff_coords = df[['dropoff_latitude', 'dropoff_longitude']]
        if self.kmeans_dropoff is None:
            self.kmeans_dropoff = KMeans(n_clusters=30, random_state=42)
            self.kmeans_dropoff.fit(dropoff_coords)
        
        df['dropoff_zone'] = self.kmeans_dropoff.predict(dropoff_coords)
        
        # Pickup density (geohash-like)
        df['pickup_geohash'] = df.apply(
            lambda row: f"{round(row['pickup_latitude'] / 0.01, 2)}_{round(row['pickup_longitude'] / 0.01, 2)}", 
            axis=1
        )
        
        # Calculate pickup frequency if not already done
        if not self.pickup_freq:
            self.pickup_freq = df['pickup_geohash'].value_counts().to_dict()
        
        df['pickup_density'] = df['pickup_geohash'].map(self.pickup_freq)
        
        return df
    
    def create_weather_features(self, df, weather_data):
        """Merge weather features"""
        if weather_data is None:
            return df
        
        df = df.copy()
        df['date'] = df['pickup_datetime'].dt.date
        
        # Merge with weather data
        weather_cols = ['date', 'temperature', 'humidity', 'wind_speed', 
                       'precipitation', 'snow_fall', 'precip_intensity', 'visibility']
        
        df = df.merge(
            weather_data[weather_cols], 
            on='date', 
            how='left'
        )
        
        # Fill missing weather data with median values
        weather_numeric_cols = ['temperature', 'humidity', 'wind_speed', 
                               'precipitation', 'snow_fall', 'precip_intensity', 'visibility']
        
        for col in weather_numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Weather interaction features
        if 'precip_intensity' in df.columns and 'manhattan_dist' in df.columns:
            df['dist_precip'] = df['manhattan_dist'] * df['precip_intensity']
        
        return df
    
    def create_holiday_features(self, df, holiday_data):
        """Add holiday features"""
        if holiday_data is None:
            return df
        
        df = df.copy()
        df['date'] = df['pickup_datetime'].dt.date
        
        # Merge with holiday data
        df = df.merge(
            holiday_data[['date', 'is_holiday']], 
            on='date', 
            how='left'
        )
        
        df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)
        
        # Federal holidays using pandas
        calendar = USFederalHolidayCalendar()
        holidays_dates = calendar.holidays(
            start=df['pickup_datetime'].min(), 
            end=df['pickup_datetime'].max()
        )
        df['is_federal_holiday'] = df['pickup_datetime'].dt.date.isin(holidays_dates.date).astype(int)
        
        return df
    
    def create_advanced_features(self, df):
        """Create advanced engineered features"""
        df = df.copy()
        
        # Passenger features
        df['high_passenger'] = (df['passenger_count'] > 2).astype(int)
        df['solo_trip'] = (df['passenger_count'] == 1).astype(int)
        
        # Store and forward flag
        if 'store_and_fwd_flag' in df.columns:
            df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map({'N': 0, 'Y': 1}).fillna(0).astype(int)
        
        # Interaction features
        df['hour_weekday'] = df['hour'] * df['weekday']
        df['hour_passenger'] = df['hour'] * df['passenger_count']
        
        # Time-based patterns
        df['pickup_hour'] = df['pickup_datetime'].dt.hour
        df['pickup_time_bin'] = df['pickup_datetime'].dt.floor('H')
        
        return df
    
    def create_target_encoding_features(self, df, target_col='trip_duration'):
        """Create target-encoded features (for training data)"""
        if target_col not in df.columns:
            return df
        
        df = df.copy()
        
        # Zone-hour average duration
        if 'pickup_zone' in df.columns and 'pickup_hour' in df.columns:
            self.zone_hour_avg = df.groupby(['pickup_zone', 'pickup_hour'])[target_col].mean().to_dict()
            df['zone_hour_duration'] = df[['pickup_zone', 'pickup_hour']].apply(
                lambda row: self.zone_hour_avg.get((row['pickup_zone'], row['pickup_hour']), df[target_col].mean()),
                axis=1
            )
        
        # Zone median duration
        if 'pickup_zone' in df.columns:
            self.zone_median = df.groupby('pickup_zone')[target_col].median().to_dict()
            df['pickup_zone_median'] = df['pickup_zone'].map(self.zone_median)
            df['pickup_zone_median'] = df['pickup_zone_median'].fillna(df[target_col].median())
        
        # Hourly trip counts per zone
        if 'pickup_zone' in df.columns and 'pickup_time_bin' in df.columns:
            trip_counts = df.groupby(['pickup_zone', 'pickup_time_bin']).size().rename('pickup_hourly_count')
            df = df.merge(trip_counts, on=['pickup_zone', 'pickup_time_bin'], how='left')
            df['pickup_hourly_count'] = df['pickup_hourly_count'].fillna(0)
        
        return df
    
    def apply_target_encoding(self, df):
        """Apply pre-computed target encodings to new data"""
        df = df.copy()
        
        # Apply zone-hour average
        if self.zone_hour_avg and 'pickup_zone' in df.columns and 'pickup_hour' in df.columns:
            overall_mean = list(self.zone_hour_avg.values())[0] if self.zone_hour_avg else 800
            df['zone_hour_duration'] = df[['pickup_zone', 'pickup_hour']].apply(
                lambda row: self.zone_hour_avg.get((row['pickup_zone'], row['pickup_hour']), overall_mean),
                axis=1
            )
        
        # Apply zone median
        if self.zone_median and 'pickup_zone' in df.columns:
            overall_median = list(self.zone_median.values())[0] if self.zone_median else 600
            df['pickup_zone_median'] = df['pickup_zone'].map(self.zone_median)
            df['pickup_zone_median'] = df['pickup_zone_median'].fillna(overall_median)
        
        return df
    
    def engineer_all_features(self, df, weather_data=None, holiday_data=None, is_training=True):
        """Apply all feature engineering steps"""
        print("Starting feature engineering...")
        
        # Create temporal features
        df = self.create_temporal_features(df)
        print("✓ Temporal features created")
        
        # Create distance features  
        df = self.create_distance_features(df)
        print("✓ Distance features created")
        
        # Create location features
        df = self.create_location_features(df)
        print("✓ Location features created")
        
        # Create clustering features
        df = self.create_clustering_features(df)
        print("✓ Clustering features created")
        
        # Create weather features
        if weather_data is not None:
            df = self.create_weather_features(df, weather_data)
            print("✓ Weather features created")
        
        # Create holiday features
        if holiday_data is not None:
            df = self.create_holiday_features(df, holiday_data)
            print("✓ Holiday features created")
        
        # Create advanced features
        df = self.create_advanced_features(df)
        print("✓ Advanced features created")
        
        # Create target encoding features
        if is_training and 'trip_duration' in df.columns:
            df = self.create_target_encoding_features(df)
            print("✓ Target encoding features created")
        elif not is_training:
            df = self.apply_target_encoding(df)
            print("✓ Target encoding applied")
        
        print(f"Feature engineering complete. Final shape: {df.shape}")
        return df

if __name__ == "__main__":
    # Example usage
    fe = FeatureEngine()
    print("Feature engineering module ready!")
