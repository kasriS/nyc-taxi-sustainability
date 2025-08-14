import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins

class TaxiVisualizer:
    def __init__(self):
        # Set style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # Color palettes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8'
        }
    
    def plot_trip_duration_distribution(self, df, log_scale=True):
        """Plot trip duration distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        if log_scale and 'log_trip_duration' not in df.columns:
            df['log_trip_duration'] = np.log1p(df['trip_duration'])
        
        # Raw distribution
        ax1.hist(df['trip_duration'], bins=50, alpha=0.7, color=self.colors['primary'])
        ax1.set_title('Trip Duration Distribution (Raw)')
        ax1.set_xlabel('Trip Duration (seconds)')
        ax1.set_ylabel('Frequency')
        
        # Log distribution
        if log_scale:
            ax2.hist(df['log_trip_duration'], bins=50, alpha=0.7, color=self.colors['secondary'])
            ax2.set_title('Trip Duration Distribution (Log Scale)')
            ax2.set_xlabel('Log(Trip Duration)')
            ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        return fig
    
    def plot_temporal_patterns(self, df):
        """Plot temporal patterns in trip duration"""
        # Prepare data
        if 'hour' not in df.columns:
            df['hour'] = df['pickup_datetime'].dt.hour
        if 'weekday' not in df.columns:
            df['weekday'] = df['pickup_datetime'].dt.weekday
        if 'log_trip_duration' not in df.columns:
            df['log_trip_duration'] = np.log1p(df['trip_duration'])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Hourly pattern
        hourly_avg = df.groupby('hour')['log_trip_duration'].mean()
        axes[0,0].plot(hourly_avg.index, hourly_avg.values, 'o-', color=self.colors['primary'], linewidth=2)
        axes[0,0].set_title('Average Trip Duration by Hour')
        axes[0,0].set_xlabel('Hour of Day')
        axes[0,0].set_ylabel('Log(Trip Duration)')
        axes[0,0].grid(True)
        
        # Weekly pattern
        weekday_avg = df.groupby('weekday')['log_trip_duration'].mean()
        weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        axes[0,1].plot(range(7), weekday_avg.values, 'o-', color=self.colors['secondary'], linewidth=2)
        axes[0,1].set_title('Average Trip Duration by Day of Week')
        axes[0,1].set_xlabel('Day of Week')
        axes[0,1].set_ylabel('Log(Trip Duration)')
        axes[0,1].set_xticks(range(7))
        axes[0,1].set_xticklabels(weekday_names)
        axes[0,1].grid(True)
        
        # Heatmap: Hour vs Weekday
        pivot_data = df.groupby(['hour', 'weekday'])['log_trip_duration'].mean().unstack()
        sns.heatmap(pivot_data, ax=axes[1,0], cmap='YlOrRd', annot=False)
        axes[1,0].set_title('Trip Duration Heatmap (Hour vs Weekday)')
        axes[1,0].set_ylabel('Hour of Day')
        axes[1,0].set_xlabel('Day of Week')
        
        # Monthly pattern (if available)
        if 'month' in df.columns:
            monthly_avg = df.groupby('month')['log_trip_duration'].mean()
            axes[1,1].plot(monthly_avg.index, monthly_avg.values, 'o-', color=self.colors['success'], linewidth=2)
            axes[1,1].set_title('Average Trip Duration by Month')
            axes[1,1].set_xlabel('Month')
            axes[1,1].set_ylabel('Log(Trip Duration)')
            axes[1,1].grid(True)
        else:
            axes[1,1].text(0.5, 0.5, 'Monthly data not available', ha='center', va='center', transform=axes[1,1].transAxes)
        
        plt.tight_layout()
        return fig
    
    def plot_geographic_distribution(self, df, sample_size=5000):
        """Plot geographic distribution of pickups and dropoffs"""
        # Sample data for performance
        df_sample = df.sample(min(sample_size, len(df)), random_state=42)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Pickup locations
        ax1.scatter(df_sample['pickup_longitude'], df_sample['pickup_latitude'], 
                   alpha=0.6, s=1, c=self.colors['primary'])
        ax1.set_title('Pickup Locations')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_xlim(-74.05, -73.75)
        ax1.set_ylim(40.60, 40.90)
        
        # Dropoff locations
        ax2.scatter(df_sample['dropoff_longitude'], df_sample['dropoff_latitude'], 
                   alpha=0.6, s=1, c=self.colors['secondary'])
        ax2.set_title('Dropoff Locations')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_xlim(-74.05, -73.75)
        ax2.set_ylim(40.60, 40.90)
        
        plt.tight_layout()
        return fig
    
    def plot_distance_vs_duration(self, df, sample_size=5000):
        """Plot distance vs duration relationship"""
        df_sample = df.sample(min(sample_size, len(df)), random_state=42)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Distance vs Duration (log scale)
        if 'haversine_dist' in df_sample.columns and 'log_trip_duration' in df_sample.columns:
            ax1.scatter(df_sample['haversine_dist'], df_sample['log_trip_duration'], 
                       alpha=0.5, s=10, c=self.colors['primary'])
            ax1.set_xlabel('Haversine Distance (km)')
            ax1.set_ylabel('Log(Trip Duration)')
            ax1.set_title('Distance vs Trip Duration')
            
            # Add trend line
            z = np.polyfit(df_sample['haversine_dist'], df_sample['log_trip_duration'], 1)
            p = np.poly1d(z)
            ax1.plot(df_sample['haversine_dist'].sort_values(), 
                    p(df_sample['haversine_dist'].sort_values()), 
                    "r--", alpha=0.8)
        
        # Distance distribution
        if 'haversine_dist' in df_sample.columns:
            ax2.hist(df_sample['haversine_dist'], bins=50, alpha=0.7, color=self.colors['secondary'])
            ax2.set_xlabel('Haversine Distance (km)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Distance Distribution')
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, feature_importance_dict):
        """Plot feature importance for different models"""
        if not feature_importance_dict:
            print("No feature importance data available")
            return None
        
        n_models = len(feature_importance_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, importance) in enumerate(feature_importance_dict.items()):
            # Sort features by importance
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
            features, values = zip(*sorted_features)
            
            # Plot
            y_pos = np.arange(len(features))
            axes[i].barh(y_pos, values, color=self.colors['primary'])
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels(features)
            axes[i].set_xlabel('Feature Importance')
            axes[i].set_title(f'{model_name.upper()} Feature Importance')
            axes[i].invert_yaxis()
        
        plt.tight_layout()
        return fig
    
    def plot_model_performance(self, cv_scores, test_metrics=None):
        """Plot model performance comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # CV Scores
        models = list(cv_scores.keys())
        scores = list(cv_scores.values())
        
        axes[0].bar(models, scores, color=[self.colors['primary'], self.colors['secondary'], 
                                          self.colors['success'], self.colors['warning']])
        axes[0].set_title('Cross-Validation RMSE Scores')
        axes[0].set_ylabel('RMSE')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Test metrics (if available)
        if test_metrics:
            metrics_names = list(test_metrics.keys())
            metrics_values = list(test_metrics.values())
            
            axes[1].bar(metrics_names, metrics_values, color=self.colors['info'])
            axes[1].set_title('Test Set Performance')
            axes[1].set_ylabel('Score')
            axes[1].tick_params(axis='x', rotation=45)
        else:
            axes[1].text(0.5, 0.5, 'Test metrics not available', 
                        ha='center', va='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        return fig
    
    def plot_co2_emissions_comparison(self, sustainability_results):
        """Plot CO2 emissions comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Emissions by transport mode
        transport_modes = ['Current Taxi', 'Optimized Route', 'Electric Taxi', 'Public Transport']
        emissions = [
            sustainability_results['current_co2_kg'],
            sustainability_results['optimized_co2_kg'],
            sustainability_results['electric_taxi_co2_kg'],
            sustainability_results['public_transport_co2_kg']
        ]
        
        colors = [self.colors['danger'], self.colors['warning'], 
                 self.colors['success'], self.colors['info']]
        
        bars = ax1.bar(transport_modes, emissions, color=colors)
        ax1.set_title('Total CO2 Emissions by Transport Mode')
        ax1.set_ylabel('CO2 Emissions (kg)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, emissions):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}', ha='center', va='bottom')
        
        # CO2 reduction potential
        reduction_scenarios = ['Route Optimization', 'Electric Conversion', 'Public Transport']
        reductions = [
            sustainability_results['co2_reduction_kg'],
            sustainability_results['current_co2_kg'] - sustainability_results['electric_taxi_co2_kg'],
            sustainability_results['current_co2_kg'] - sustainability_results['public_transport_co2_kg']
        ]
        
        bars2 = ax2.bar(reduction_scenarios, reductions, 
                       color=[self.colors['warning'], self.colors['success'], self.colors['info']])
        ax2.set_title('CO2 Reduction Potential')
        ax2.set_ylabel('CO2 Reduction (kg)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars2, reductions):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def create_interactive_map(self, df, sample_size=1000):
        """Create interactive map with pickup/dropoff locations"""
        # Sample data
        df_sample = df.sample(min(sample_size, len(df)), random_state=42)
        
        # Center of NYC
        nyc_center = [40.7128, -74.0060]
        
        # Create map
        m = folium.Map(location=nyc_center, zoom_start=11)
        
        # Add pickup points
        for _, row in df_sample.iterrows():
            folium.CircleMarker(
                location=[row['pickup_latitude'], row['pickup_longitude']],
                radius=2,
                popup=f"Pickup: {row['pickup_datetime']}",
                color='blue',
                fill=True,
                weight=1
            ).add_to(m)
            
            # Add dropoff points
            folium.CircleMarker(
                location=[row['dropoff_latitude'], row['dropoff_longitude']],
                radius=2,
                popup=f"Dropoff",
                color='red',
                fill=True,
                weight=1
            ).add_to(m)
            
            # Draw line between pickup and dropoff
            folium.PolyLine(
                locations=[[row['pickup_latitude'], row['pickup_longitude']],
                          [row['dropoff_latitude'], row['dropoff_longitude']]],
                color='green',
                weight=1,
                opacity=0.5
            ).add_to(m)
        
        return m
    
    def create_interactive_dashboard_plots(self, df, predictions=None, sustainability_results=None):
        """Create interactive plots for dashboard"""
        plots = {}
        
        # 1. Trip Duration Distribution
        fig_duration = px.histogram(
            df, x='trip_duration', nbins=50,
            title='Trip Duration Distribution',
            labels={'trip_duration': 'Trip Duration (seconds)', 'count': 'Frequency'}
        )
        plots['duration_dist'] = fig_duration
        
        # 2. Hourly Pattern
        if 'hour' not in df.columns:
            df['hour'] = df['pickup_datetime'].dt.hour
        
        hourly_data = df.groupby('
