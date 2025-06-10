#!/usr/bin/env python3
"""
FFT Analysis Script for Mask Morphology Analysis

This script analyzes mask images using Fast Fourier Transform (FFT) to identify
potentially erroneous shapes based on their frequency domain characteristics.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from scipy import ndimage
from scipy.fft import fft2, fftshift, fftfreq
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import argparse
import warnings
warnings.filterwarnings('ignore')

class FFTMorphologyAnalyzer:
    def __init__(self, output_folder):
        """
        Initialize the FFT analyzer with the output folder path.
        
        Args:
            output_folder (str): Path to the output folder containing filtered masks
        """
        self.output_folder = output_folder
        self.filtered_folder = os.path.join(output_folder, 'filtered')
        self.metadata_path = os.path.join(self.filtered_folder, 'metadata.csv')
        
        # Check if paths exist
        if not os.path.exists(self.filtered_folder):
            raise FileNotFoundError(f"Filtered folder not found: {self.filtered_folder}")
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
            
        # Load metadata
        self.metadata = pd.read_csv(self.metadata_path)
        print(f"Loaded metadata for {len(self.metadata)} masks")
        
        # Initialize results storage
        self.fft_features = []
        self.analysis_results = {}
        
    def load_mask_image(self, filename):
        """
        Load and preprocess a mask image.
        
        Args:
            filename (str): Name of the mask image file
            
        Returns:
            numpy.ndarray: Binary mask image
        """
        image_path = os.path.join(self.filtered_folder, filename)
        if not os.path.exists(image_path):
            return None
            
        # Load image and convert to binary mask
        img = Image.open(image_path).convert('L')
        mask = np.array(img)
        
        # Threshold to ensure binary mask
        mask = (mask > 128).astype(np.uint8)
        
        return mask
    
    def compute_fft_features(self, mask):
        """
        Compute FFT-based morphological features from a binary mask.
        
        Args:
            mask (numpy.ndarray): Binary mask image
            
        Returns:
            dict: Dictionary of FFT-based features
        """
        if mask is None or mask.sum() == 0:
            return None
            
        # Compute FFT
        fft = fft2(mask.astype(np.float64))
        fft_shifted = fftshift(fft)
        magnitude_spectrum = np.abs(fft_shifted)
        
        # Log transform for better visualization
        log_magnitude = np.log(magnitude_spectrum + 1)
        
        # Get frequency coordinates
        rows, cols = mask.shape
        freq_y = fftfreq(rows)
        freq_x = fftfreq(cols)
        
        # Create frequency grid
        fy, fx = np.meshgrid(freq_y, freq_x, indexing='ij')
        freq_magnitude = np.sqrt(fx**2 + fy**2)
        
        # Compute radial frequency profile
        max_freq = np.max(freq_magnitude)
        freq_bins = np.linspace(0, max_freq, 50)
        radial_profile = []
        
        for i in range(len(freq_bins) - 1):
            mask_ring = (freq_magnitude >= freq_bins[i]) & (freq_magnitude < freq_bins[i+1])
            if np.any(mask_ring):
                radial_profile.append(np.mean(magnitude_spectrum[mask_ring]))
            else:
                radial_profile.append(0)
        
        radial_profile = np.array(radial_profile)
        
        # Compute features
        features = {
            'total_energy': np.sum(magnitude_spectrum**2),
            'dc_component': magnitude_spectrum[rows//2, cols//2],
            'high_freq_energy': np.sum(magnitude_spectrum[freq_magnitude > 0.1]**2),
            'low_freq_energy': np.sum(magnitude_spectrum[freq_magnitude <= 0.1]**2),
            'freq_centroid_x': np.sum(fx * magnitude_spectrum**2) / np.sum(magnitude_spectrum**2),
            'freq_centroid_y': np.sum(fy * magnitude_spectrum**2) / np.sum(magnitude_spectrum**2),
            'freq_spread_x': np.sqrt(np.sum((fx - np.sum(fx * magnitude_spectrum**2) / np.sum(magnitude_spectrum**2))**2 * magnitude_spectrum**2) / np.sum(magnitude_spectrum**2)),
            'freq_spread_y': np.sqrt(np.sum((fy - np.sum(fy * magnitude_spectrum**2) / np.sum(magnitude_spectrum**2))**2 * magnitude_spectrum**2) / np.sum(magnitude_spectrum**2)),
            'spectral_entropy': -np.sum((magnitude_spectrum**2 / np.sum(magnitude_spectrum**2)) * np.log(magnitude_spectrum**2 / np.sum(magnitude_spectrum**2) + 1e-10)),
            'peak_frequency': freq_magnitude.flatten()[np.argmax(magnitude_spectrum.flatten())],
            'radial_profile_peak': np.max(radial_profile),
            'radial_profile_mean': np.mean(radial_profile),
            'radial_profile_std': np.std(radial_profile),
            'high_low_freq_ratio': np.sum(magnitude_spectrum[freq_magnitude > 0.1]**2) / (np.sum(magnitude_spectrum[freq_magnitude <= 0.1]**2) + 1e-10),
            'magnitude_spectrum': magnitude_spectrum,
            'log_magnitude': log_magnitude,
            'radial_profile': radial_profile
        }
        
        return features
    
    def analyze_all_masks(self):
        """
        Analyze all mask images and compute FFT features.
        """
        print("Analyzing masks with FFT...")
        
        for idx, row in self.metadata.iterrows():
            filename = row['filename']
            mask_id = row['id']
            
            # Load mask
            mask = self.load_mask_image(filename)
            if mask is None:
                print(f"Warning: Could not load mask {filename}")
                continue
                
            # Compute FFT features
            fft_features = self.compute_fft_features(mask)
            if fft_features is None:
                print(f"Warning: Could not compute FFT features for {filename}")
                continue
                
            # Combine with metadata
            feature_dict = {
                'filename': filename,
                'id': mask_id,
                'status': row['status'],
                'area': row['area'],
                'circularity': row['circularity'],
                'bbox_w': row['bbox_w'],
                'bbox_h': row['bbox_h']
            }
            
            # Add FFT features (excluding arrays)
            for key, value in fft_features.items():
                if key not in ['magnitude_spectrum', 'log_magnitude', 'radial_profile']:
                    feature_dict[key] = value
                    
            self.fft_features.append(feature_dict)
            
            # Store full results for visualization
            self.analysis_results[mask_id] = {
                'metadata': row,
                'fft_features': fft_features,
                'mask': mask
            }
            
            if len(self.fft_features) % 50 == 0:
                print(f"Processed {len(self.fft_features)} masks...")
                
        print(f"Completed analysis of {len(self.fft_features)} masks")
        
        # Convert to DataFrame
        self.fft_df = pd.DataFrame(self.fft_features)
        
    def identify_anomalous_shapes(self, n_clusters=3):
        """
        Identify potentially anomalous shapes using clustering on FFT features.
        
        Args:
            n_clusters (int): Number of clusters for K-means
            
        Returns:
            pandas.DataFrame: DataFrame with anomaly scores and cluster assignments
        """
        print("Identifying anomalous shapes...")
        
        # Select features for clustering
        feature_cols = [
            'total_energy', 'high_freq_energy', 'low_freq_energy',
            'freq_spread_x', 'freq_spread_y', 'spectral_entropy',
            'high_low_freq_ratio', 'radial_profile_peak', 'radial_profile_std'
        ]
        
        # Prepare data
        X = self.fft_df[feature_cols].fillna(0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Compute distances to cluster centers (anomaly scores)
        distances = np.min(kmeans.transform(X_scaled), axis=1)
        
        # Add results to dataframe
        self.fft_df['cluster'] = clusters
        self.fft_df['anomaly_score'] = distances
        
        # Identify anomalies (high anomaly score or failed status)
        anomaly_threshold = np.percentile(distances, 90)  # Top 10% as anomalies
        self.fft_df['is_anomaly'] = (self.fft_df['anomaly_score'] > anomaly_threshold) | (self.fft_df['status'] == 'FAILED')
        
        print(f"Identified {self.fft_df['is_anomaly'].sum()} potentially anomalous shapes")
        
        return self.fft_df
    
    def create_visualizations(self, output_dir=None):
        """
        Create comprehensive visualizations of the FFT analysis results.
        
        Args:
            output_dir (str): Directory to save plots (default: output_folder)
        """
        if output_dir is None:
            output_dir = self.output_folder
            
        print("Creating visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. FFT Feature Distribution
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.ravel()
        
        feature_cols = [
            'total_energy', 'high_freq_energy', 'spectral_entropy',
            'freq_spread_x', 'freq_spread_y', 'high_low_freq_ratio',
            'radial_profile_peak', 'radial_profile_std', 'anomaly_score'
        ]
        
        for i, col in enumerate(feature_cols):
            if i < len(axes):
                axes[i].hist(self.fft_df[col], bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{col.replace("_", " ").title()}')
                axes[i].set_xlabel('Value')
                axes[i].set_ylabel('Frequency')
                
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fft_feature_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Anomaly Detection Results
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Scatter plot: Circularity vs Anomaly Score
        scatter = axes[0,0].scatter(self.fft_df['circularity'], self.fft_df['anomaly_score'], 
                                  c=self.fft_df['is_anomaly'], cmap='coolwarm', alpha=0.6)
        axes[0,0].set_xlabel('Circularity')
        axes[0,0].set_ylabel('Anomaly Score')
        axes[0,0].set_title('Circularity vs Anomaly Score')
        plt.colorbar(scatter, ax=axes[0,0])
        
        # Box plot: Status vs Anomaly Score
        status_order = ['PASSED', 'FAILED']
        sns.boxplot(data=self.fft_df, x='status', y='anomaly_score', order=status_order, ax=axes[0,1])
        axes[0,1].set_title('Anomaly Score by Status')
        
        # Cluster visualization
        scatter2 = axes[1,0].scatter(self.fft_df['spectral_entropy'], self.fft_df['high_low_freq_ratio'],
                                    c=self.fft_df['cluster'], cmap='viridis', alpha=0.6)
        axes[1,0].set_xlabel('Spectral Entropy')
        axes[1,0].set_ylabel('High/Low Frequency Ratio')
        axes[1,0].set_title('Clusters in Feature Space')
        plt.colorbar(scatter2, ax=axes[1,0])
        
        # Area vs High Frequency Energy
        scatter3 = axes[1,1].scatter(self.fft_df['area'], self.fft_df['high_freq_energy'],
                                    c=self.fft_df['is_anomaly'], cmap='coolwarm', alpha=0.6)
        axes[1,1].set_xlabel('Area')
        axes[1,1].set_ylabel('High Frequency Energy')
        axes[1,1].set_title('Area vs High Frequency Energy')
        axes[1,1].set_xscale('log')
        axes[1,1].set_yscale('log')
        plt.colorbar(scatter3, ax=axes[1,1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'anomaly_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Sample FFT Visualizations
        self.visualize_sample_ffts(output_dir)
        
        # 4. Correlation Matrix
        feature_cols_corr = [
            'circularity', 'area', 'total_energy', 'high_freq_energy', 
            'spectral_entropy', 'freq_spread_x', 'freq_spread_y', 
            'high_low_freq_ratio', 'anomaly_score'
        ]
        
        plt.figure(figsize=(10, 8))
        correlation_matrix = self.fft_df[feature_cols_corr].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_correlation.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}")
    
    def visualize_sample_ffts(self, output_dir, n_samples=12):
        """
        Visualize FFT spectra for sample masks (normal and anomalous).
        
        Args:
            output_dir (str): Directory to save plots
            n_samples (int): Number of samples to visualize
        """
        # Select samples
        normal_samples = self.fft_df[~self.fft_df['is_anomaly']].sample(n=min(n_samples//2, 6), random_state=42)
        anomaly_samples = self.fft_df[self.fft_df['is_anomaly']].sample(n=min(n_samples//2, 6), random_state=42)
        
        samples = pd.concat([normal_samples, anomaly_samples])
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, (_, row) in enumerate(samples.iterrows()):
            if i >= len(axes):
                break
                
            mask_id = row['id']
            if mask_id in self.analysis_results:
                result = self.analysis_results[mask_id]
                log_magnitude = result['fft_features']['log_magnitude']
                
                im = axes[i].imshow(log_magnitude, cmap='hot', origin='lower')
                axes[i].set_title(f'ID: {mask_id}\nStatus: {row["status"]}\nAnomaly: {row["is_anomaly"]}')
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        
        # Hide unused subplots
        for i in range(len(samples), len(axes)):
            axes[i].axis('off')
            
        plt.suptitle('FFT Magnitude Spectra (Log Scale)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sample_fft_spectra.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, output_dir=None):
        """
        Generate a comprehensive analysis report.
        
        Args:
            output_dir (str): Directory to save report (default: output_folder)
        """
        if output_dir is None:
            output_dir = self.output_folder
            
        report_path = os.path.join(output_dir, 'fft_morphology_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("FFT Morphology Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total masks analyzed: {len(self.fft_df)}\n")
            f.write(f"Passed masks: {(self.fft_df['status'] == 'PASSED').sum()}\n")
            f.write(f"Failed masks: {(self.fft_df['status'] == 'FAILED').sum()}\n")
            f.write(f"Identified anomalies: {self.fft_df['is_anomaly'].sum()}\n")
            f.write(f"Anomaly rate: {self.fft_df['is_anomaly'].mean():.2%}\n\n")
            
            # Feature statistics
            f.write("FFT FEATURE STATISTICS\n")
            f.write("-" * 25 + "\n")
            feature_stats = self.fft_df[['total_energy', 'high_freq_energy', 'spectral_entropy', 
                                       'high_low_freq_ratio', 'anomaly_score']].describe()
            f.write(feature_stats.to_string())
            f.write("\n\n")
            
            # Anomaly analysis
            f.write("ANOMALY ANALYSIS\n")
            f.write("-" * 16 + "\n")
            anomalies = self.fft_df[self.fft_df['is_anomaly']].sort_values('anomaly_score', ascending=False)
            f.write("Top 10 most anomalous masks:\n")
            for _, row in anomalies.head(10).iterrows():
                f.write(f"ID: {row['id']}, File: {row['filename']}, Score: {row['anomaly_score']:.3f}, "
                       f"Status: {row['status']}, Circularity: {row['circularity']:.3f}\n")
            
            f.write("\n")
            
            # Cluster analysis
            f.write("CLUSTER ANALYSIS\n")
            f.write("-" * 16 + "\n")
            cluster_stats = self.fft_df.groupby('cluster').agg({
                'anomaly_score': ['mean', 'std'],
                'circularity': ['mean', 'std'],
                'area': ['mean', 'std'],
                'is_anomaly': 'sum'
            }).round(3)
            f.write(cluster_stats.to_string())
            f.write("\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 15 + "\n")
            f.write("1. Review masks with high anomaly scores (>90th percentile)\n")
            f.write("2. Pay special attention to masks with low circularity and high frequency content\n")
            f.write("3. Consider re-segmentation for masks in the anomalous cluster\n")
            f.write("4. Validate masks with unusual spectral entropy values\n")
            
        print(f"Report saved to {report_path}")
        
        # Save detailed results
        results_path = os.path.join(output_dir, 'fft_analysis_results.csv')
        self.fft_df.to_csv(results_path, index=False)
        print(f"Detailed results saved to {results_path}")

def main():
    parser = argparse.ArgumentParser(description='FFT Morphology Analysis for Mask Images')
    parser.add_argument('output_folder', help='Path to the output folder containing filtered masks')
    parser.add_argument('--clusters', type=int, default=3, help='Number of clusters for anomaly detection')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = FFTMorphologyAnalyzer(args.output_folder)
        
        # Perform analysis
        analyzer.analyze_all_masks()
        analyzer.identify_anomalous_shapes(n_clusters=args.clusters)
        
        # Generate visualizations and report
        analyzer.create_visualizations()
        analyzer.generate_report()
        
        print("\nFFT morphology analysis completed successfully!")
        print(f"Results saved in: {args.output_folder}")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
