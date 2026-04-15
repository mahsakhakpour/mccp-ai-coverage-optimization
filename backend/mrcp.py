import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from matplotlib.patches import Circle
import time
import re
from matplotlib.lines import Line2D

def get_user_input():
    print("\n*** Maximum Range Count Problem *** \n ** Mahsa Khakpour **")
    points = []
    print("\nEnter your 2D points (x,y). Type 'done' when finished:")
    while True:
        user_input = input("Enter point (formats accepted: x,y or x y or x\t y etc.): ").strip()
        if user_input.lower() == 'done':
            if len(points) < 2:
                print("Please enter at least 2 points!")
                continue
            break
        
        parts = re.split(r'[,\s]+', user_input.strip())
        if len(parts) == 2:
            try:
                x = float(parts[0])
                y = float(parts[1])
                points.append([x, y])
                print(f"Added point ({x:.6f}, {y:.6f}) | Total points: {len(points)}")
            except ValueError:
                print("Invalid number format! Please enter valid numbers.")
        else:
            print("Invalid format! Please enter exactly two numbers separated by space, tab, or comma.")

    while True:
        try:
            eps = float(input("\nEnter maximum cluster distance (eps): "))
            min_samples = int(input("Enter minimum points per cluster: "))
            radius = float(input("Enter query circle radius: "))
            if eps > 0 and min_samples > 0 and radius > 0:
                break
            print("All values must be positive numbers!")
        except ValueError:
            print("Please enter valid numbers!")

    return np.array(points), eps, min_samples, radius

def sliding_circle_algorithm(points, radius, cluster_points):
    if len(cluster_points) == 0:
        return None, 0, []

    best_center = None
    max_count = 0
    all_circles = []
    
    # Process all clusters but start with the densest ones first
    sorted_clusters = sorted(cluster_points.items(), key=lambda x: len(x[1]), reverse=True)
    
    for cid, cpoints in sorted_clusters:
        current_points = np.array(cpoints)
        
        # Skip clusters that can't possibly contain a better solution
        if len(current_points) < max_count:
            continue
            
        # Use more samples for better exploration (1/10 of points)
        sample_indices = range(0, len(current_points), max(1, len(current_points)//10))
        sample_points = current_points[list(sample_indices)]
        
        cluster_best_center = None
        cluster_best_count = 0
        
        for start_point in sample_points:
            current_center = start_point.copy()
            current_count = np.sum(np.linalg.norm(current_points - current_center, axis=1) <= radius)
            
            # Optimize with more iterations but adaptive step size
            for _ in range(75):  # Increased from 50 to 75
                in_circle = np.linalg.norm(current_points - current_center, axis=1) <= radius
                points_in_circle = current_points[in_circle]
                
                if len(points_in_circle) == 0:
                    break
                    
                new_center = points_in_circle.mean(axis=0)
                direction = new_center - current_center
                step_size = min(radius/2, np.linalg.norm(direction))
                
                if step_size < 1e-6:
                    break
                    
                current_center += direction * (step_size / np.linalg.norm(direction))
                new_count = np.sum(np.linalg.norm(current_points - current_center, axis=1) <= radius)
                
                if new_count > current_count:
                    current_count = new_count
                else:
                    # Reduce step size when no improvement
                    step_size *= 0.7
                    if step_size < 1e-6:
                        break
            
            if current_count > cluster_best_count:
                cluster_best_count = current_count
                cluster_best_center = current_center
        
        # Track all circles found
        if cluster_best_center is not None:
            if cluster_best_count > max_count:
                max_count = cluster_best_count
                best_center = cluster_best_center
                all_circles.append(('optimal', best_center, max_count))
            elif cluster_best_count == max_count:
                all_circles.append(('optimal', cluster_best_center, cluster_best_count))
            else:
                all_circles.append(('secondary', cluster_best_center, cluster_best_count))

    return best_center, max_count, all_circles

def brute_force_algorithm(points, radius, resolution=0.05):
    if len(points) == 0:
        return None, 0
    
    padding = radius * 1.5
    x_min, y_min = np.min(points, axis=0) - padding
    x_max, y_max = np.max(points, axis=0) + padding
    step = radius * resolution

    best_center = None
    max_count = 0
    
    x_grid = np.arange(x_min, x_max + step, step)
    y_grid = np.arange(y_min, y_max + step, step)
    
    for x in x_grid:
        for y in y_grid:
            center = np.array([x, y])
            distances = np.linalg.norm(points - center, axis=1)
            count = np.sum(distances <= radius)
            if count > max_count:
                max_count = count
                best_center = center

    return best_center, max_count

def show_visualization(points, labels=None, sliding_circles=None, brute_result=None, radius=None, stage=0):
    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    color_palette = [
        '#0000FF',  # Blue for cluster 0
        '#FF0000', '#00FF00', '#FFFF00', '#FF00FF', '#00FFFF',
        '#FF4500', '#9400D3', '#4B0082', '#008000', '#800000', '#000080',
        '#8B4513', '#FF8C00', '#7CFC00', '#DC143C', '#00BFFF', '#FF1493'
    ]
    
    if stage == 0:  # Raw points in grey
        ax.scatter(points[:, 0], points[:, 1], 
                  color='#777777', 
                  alpha=0.7, 
                  s=80,
                  label=f'Raw Points ({len(points)} pts)')
        ax.set_title("Initial Points Visualization", fontsize=14)
        ax.legend(fontsize=10)
    
    elif stage == 1:  # Colored clusters
        clusters = sorted(set(labels) - {-1})  # Sort to ensure cluster 0 comes first
        cluster_counts = {}
        
        for cluster_id in clusters:
            cluster_counts[cluster_id] = np.sum(labels == cluster_id)
        
        for i, cluster_id in enumerate(clusters):
            cluster_points = points[labels == cluster_id]
            # Special color for cluster 0
            color = '#0000FF' if cluster_id == 0 else color_palette[i % len(color_palette)]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                      color=color, 
                      edgecolor='black',
                      linewidth=0.5,
                      label=f'Cluster {cluster_id} ({cluster_counts[cluster_id]} pts)', 
                      alpha=0.9,
                      s=100)

        if -1 in labels:
            noise_points = points[labels == -1]
            noise_count = len(noise_points)
            ax.scatter(noise_points[:, 0], noise_points[:, 1],
                      c='#777777',
                      marker='x', 
                      linewidth=1.5,
                      label=f'Noise ({noise_count} pts)', 
                      alpha=0.9)
        
        ax.set_title("Cluster Visualization", fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=1, edgecolor='black')
    
    elif stage == 2:  # Sliding circles
        clusters = sorted(set(labels) - {-1})
        cluster_counts = {}
        
        for cluster_id in clusters:
            cluster_counts[cluster_id] = np.sum(labels == cluster_id)
        
        for i, cluster_id in enumerate(clusters):
            cluster_points = points[labels == cluster_id]
            # Special color for cluster 0
            color = '#0000FF' if cluster_id == 0 else color_palette[i % len(color_palette)]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                      color=color, 
                      edgecolor='black',
                      linewidth=0.5,
                      label=f'Cluster {cluster_id} ({cluster_counts[cluster_id]} pts)', 
                      alpha=0.9,
                      s=100)

        if -1 in labels:
            noise_points = points[labels == -1]
            noise_count = len(noise_points)
            ax.scatter(noise_points[:, 0], noise_points[:, 1],
                      c='#777777',
                      marker='x', 
                      linewidth=1.5,
                      label=f'Noise ({noise_count} pts)', 
                      alpha=0.9)
        
        if sliding_circles:
            max_count = max([circle[2] for circle in sliding_circles])
            for circle_type, center, count in sliding_circles:
                if center is not None:
                    if circle_type == 'optimal':
                        ax.add_patch(Circle(center, radius, 
                                         fill=False, 
                                         color='red',
                                         linewidth=3, 
                                         linestyle='-'))
                        ax.text(center[0], center[1], f'{count}', 
                               color='red', 
                               ha='center',
                               va='center',
                               fontweight='bold')
                    else:
                        ax.add_patch(Circle(center, radius, 
                                         fill=False, 
                                         color='#333333',
                                         linewidth=1, 
                                         linestyle='-'))
                        ax.text(center[0], center[1], f'{count}', 
                               color='#333333', 
                               ha='center',
                               va='center')
        
        ax.set_title("Sliding Circle Algorithm Results", fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=1, edgecolor='black')
    
    elif stage == 3:  # Final comparison
        clusters = sorted(set(labels) - {-1})
        cluster_counts = {}
        
        for cluster_id in clusters:
            cluster_counts[cluster_id] = np.sum(labels == cluster_id)
        
        for i, cluster_id in enumerate(clusters):
            cluster_points = points[labels == cluster_id]
            # Special color for cluster 0
            color = '#0000FF' if cluster_id == 0 else color_palette[i % len(color_palette)]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                      color=color, 
                      edgecolor='black',
                      linewidth=0.5,
                      label=f'Cluster {cluster_id} ({cluster_counts[cluster_id]} pts)', 
                      alpha=0.9,
                      s=100)

        if -1 in labels:
            noise_points = points[labels == -1]
            noise_count = len(noise_points)
            ax.scatter(noise_points[:, 0], noise_points[:, 1],
                      c='#777777',
                      marker='x', 
                      linewidth=1.5,
                      label=f'Noise ({noise_count} pts)', 
                      alpha=0.9)
        
        # Draw sliding circle results
        if sliding_circles:
            max_sliding = max([circle[2] for circle in sliding_circles])
            for circle_type, center, count in sliding_circles:
                if center is not None and circle_type == 'optimal' and count == max_sliding:
                    ax.add_patch(Circle(center, radius, 
                                     fill=False, 
                                     color='red',
                                     linewidth=3, 
                                     linestyle='-'))
                    ax.text(center[0], center[1], f'Sliding: {count}',
                           color='red', 
                           ha='center',
                           va='center',
                           fontsize=12,
                           fontweight='bold')
        
        # Draw brute force result
        if brute_result:
            brute_center, brute_count = brute_result
            if brute_center is not None:
                ax.add_patch(Circle(brute_center, radius, 
                                 fill=False, 
                                 color='green',
                                 linewidth=3, 
                                 linestyle='--'))
                ax.text(brute_center[0], brute_center[1], f'Brute: {brute_count}',
                       color='green', 
                       ha='center',
                       va='center',
                       fontsize=12,
                       fontweight='bold')
        
        ax.set_title("Algorithm Comparison (Close window to exit)", fontsize=14, fontweight='bold')
        
        # Create custom legend
        legend_elements = []
        if sliding_circles:
            max_sliding = max([circle[2] for circle in sliding_circles])
            legend_elements.append(Line2D([0], [0], color='red', lw=3, linestyle='-', 
                                       label=f'Optimal Sliding: {max_sliding} pts'))
        if brute_result:
            legend_elements.append(Line2D([0], [0], color='green', lw=3, linestyle='--', 
                                       label=f'Brute Force: {brute_result[1]} pts'))
        
        handles, labels_leg = ax.get_legend_handles_labels()
        handles.extend(legend_elements)
        labels_leg.extend([f'Optimal Sliding: {max_sliding} pts' if sliding_circles else '',
                         f'Brute Force: {brute_result[1]} pts' if brute_result else ''])
        
        ax.legend(handles=handles, labels=labels_leg,
                bbox_to_anchor=(1.05, 1), loc='upper left',
                framealpha=1, edgecolor='black')
    
    plt.xlabel('X coordinate', fontsize=12)
    plt.ylabel('Y coordinate', fontsize=12)
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    if stage < 3:
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    else:
        plt.show()

def main():
    points, eps, min_samples, radius = get_user_input()
    
    # Show raw points for 3 seconds
    print("\nShowing raw points visualization for 3 seconds...")
    show_visualization(points, stage=0)
    
    print("\nRunning DBSCAN clustering...")
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    clusters = set(labels) - {-1}
    cluster_points = {cid: points[labels == cid] for cid in clusters}
    if not clusters:
        cluster_points = {-1: points}

    # Show clustered points for 3 seconds
    print("Showing clustered points visualization for 3 seconds...")
    show_visualization(points, labels=labels, stage=1)

    print("\n🔴 Running Sliding Circle Algorithm...")
    start_time = time.time()
    sliding_center, sliding_count, sliding_circles = sliding_circle_algorithm(points, radius, cluster_points)
    sliding_time = time.time() - start_time
    print(f"  Optimal Sliding Circle: {sliding_count} points at ({sliding_center[0]:.6f}, {sliding_center[1]:.6f})")
    print(f"  Found in {sliding_time:.4f} seconds")

    # Show sliding circle results for 3 seconds
    print("Showing sliding circle results for 3 seconds...")
    show_visualization(points, labels=labels, sliding_circles=sliding_circles, radius=radius, stage=2)

    print("\n🟢 Running Brute Force Algorithm on all points...")
    start_time = time.time()
    brute_center, brute_count = brute_force_algorithm(points, radius)
    brute_time = time.time() - start_time
    print(f"  Brute Force Optimal: {brute_count} points at ({brute_center[0]:.6f}, {brute_center[1]:.6f})")
    print(f"  Found in {brute_time:.4f} seconds")

    print("\n⏱ Timing Summary:")
    print(f"  🔴 Sliding Circle: {sliding_time:.4f} seconds")
    print(f"  🟢 Brute Force   : {brute_time:.4f} seconds")
    percentage_faster = ((brute_time - sliding_time) / brute_time) * 100
    print(f"  🚀 Sliding Circle is {percentage_faster:.2f}% faster than Brute Force")
    print(f"  🔍 Accuracy      : {sliding_count/brute_count*100:.2f}% of brute force")

    # Show final comparison (stays until window is closed)
    print("\nShowing final comparison (close window to exit)...")
    show_visualization(points, labels=labels, 
                      sliding_circles=sliding_circles,
                      brute_result=(brute_center, brute_count), 
                      radius=radius, stage=3)

if __name__ == "__main__":
    main()
