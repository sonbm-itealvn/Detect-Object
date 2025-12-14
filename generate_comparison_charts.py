"""
Script to generate comparison charts for Hungarian Algorithm vs Greedy Algorithm
Run: python generate_comparison_charts.py
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Configure style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

print("Generating comparison charts...")

# ============================================
# 1. Computational Complexity Chart
# ============================================
def plot_complexity():
    n_values = np.array([10, 20, 50, 100, 200, 500, 1000])
    hungarian = n_values ** 3
    greedy = n_values ** 2 * np.log2(n_values)
    
    plt.figure(figsize=(12, 7))
    plt.plot(n_values, hungarian, 'r-o', linewidth=2.5, markersize=8, 
             label='Hungarian O(n³)', markerfacecolor='white', markeredgewidth=2)
    plt.plot(n_values, greedy, 'b-s', linewidth=2.5, markersize=8, 
             label='Greedy O(n² log n)', markerfacecolor='white', markeredgewidth=2)
    plt.xlabel('Number of triplets (n)', fontsize=13, fontweight='bold')
    plt.ylabel('Number of operations (log scale)', fontsize=13, fontweight='bold')
    plt.title('Computational Complexity Comparison: Hungarian vs Greedy', 
              fontsize=15, fontweight='bold', pad=15)
    plt.yscale('log')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12, loc='upper left')
    
    # Add annotations
    plt.annotate('Gap increases rapidly\nas n grows', 
                xy=(500, 500**3), xytext=(300, 500**3 * 0.3),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=11, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('complexity_comparison.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created: complexity_comparison.png")

# ============================================
# 2. Speedup Ratio Chart
# ============================================
def plot_speedup():
    n_values = [10, 50, 100, 200, 500, 1000]
    speedup = [3.0, 15.1, 15.0, 26.2, 55.6, 100.4]
    
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']
    
    plt.figure(figsize=(12, 7))
    bars = plt.bar(range(len(n_values)), speedup, 
                   color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Add values on each bar
    for i, (bar, ratio) in enumerate(zip(bars, speedup)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{ratio:.1f}x', ha='center', va='bottom', 
                fontsize=11, fontweight='bold', color='black')
    
    plt.xlabel('Number of triplets (n)', fontsize=13, fontweight='bold')
    plt.ylabel('Speedup ratio (times)', fontsize=13, fontweight='bold')
    plt.title('How many times faster is Greedy than Hungarian?', 
              fontsize=15, fontweight='bold', pad=15)
    plt.xticks(range(len(n_values)), [f'n={n}' for n in n_values], fontsize=11)
    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
    plt.ylim(0, max(speedup) * 1.2)
    
    # Add trend line
    z = np.polyfit(range(len(n_values)), speedup, 2)
    p = np.poly1d(z)
    plt.plot(range(len(n_values)), p(range(len(n_values))), 
            "r--", alpha=0.5, linewidth=2, label='Trend')
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('speedup_comparison.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created: speedup_comparison.png")

# ============================================
# 3. Execution Time Chart (Simulated)
# ============================================
def plot_execution_time():
    n_values = np.array([10, 20, 50, 100, 200, 500, 1000])
    
    # Simulation: assume 1 operation = 1 nanosecond
    hungarian_time = (n_values ** 3) * 1e-9 * 1000  # milliseconds
    greedy_time = (n_values ** 2 * np.log2(n_values)) * 1e-9 * 1000  # milliseconds
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Linear scale
    ax1.plot(n_values, hungarian_time, 'r-o', linewidth=2.5, markersize=8, 
            label='Hungarian', markerfacecolor='white', markeredgewidth=2)
    ax1.plot(n_values, greedy_time, 'b-s', linewidth=2.5, markersize=8, 
            label='Greedy', markerfacecolor='white', markeredgewidth=2)
    ax1.set_xlabel('Number of triplets (n)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Time (milliseconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Execution Time (Linear Scale)', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=11)
    
    # Log scale
    ax2.plot(n_values, hungarian_time, 'r-o', linewidth=2.5, markersize=8, 
            label='Hungarian', markerfacecolor='white', markeredgewidth=2)
    ax2.plot(n_values, greedy_time, 'b-s', linewidth=2.5, markersize=8, 
            label='Greedy', markerfacecolor='white', markeredgewidth=2)
    ax2.set_xlabel('Number of triplets (n)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Time (milliseconds, log scale)', fontsize=12, fontweight='bold')
    ax2.set_title('Execution Time (Log Scale)', fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig('execution_time_comparison.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created: execution_time_comparison.png")

# ============================================
# 4. Operations Distribution Chart
# ============================================
def plot_operations_distribution():
    n = 100
    
    # Hungarian
    hungarian_init = n ** 2
    hungarian_compute = n ** 2
    hungarian_match = n ** 3
    hungarian_total = hungarian_init + hungarian_compute + hungarian_match
    
    # Greedy
    greedy_init = n ** 2
    greedy_compute = int(n ** 2 * np.log2(n))
    greedy_match = n ** 2
    greedy_total = greedy_init + greedy_compute + greedy_match
    
    # Normalize
    hungarian_pct = [x/hungarian_total*100 for x in [hungarian_init, hungarian_compute, hungarian_match]]
    greedy_pct = [x/greedy_total*100 for x in [greedy_init, greedy_compute, greedy_match]]
    
    steps = ['Initialization', 'Computation', 'Matching']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Hungarian
    ax1.barh(steps, hungarian_pct, color=['#e74c3c', '#c0392b', '#a93226'], 
            edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.set_xlabel('Percentage of operations (%)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Hungarian Algorithm (n={n})\nTotal: {hungarian_total:,} operations', 
                 fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Add values
    for i, (step, pct) in enumerate(zip(steps, hungarian_pct)):
        ax1.text(pct/2, i, f'{pct:.1f}%', ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white')
    
    # Greedy
    ax2.barh(steps, greedy_pct, color=['#3498db', '#2980b9', '#1f618d'], 
            edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Percentage of operations (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Greedy Algorithm (n={n})\nTotal: {greedy_total:,} operations', 
                 fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Add values
    for i, (step, pct) in enumerate(zip(steps, greedy_pct)):
        ax2.text(pct/2, i, f'{pct:.1f}%', ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig('operations_distribution.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created: operations_distribution.png")

# ============================================
# 5. Stacked Area Chart
# ============================================
def plot_stacked_area():
    n_values = np.array([10, 50, 100, 200, 500, 1000])
    
    # Hungarian
    hungarian_init = n_values ** 2
    hungarian_compute = n_values ** 2
    hungarian_match = n_values ** 3
    
    # Greedy
    greedy_init = n_values ** 2
    greedy_compute = n_values ** 2 * np.log2(n_values)
    greedy_match = n_values ** 2
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Hungarian
    ax1.fill_between(n_values, 0, hungarian_init, label='Initialization O(n²)', 
                     alpha=0.7, color='#e74c3c')
    ax1.fill_between(n_values, hungarian_init, hungarian_init + hungarian_compute, 
                     label='Computation O(n²)', alpha=0.7, color='#c0392b')
    ax1.fill_between(n_values, hungarian_init + hungarian_compute, 
                     hungarian_init + hungarian_compute + hungarian_match,
                     label='Matching O(n³)', alpha=0.7, color='#a93226')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of triplets (n)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Operations (log scale)', fontsize=12, fontweight='bold')
    ax1.set_title('Hungarian Algorithm - Operations Distribution', 
                 fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Greedy
    ax2.fill_between(n_values, 0, greedy_init, label='Initialization O(n²)', 
                     alpha=0.7, color='#3498db')
    ax2.fill_between(n_values, greedy_init, greedy_init + greedy_compute,
                     label='Computation O(n² log n)', alpha=0.7, color='#2980b9')
    ax2.fill_between(n_values, greedy_init + greedy_compute, 
                     greedy_init + greedy_compute + greedy_match,
                     label='Matching O(n²)', alpha=0.7, color='#1f618d')
    ax2.set_yscale('log')
    ax2.set_xlabel('Number of triplets (n)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Operations (log scale)', fontsize=12, fontweight='bold')
    ax2.set_title('Greedy Algorithm - Operations Distribution', 
                 fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('stacked_area_comparison.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created: stacked_area_comparison.png")

# ============================================
# 6. Radar Chart (Multi-dimensional Comparison)
# ============================================
def plot_radar():
    categories = ['Speed', 'Accuracy', 'Ease of Implementation', 
                 'Memory', 'Stability', 'Scalability']
    
    hungarian_scores = [3, 10, 5, 6, 10, 4]
    greedy_scores = [10, 8, 9, 9, 7, 10]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    hungarian_scores += hungarian_scores[:1]
    greedy_scores += greedy_scores[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    ax.plot(angles, hungarian_scores, 'o-', linewidth=2.5, label='Hungarian', 
           color='#e74c3c', markersize=8)
    ax.fill(angles, hungarian_scores, alpha=0.25, color='#e74c3c')
    
    ax.plot(angles, greedy_scores, 's-', linewidth=2.5, label='Greedy', 
           color='#3498db', markersize=8)
    ax.fill(angles, greedy_scores, alpha=0.25, color='#3498db')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title('Multi-dimensional Comparison: Hungarian vs Greedy', 
                size=15, fontweight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
    
    plt.tight_layout()
    plt.savefig('radar_comparison.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Created: radar_comparison.png")

# ============================================
# Run all
# ============================================
if __name__ == "__main__":
    try:
        plot_complexity()
        plot_speedup()
        plot_execution_time()
        plot_operations_distribution()
        plot_stacked_area()
        plot_radar()
        
        print("\n" + "="*50)
        print("✓ Complete! All 6 charts have been generated:")
        print("  1. complexity_comparison.png")
        print("  2. speedup_comparison.png")
        print("  3. execution_time_comparison.png")
        print("  4. operations_distribution.png")
        print("  5. stacked_area_comparison.png")
        print("  6. radar_comparison.png")
        print("="*50)
        
    except Exception as e:
        print(f"\n✗ Error generating charts: {e}")
        print("Please ensure the following packages are installed: matplotlib, numpy, seaborn")

