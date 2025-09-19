import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import scipy.stats as stats
from local_realtime_scheduling.result_collector import ResultCollector


class ResultAnalyzer:
    """Class for analyzing and visualizing experiment results"""
    
    def __init__(self, result_collector: ResultCollector = None, csv_path: str = None):
        """
        Initialize analyzer with either a ResultCollector or CSV path
        
        Args:
            result_collector: ResultCollector instance with loaded results
            csv_path: Path to CSV file with episode results
        """
        if result_collector is not None:
            self.result_collector = result_collector
            self.episode_df = result_collector.get_results_dataframe()
            self.summary_df = result_collector.get_summary_dataframe()
        elif csv_path is not None:
            self.episode_df = pd.read_csv(csv_path)
            self.result_collector = None
            self.summary_df = None
        else:
            raise ValueError("Either result_collector or csv_path must be provided")
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def get_basic_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the experiment"""
        stats = {}
        
        if len(self.episode_df) > 0:
            stats['total_episodes'] = len(self.episode_df)
            stats['total_instances'] = self.episode_df['instance_filename'].nunique()
            stats['policies_tested'] = self.episode_df['policy_name'].unique().tolist()
            stats['avg_episodes_per_policy_per_instance'] = (
                len(self.episode_df) / 
                (self.episode_df['instance_filename'].nunique() * 
                 self.episode_df['policy_name'].nunique())
            )
            
            # Instance characteristics
            stats['job_range'] = (
                self.episode_df['n_jobs'].min(), 
                self.episode_df['n_jobs'].max()
            )
            stats['ops_range'] = (
                self.episode_df['n_ops'].min(), 
                self.episode_df['n_ops'].max()
            )
            
        return stats
    
    def compare_policies_overall(self) -> pd.DataFrame:
        """Compare policies across all instances"""
        
        # Detect if we have global instances
        has_global_instances = self.episode_df['is_global_instance'].any() if 'is_global_instance' in self.episode_df.columns else False
        
        if has_global_instances:
            # For global instances, use actual makespans
            makespan_column = 'makespan'
        else:
            # For local instances, use delta makespans
            makespan_column = 'delta_makespan'
        
        comparison = self.episode_df.groupby('policy_name').agg({
            makespan_column: ['mean', 'std', 'min', 'max', 'median'],
            'execution_time': ['mean', 'std'],
            'execution_time_per_step': ['mean', 'std', 'min', 'max', 'median'],
            'total_reward': ['mean', 'std'],
            'is_truncated': 'mean'
        }).round(6)
        
        # Flatten column names
        comparison.columns = [f"{col[1]}_{col[0]}" for col in comparison.columns]
        
        return comparison
    
    def compare_policies_by_instance_size(self) -> pd.DataFrame:
        """Compare policies grouped by instance size (number of jobs)"""
        
        # Detect if we have global instances
        has_global_instances = self.episode_df['is_global_instance'].any() if 'is_global_instance' in self.episode_df.columns else False
        
        if has_global_instances:
            # For global instances, use actual makespans
            makespan_column = 'makespan'
        else:
            # For local instances, use delta makespans
            makespan_column = 'delta_makespan'
        
        comparison = self.episode_df.groupby(['n_jobs', 'policy_name']).agg({
            makespan_column: ['mean', 'std'],
            'execution_time': 'mean',
        }).round(3)
        
        return comparison
    
    def statistical_significance_test(self, policy1: str, policy2: str, 
                                    metric: str = None) -> Dict[str, Any]:
        """Perform statistical significance test between two policies"""
        
        # Auto-detect appropriate metric if not specified
        if metric is None:
            has_global_instances = self.episode_df['is_global_instance'].any() if 'is_global_instance' in self.episode_df.columns else False
            metric = 'makespan' if has_global_instances else 'delta_makespan'
        
        data1 = self.episode_df[self.episode_df['policy_name'] == policy1][metric]
        data2 = self.episode_df[self.episode_df['policy_name'] == policy2][metric]
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(data1, data2)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(data1) - 1) * data1.var() + 
                             (len(data2) - 1) * data2.var()) / 
                            (len(data1) + len(data2) - 2))
        cohens_d = (data1.mean() - data2.mean()) / pooled_std
        
        # Perform Mann-Whitney U test (non-parametric)
        u_stat, u_p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        
        return {
            'policy1': policy1,
            'policy2': policy2,
            'metric': metric,
            'policy1_mean': data1.mean(),
            'policy2_mean': data2.mean(),
            'improvement': (data1.mean() - data2.mean()) / data1.mean() * 100,
            't_statistic': t_stat,
            't_test_p_value': p_value,
            'cohens_d': cohens_d,
            'mann_whitney_u': u_stat,
            'mann_whitney_p': u_p_value,
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01,
        }
    
    def plot_policy_comparison_boxplot(self, metric: str = None, 
                                     save_path: str = None, figsize: Tuple[int, int] = (14, 8)):
        """Create enhanced distribution analysis with scatter points showing individual instances"""
        
        # Auto-detect appropriate metric if not specified
        if metric is None:
            has_global_instances = self.episode_df['is_global_instance'].any() if 'is_global_instance' in self.episode_df.columns else False
            metric = 'makespan' if has_global_instances else 'delta_makespan'
        
        # Function to transform policy names for better readability
        def transform_policy_name(policy_name):
            name_mapping = {
                'heuristic_MONR_PERIODIC_NEAREST_NEVER': 'MONR-PERIODIC-NEAREST-NEVER',
                'heuristic_SPRO_PERIODIC_NEAREST_THRESHOLD': 'SPRO-PERIODIC-NEAREST-THRESHOLD',
                'heuristic_SPT_PERIODIC_EET_NEVER': 'SPT-PERIODIC-EET-NEVER',
                'heuristic_SPT_PERIODIC_MONR_THRESHOLD': 'SPT-PERIODIC-MONR-THRESHOLD',
                'heuristic_SPRO_PERIODIC_SPRO_NEVER': 'SPRO-PERIODIC-SPRO-NEVER',
                'initial': 'INITIAL-SCHEDULE',
                'madrl': 'EMADRL-LSP'
            }
            return name_mapping.get(policy_name, policy_name)
        
        # Sort original policy names first (same as plot_performance_by_instance_size)
        policies_original_sorted = sorted(self.episode_df['policy_name'].unique())
        
        # Transform policy names and create consistent mapping
        plot_df = self.episode_df.copy()
        
        # Define consistent colors and markers (same as plot_performance_by_instance_size)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', '+', 'x']
        
        # Create consistent mapping based on ORIGINAL sorted order
        policy_colors = {transform_policy_name(policy): colors[i % len(colors)] for i, policy in enumerate(policies_original_sorted)}
        policy_markers = {transform_policy_name(policy): markers[i % len(markers)] for i, policy in enumerate(policies_original_sorted)}
        
        # Create sorted list of transformed names in the same order as original
        policies_sorted = [transform_policy_name(policy) for policy in policies_original_sorted]
        
        # Create single plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create box plot for quartiles and statistical summary
        box_plot = ax.boxplot([plot_df[plot_df['policy_name'] == policy][metric].values 
                              for policy in policies_original_sorted],
                             positions=range(len(policies_original_sorted)),
                             patch_artist=True, widths=0.5, showmeans=True)
        
        # Customize box plot with better visibility
        for i, patch in enumerate(box_plot['boxes']):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(2)
        
        # Customize other box plot elements
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            for item in box_plot[element]:
                item.set_color('black')
                item.set_linewidth(1.5)
        
        # Customize means
        for item in box_plot['means']:
            item.set_marker('D')
            item.set_markerfacecolor('white')
            item.set_markeredgecolor('black')
            item.set_markersize(6)
        
        # Add scatter points for individual data points with jittering
        np.random.seed(41)  # For reproducible jittering
        
        for i, policy_original in enumerate(policies_original_sorted):
            policy_data = plot_df[plot_df['policy_name'] == policy_original]
            
            # Get all values for this policy (across all instances)
            values = policy_data[metric].values
            
            # Add horizontal jittering to avoid overlapping points
            x_positions = i + np.random.normal(0, 0.08, len(values))
            
            # Use consistent marker for this policy (same as in plot_performance_by_instance_size)
            policy_transformed = transform_policy_name(policy_original)
            marker = policy_markers[policy_transformed]
            color = policy_colors[policy_transformed]
            
            # Plot scatter points for this policy only at position i
            ax.scatter(x_positions, values, 
                      color=color, 
                      marker=marker,
                      s=45,  # increased marker size for better visibility
                      alpha=0.8,  # increased alpha for better visibility
                      edgecolors='black',
                      linewidths=0.7,  # slightly thicker edge
                      zorder=10)  # Ensure points are on top
        
        # Customize plot appearance
        # ax.set_title(f'Policy Performance Distribution Analysis - {metric.replace("_", " ").title()}',
        #             fontsize=16, fontweight='bold', pad=20)
        # ax.set_xlabel('Policy', fontweight='bold', fontsize=12)
        # ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontweight='bold', fontsize=12)
        ax.set_xticks(range(len(policies_sorted)))
        ax.set_xticklabels(policies_sorted, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add statistical annotations
        for i, policy_original in enumerate(policies_original_sorted):
            data = plot_df[plot_df['policy_name'] == policy_original][metric]
            mean_val = data.mean()
            std_val = data.std()
            
            # Add mean value above the violin plot
            ax.text(i, mean_val, f'μ={mean_val:.2f}', 
                   ha='center', va='bottom', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black'),
                   zorder=20)  # Set higher zorder to ensure text appears on top
        
        # # Create legend for policy markers (consistent with plot_performance_by_instance_size)
        # if len(policies_sorted) > 1:
        #     legend_elements = []
        #     for policy in policies_sorted:
        #         marker = policy_markers[policy]
        #         color = policy_colors[policy]
        #         legend_elements.append(plt.Line2D([0], [0], marker=marker, color=color,
        #                                         linestyle='None', markersize=6,
        #                                         label=policy))
        #
        #     # Add legend for policies
        #     ax.legend(handles=legend_elements, loc='upper right',
        #              title='Policies', frameon=True, fancybox=True, shadow=False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_performance_by_instance_size(self, metric: str = None,
                                        save_path: str = None, figsize: Tuple[int, int] = (12, 8)):
        """Plot performance vs instance size"""
        
        # Auto-detect appropriate metric if not specified
        if metric is None:
            has_global_instances = self.episode_df['is_global_instance'].any() if 'is_global_instance' in self.episode_df.columns else False
            metric = 'makespan' if has_global_instances else 'delta_makespan'
        
        plt.figure(figsize=figsize)
        
        # Calculate mean performance by instance size and policy
        perf_by_size = self.episode_df.groupby(['n_jobs', 'policy_name'])[metric].mean().reset_index()
        
        # Function to transform policy names for better readability
        def transform_policy_name(policy_name):
            name_mapping = {
                'heuristic_MONR_PERIODIC_NEAREST_NEVER': 'MONR-PERIODIC-NEAREST-NEVER',
                'heuristic_SPRO_PERIODIC_NEAREST_THRESHOLD': 'SPRO-PERIODIC-NEAREST-THRESHOLD',
                'heuristic_SPT_PERIODIC_EET_NEVER': 'SPT-PERIODIC-EET-NEVER',
                'heuristic_SPT_PERIODIC_MONR_THRESHOLD': 'SPT-PERIODIC-MONR-THRESHOLD',
                'heuristic_SPRO_PERIODIC_SPRO_NEVER': 'SPRO-PERIODIC-SPRO-NEVER',
                'initial': 'INITIAL-SCHEDULE',
                'madrl': 'EMADRL-LSP'
            }
            return name_mapping.get(policy_name, policy_name)
        
        # Define distinct colors and markers for different strategies
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', '+', 'x']
        # linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
        
        # Create line plot with distinct colors and markers
        policies = sorted(self.episode_df['policy_name'].unique())  # Sort for consistent ordering
        for i, policy in enumerate(policies):
            policy_data = perf_by_size[perf_by_size['policy_name'] == policy]
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            # linestyle = linestyles[i % len(linestyles)]
            linestyle = '-'
            
            # Use transformed policy name for the label
            display_name = transform_policy_name(policy)
            
            plt.plot(policy_data['n_jobs'], policy_data[metric], 
                    marker=marker, label=display_name,
                    linewidth=1, markersize=5,
                    color=color, linestyle=linestyle, markerfacecolor=color,
                    markeredgecolor='black', markeredgewidth=0.5)
        
        plt.xlabel('Number of Jobs')
        # plt.ylabel(f'Mean {metric.replace("_", " ").title()}')
        # plt.title(f'Policy Performance vs Instance Size')
        
        # Set x-axis to only show the specific job counts in the dataset
        job_counts = sorted(self.episode_df['n_jobs'].unique())
        plt.xticks(job_counts)
        plt.xlim(min(job_counts) - 5, max(job_counts) + 5)  # Add small margin
        
        # Show vertical grid lines only at the job count positions
        plt.grid(True, alpha=0.3, axis='y')  # Horizontal grid lines
        for job_count in job_counts:
            plt.axvline(x=job_count, color='gray', alpha=0.3, linewidth=0.8)  # Vertical grid lines
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, fancybox=True, shadow=False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _setup_chinese_font(self):
        """设置中文字体支持"""
        import matplotlib.font_manager as fm
        from matplotlib.font_manager import FontProperties
        
        # 常见的中文字体列表（按优先级排序）
        chinese_fonts = [
            'Hiragino Sans GB', 'STHeiti', 'PingFang SC', 
            'SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 
            'Noto Sans CJK SC', 'Arial Unicode MS'
        ]
        
        # 获取系统可用字体
        available_fonts = set(f.name for f in fm.fontManager.ttflist)
        
        # 查找第一个可用的中文字体
        selected_font = None
        for font_name in chinese_fonts:
            if font_name in available_fonts:
                selected_font = font_name
                break
        
        if selected_font:
            # 设置matplotlib使用中文字体
            plt.rcParams['font.sans-serif'] = [selected_font] + chinese_fonts
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['axes.unicode_minus'] = False
            
            # 返回字体属性对象用于特定文本
            return FontProperties(family=selected_font)
        else:
            print("警告: 未找到合适的中文字体")
            return FontProperties()

    def plot_combined_comparison_bw_chinese(self, metric: str = None, 
                                           save_path: str = None, figsize: Tuple[int, int] = (18, 6),
                                           policies_filter: List[str] = None):
        """Create combined black & white comparison plots with Chinese font support"""
        
        # 设置黑白样式和中文字体
        plt.style.use('grayscale')
        chinese_font_prop = self._setup_chinese_font()
        
        # Auto-detect appropriate metric if not specified
        if metric is None:
            has_global_instances = self.episode_df['is_global_instance'].any() if 'is_global_instance' in self.episode_df.columns else False
            metric = 'makespan' if has_global_instances else 'delta_makespan'
        
        # Filter policies if specified
        if policies_filter:
            filtered_df = self.episode_df[self.episode_df['policy_name'].isin(policies_filter)].copy()
        else:
            filtered_df = self.episode_df.copy()
        
        if len(filtered_df) == 0:
            print("未找到指定策略的数据")
            return
        
        # Function to transform policy names for better readability
        def transform_policy_name(policy_name):
            name_mapping = {
                'initial': '执行初始方案（无实时适应）',
                'madrl': 'EMADRL'
            }
            return name_mapping.get(policy_name, policy_name)
        
        # Sort policies and create mappings
        policies_original_sorted = sorted(filtered_df['policy_name'].unique())
        policies_sorted = [transform_policy_name(policy) for policy in policies_original_sorted]
        
        # Define black & white styling
        colors = ['#333333', '#666666', '#999999', '#CCCCCC']  # Dark gray to light gray
        # colors = ['#333333', '#666666', '#999999', '#CCCCCC']
        hatch_patterns = [None, '///', '\\\\\\', '---', '|||', '+++']
        markers = ['o', 's', '^', 'D', 'v']
        linestyles = ['-', '--', '-.', ':']
        
        # Create subplot layout: 1 row, 3 columns
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        # ===== 第一个子图：箱线图 =====
        box_plot = ax1.boxplot([filtered_df[filtered_df['policy_name'] == policy][metric].values 
                               for policy in policies_original_sorted],
                              positions=range(len(policies_original_sorted)),
                              patch_artist=True, widths=0.5, showmeans=True)
        
        # Customize box plot with black & white styling
        for i, patch in enumerate(box_plot['boxes']):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_hatch(hatch_patterns[i % len(hatch_patterns)])
            patch.set_alpha(0.3)
            patch.set_edgecolor('black')
            patch.set_linewidth(2)
        
        # Customize other box plot elements
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            for item in box_plot[element]:
                item.set_color('black')
                item.set_linewidth(2)
        
        # Customize means
        for item in box_plot['means']:
            item.set_marker('D')
            item.set_markerfacecolor('white')
            item.set_markeredgecolor('black')
            item.set_markersize(6)
            item.set_markeredgewidth(2)
        
        # Add scatter points
        np.random.seed(42)
        for i, policy_original in enumerate(policies_original_sorted):
            policy_data = filtered_df[filtered_df['policy_name'] == policy_original]
            values = policy_data[metric].values
            x_positions = i + np.random.normal(0, 0.08, len(values))
            
            marker = markers[i % len(markers)]
            color = colors[i % len(colors)]
            
            ax1.scatter(x_positions, values, 
                       color=color, marker=marker, s=40, alpha=0.9,
                       edgecolors='black', linewidths=1, zorder=10)
        
        # Add mean annotations
        for i, policy_original in enumerate(policies_original_sorted):
            data = filtered_df[filtered_df['policy_name'] == policy_original][metric]
            mean_val = data.mean()
            ax1.text(i, mean_val, f'μ={mean_val:.2f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=13,
                    fontproperties=chinese_font_prop,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black'),
                    zorder=30)  # Higher zorder to appear above scatter points
        
        ax1.set_xticks(range(len(policies_sorted)))
        ax1.set_xticklabels(policies_sorted, fontsize=17, fontweight='bold', fontproperties=chinese_font_prop)
        ax1.set_ylabel('总完工时间 (秒)', fontsize=17, fontweight='bold', fontproperties=chinese_font_prop)
        ax1.set_title('策略性能分布对比', fontsize=17, fontweight='bold', fontproperties=chinese_font_prop)
        ax1.tick_params(axis='y', labelsize=14)  # Set y-axis tick label size
        ax1.grid(True, alpha=0.3)
        
        # ===== 第二个子图：按实例大小的性能曲线 =====
        # Calculate mean performance by instance size and policy
        perf_by_size = filtered_df.groupby(['n_jobs', 'policy_name'])[metric].mean().reset_index()
        
        for i, policy in enumerate(policies_original_sorted):
            policy_data = perf_by_size[perf_by_size['policy_name'] == policy]
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            linestyle = linestyles[i % len(linestyles)]
            
            display_name = transform_policy_name(policy)
            
            ax2.plot(policy_data['n_jobs'], policy_data[metric], 
                    marker=marker, label=display_name,
                    linewidth=2.5, markersize=7,
                    color=color, linestyle=linestyle, 
                    markerfacecolor=color if i == 0 else 'white',
                    markeredgecolor='black', markeredgewidth=1.5)
        
        ax2.set_xlabel('作业数量', fontsize=17, fontweight='bold', fontproperties=chinese_font_prop)
        ax2.set_ylabel('总完工时间 (秒)', fontsize=17, fontweight='bold', fontproperties=chinese_font_prop)
        ax2.set_title('不同实例大小下的性能对比', fontsize=17, fontweight='bold', fontproperties=chinese_font_prop)
        
        # Set x-axis to only show specific job counts
        job_counts = sorted(filtered_df['n_jobs'].unique())
        ax2.set_xticks(job_counts)
        ax2.tick_params(axis='x', labelsize=10)  # Set y-axis tick label size
        ax2.tick_params(axis='y', labelsize=14)  # Set y-axis tick label size
        ax2.set_xlim(min(job_counts) - 5, max(job_counts) + 5)
        
        # Grid styling
        ax2.grid(True, alpha=0.4, axis='y')
        for job_count in job_counts:
            ax2.axvline(x=job_count, color='gray', alpha=0.4, linewidth=0.8)
        
        ax2.legend(fontsize=17, prop=chinese_font_prop)
        
        # ===== 第三个子图：执行时间对比 =====
        # Filter execution time data
        max_execution_time = 300.0
        exec_filtered_df = filtered_df[filtered_df['execution_time'] <= max_execution_time].copy()
        
        if len(exec_filtered_df) == 0:
            ax3.text(0.5, 0.5, '无有效执行时间数据', ha='center', va='center', 
                    transform=ax3.transAxes, fontproperties=chinese_font_prop)
        else:
            exec_filtered_df['policy_display'] = exec_filtered_df['policy_name'].apply(transform_policy_name)
            
            # Create violin plot with custom styling
            policies_display = [transform_policy_name(p) for p in policies_original_sorted if p in exec_filtered_df['policy_name'].unique()]
            
            if len(policies_display) > 0:
                import seaborn as sns
                
                # Map colors for violin plot
                policy_colors = {}
                for i, policy in enumerate(policies_original_sorted):
                    policy_colors[transform_policy_name(policy)] = colors[i % len(colors)]
                
                violin_plot = sns.violinplot(
                    data=exec_filtered_df, 
                    x='policy_display', 
                    y='execution_time',
                    inner=None,
                    palette=[policy_colors.get(p, '#666666') for p in policies_display],
                    alpha=0.8,
                    ax=ax3
                )
                
                # Add hatching to violin plots
                for i, violin in enumerate(violin_plot.collections):
                    if i < len(policies_display):
                        hatch = hatch_patterns[(i + 1) % len(hatch_patterns)]  # Skip None for first
                        if hatch:
                            violin.set_hatch(hatch)
                        violin.set_edgecolor('black')
                        violin.set_linewidth(1.5)
                
                # Add mean annotations
                for i, policy in enumerate(policies_original_sorted):
                    if policy in exec_filtered_df['policy_name'].unique():
                        policy_data = exec_filtered_df[exec_filtered_df['policy_name'] == policy]['execution_time']
                        if len(policy_data) > 0:
                            mean_val = policy_data.mean()
                            std_val = policy_data.std()
                            ax3.text(i, mean_val, f'{mean_val:.2f}\n±{std_val:.2f}',
                                    ha='center', va='bottom', fontweight='bold', fontsize=13,
                                    fontproperties=chinese_font_prop,
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                            alpha=0.9, edgecolor='black'))
                
                ax3.set_ylabel('总推理时间 (秒)', fontsize=17, fontweight='bold', fontproperties=chinese_font_prop)
                ax3.set_xlabel('')  # Remove x-axis label
                ax3.set_xticklabels(policies_display, fontsize=17, fontweight='bold', fontproperties=chinese_font_prop)
                ax3.tick_params(axis='y', labelsize=14)  # Set y-axis tick label size
            
        ax3.set_title('总推理时间分布对比', fontsize=17, fontweight='bold', fontproperties=chinese_font_prop)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"合并的黑白中文对比图已保存到: {save_path}")
        
        plt.show()

    def plot_policy_comparison_boxplot_bw(self, metric: str = None, 
                                        save_path: str = None, figsize: Tuple[int, int] = (10, 8),
                                        policies_filter: List[str] = None):
        """Create black & white policy comparison boxplot with filtered policies"""
        
        # 设置黑白样式
        plt.style.use('grayscale')
        
        # Auto-detect appropriate metric if not specified
        if metric is None:
            has_global_instances = self.episode_df['is_global_instance'].any() if 'is_global_instance' in self.episode_df.columns else False
            metric = 'makespan' if has_global_instances else 'delta_makespan'
        
        # Filter policies if specified
        if policies_filter:
            filtered_df = self.episode_df[self.episode_df['policy_name'].isin(policies_filter)].copy()
        else:
            filtered_df = self.episode_df.copy()
        
        if len(filtered_df) == 0:
            print("No data found for specified policies")
            return
        
        # Function to transform policy names for better readability
        def transform_policy_name(policy_name):
            name_mapping = {
                'initial': 'Initial Schedule',
                'madrl': 'EMADRL'
            }
            return name_mapping.get(policy_name, policy_name)
        
        # Sort policies and create mappings
        policies_original_sorted = sorted(filtered_df['policy_name'].unique())
        policies_sorted = [transform_policy_name(policy) for policy in policies_original_sorted]
        
        # Define black & white styling
        colors = ['#333333', '#666666', '#999999', '#CCCCCC']  # Dark gray to light gray
        hatch_patterns = [None, '///', '\\\\\\', '---', '|||', '+++']
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create box plot
        box_plot = ax.boxplot([filtered_df[filtered_df['policy_name'] == policy][metric].values 
                              for policy in policies_original_sorted],
                             positions=range(len(policies_original_sorted)),
                             patch_artist=True, widths=0.5, showmeans=True)
        
        # Customize box plot with black & white styling
        for i, patch in enumerate(box_plot['boxes']):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_hatch(hatch_patterns[i % len(hatch_patterns)])
            patch.set_alpha(0.8)
            patch.set_edgecolor('black')
            patch.set_linewidth(2)
        
        # Customize other box plot elements
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            for item in box_plot[element]:
                item.set_color('black')
                item.set_linewidth(2)
        
        # Customize means
        for item in box_plot['means']:
            item.set_marker('D')
            item.set_markerfacecolor('white')
            item.set_markeredgecolor('black')
            item.set_markersize(8)
            item.set_markeredgewidth(2)
        
        # Add scatter points with different markers
        markers = ['o', 's', '^', 'D', 'v']
        np.random.seed(42)  # For reproducible jittering
        
        for i, policy_original in enumerate(policies_original_sorted):
            policy_data = filtered_df[filtered_df['policy_name'] == policy_original]
            values = policy_data[metric].values
            x_positions = i + np.random.normal(0, 0.08, len(values))
            
            marker = markers[i % len(markers)]
            color = colors[i % len(colors)]
            
            ax.scatter(x_positions, values, 
                      color=color,
                      marker=marker,
                      s=60,
                      alpha=0.9,
                      edgecolors='black',
                      linewidths=1.2,
                      zorder=10)
        
        # Customize plot appearance
        ax.set_xticks(range(len(policies_sorted)))
        ax.set_xticklabels(policies_sorted, fontsize=17, fontweight='bold')
        ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=17, fontweight='bold')
        ax.tick_params(axis='y', labelsize=17)  # Set y-axis tick label size
        ax.grid(True, alpha=0.3)
        
        # Add mean annotations
        for i, policy_original in enumerate(policies_original_sorted):
            data = filtered_df[filtered_df['policy_name'] == policy_original][metric]
            mean_val = data.mean()
            
            ax.text(i, mean_val, f'μ={mean_val:.2f}', 
                   ha='center', va='bottom', fontweight='bold', fontsize=17,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='black'),
                   zorder=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_performance_by_instance_size_bw(self, metric: str = None,
                                           save_path: str = None, figsize: Tuple[int, int] = (10, 8),
                                           policies_filter: List[str] = None):
        """Plot black & white performance vs instance size with filtered policies"""
        
        # 设置黑白样式
        plt.style.use('grayscale')
        
        # Auto-detect appropriate metric if not specified
        if metric is None:
            has_global_instances = self.episode_df['is_global_instance'].any() if 'is_global_instance' in self.episode_df.columns else False
            metric = 'makespan' if has_global_instances else 'delta_makespan'
        
        # Filter policies if specified
        if policies_filter:
            filtered_df = self.episode_df[self.episode_df['policy_name'].isin(policies_filter)].copy()
        else:
            filtered_df = self.episode_df.copy()
        
        if len(filtered_df) == 0:
            print("No data found for specified policies")
            return
        
        plt.figure(figsize=figsize)
        
        # Calculate mean performance by instance size and policy
        perf_by_size = filtered_df.groupby(['n_jobs', 'policy_name'])[metric].mean().reset_index()
        
        # Function to transform policy names
        def transform_policy_name(policy_name):
            name_mapping = {
                'initial': 'Initial Schedule',
                'madrl': 'EMADRL'
            }
            return name_mapping.get(policy_name, policy_name)
        
        # Define black & white styling
        colors = ['#333333', '#666666']  # Dark gray and gray
        markers = ['o', 's', '^', 'D']
        linestyles = ['-', '--', '-.', ':']
        
        # Create line plot with distinct markers and line styles
        policies = sorted(filtered_df['policy_name'].unique())
        for i, policy in enumerate(policies):
            policy_data = perf_by_size[perf_by_size['policy_name'] == policy]
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            linestyle = linestyles[i % len(linestyles)]
            
            display_name = transform_policy_name(policy)
            
            plt.plot(policy_data['n_jobs'], policy_data[metric], 
                    marker=marker, label=display_name,
                    linewidth=3, markersize=8,
                    color=color, linestyle=linestyle, 
                    markerfacecolor=color if i == 0 else 'white',
                    markeredgecolor='black', markeredgewidth=2)
        
        plt.xlabel('Number of Jobs', fontsize=17, fontweight='bold')
        plt.ylabel(f'{metric.replace("_", " ").title()}', fontsize=17, fontweight='bold')
        
        # Set x-axis to only show specific job counts
        job_counts = sorted(filtered_df['n_jobs'].unique())
        plt.xticks(job_counts, fontsize=17)
        plt.yticks(fontsize=17)  # Set y-axis tick label size
        plt.xlim(min(job_counts) - 5, max(job_counts) + 5)
        
        # Grid styling
        plt.grid(True, alpha=0.4, axis='y')
        for job_count in job_counts:
            plt.axvline(x=job_count, color='gray', alpha=0.4, linewidth=1)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True, 
                  fancybox=True, shadow=False, fontsize=17)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_execution_time_comparison_bw(self, save_path: str = None,
                                        figsize: Tuple[int, int] = (10, 8),
                                        max_execution_time: float = 300.0,
                                        policies_filter: List[str] = None):
        """Plot black & white execution time comparison with filtered policies"""
        
        # 设置黑白样式
        plt.style.use('grayscale')
        
        # Filter policies if specified
        if policies_filter:
            initial_df = self.episode_df[self.episode_df['policy_name'].isin(policies_filter)].copy()
        else:
            initial_df = self.episode_df.copy()
        
        # Filter out abnormal data points
        filtered_df = initial_df[initial_df['execution_time'] <= max_execution_time].copy()
        
        if len(filtered_df) == 0:
            print(f"Warning: No data points found with execution_time <= {max_execution_time}")
            return
        
        # Transform policy names
        def transform_policy_name(policy_name):
            name_mapping = {
                'initial': 'Initial Schedule',
                'madrl': 'EMADRL'
            }
            return name_mapping.get(policy_name, policy_name)
        
        filtered_df['policy_display'] = filtered_df['policy_name'].apply(transform_policy_name)
        
        # Define black & white styling
        policies = sorted(filtered_df['policy_name'].unique())
        colors = ['#333333', '#666666', '#999999']  # Dark gray to gray
        hatch_patterns = [None, '///', '\\\\\\']
        
        policy_colors = {}
        policy_hatches = {}
        for i, policy in enumerate(policies):
            policy_colors[transform_policy_name(policy)] = colors[i % len(colors)]
            policy_hatches[transform_policy_name(policy)] = hatch_patterns[i % len(hatch_patterns)]
        
        plt.figure(figsize=figsize)
        
        # Create violin plot with custom styling
        policies_display = [transform_policy_name(p) for p in policies]
        ax = sns.violinplot(
            data=filtered_df, 
            x='policy_display', 
            y='execution_time',
            inner=None,
            palette=[policy_colors[p] for p in policies_display],
            alpha=0.8
        )
        
        # Add hatching to violin plots
        for i, violin in enumerate(ax.collections):
            if i < len(policies_display):
                hatch = policy_hatches[policies_display[i]]
                if hatch:
                    violin.set_hatch(hatch)
                violin.set_edgecolor('black')
                violin.set_linewidth(2)
        
        # Add mean annotations
        for i, policy in enumerate(policies):
            policy_data = filtered_df[filtered_df['policy_name'] == policy]['execution_time']
            if len(policy_data) > 0:
                mean_val = policy_data.mean()
                std_val = policy_data.std()
                plt.text(i, mean_val, f'{mean_val:.2f}\n±{std_val:.2f}',
                        ha='center', va='bottom', fontweight='bold', fontsize=17,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                alpha=0.9, edgecolor='black'))
        
        plt.ylabel('Execution Time (seconds)', fontsize=17, fontweight='bold')
        ax.set_xticks(range(len(policies_display)))
        ax.set_xticklabels(policies_display, fontsize=17, fontweight='bold')
        ax.tick_params(axis='y', labelsize=17)  # Set y-axis tick label size
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        
        plt.show()

    def plot_execution_time_comparison(self,
        save_path: str = None,
        figsize: Tuple[int, int] = (12, 6),
        max_execution_time: float = 300.0
    ):
        """
        Plot execution time comparison between policies with improved visualization
        
        Args:
            save_path: Path to save the plot
            figsize: Figure size tuple
            max_execution_time: Maximum execution time threshold to filter outliers
        """
        # Filter out abnormal data points
        filtered_df = self.episode_df[self.episode_df['execution_time'] <= max_execution_time].copy()
        
        if len(filtered_df) == 0:
            print(f"Warning: No data points found with execution_time <= {max_execution_time}")
            return
        
        # Report filtering statistics
        total_points = len(self.episode_df)
        filtered_points = len(filtered_df)
        outliers_removed = total_points - filtered_points
        print(f"Filtered out {outliers_removed} outliers (>{max_execution_time}s) from {total_points} total data points")
        
        # Transform policy names for better readability
        def transform_policy_name(policy_name):
            name_mapping = {
                'heuristic_MONR_PERIODIC_NEAREST_NEVER': 'MONR-PERIODIC-NEAREST-NEVER',
                'heuristic_SPRO_PERIODIC_NEAREST_THRESHOLD': 'SPRO-PERIODIC-NEAREST-THRESHOLD',
                'heuristic_SPRO_PERIODIC_SPRO_NEVER': 'SPRO-PERIODIC-SPRO-NEVER',
                'heuristic_SPT_PERIODIC_EET_NEVER': 'SPT-PERIODIC-EET-NEVER',
                'heuristic_SPT_PERIODIC_MONR_THRESHOLD': 'SPT-PERIODIC-MONR-THRESHOLD',
                'initial': 'INITIAL-SCHEDULE',
                'madrl': 'EMADRL-LSP'
            }
            return name_mapping.get(policy_name, policy_name)
        
        filtered_df['policy_display'] = filtered_df['policy_name'].apply(transform_policy_name)
        
        # Create custom color palette with better visibility
        policies = sorted(filtered_df['policy_name'].unique())
        n_policies = len(policies)
        # Create sorted list of transformed names in the same order as original
        policies_sorted = [transform_policy_name(policy) for policy in policies]
        
        # Use a professional color palette
        # if n_policies <= 6:
        #     colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8B5A3C', '#6A994E']
        # else:
        #     # For more policies, use a larger palette
        #     colors = plt.cm.Set1(np.linspace(0, 1, n_policies))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Create mapping for consistent colors
        policy_colors = {}
        for i, policy in enumerate(policies):
            policy_colors[transform_policy_name(policy)] = colors[i % len(colors)]
        
        plt.figure(figsize=figsize)
        
        # Create violin plot with custom styling
        ax = sns.violinplot(
            data=filtered_df, 
            x='policy_display', 
            y='execution_time',
            inner=None,
            palette=[policy_colors[transform_policy_name(p)] for p in policies],
            alpha=0.8
        )
        
        # # Add box plot overlay for better statistical summary
        # box_plot = sns.boxplot(
        #     data=filtered_df,
        #     x='policy_display',
        #     y='execution_time',
        #     width=0.3,
        #     boxprops=dict(alpha=0.8, facecolor='white', edgecolor='black', linewidth=1.5),
        #     medianprops=dict(color='red', linewidth=2),
        #     whiskerprops=dict(color='black', linewidth=1.5),
        #     capprops=dict(color='black', linewidth=1.5),
        #     flierprops=dict(marker='o', markerfacecolor='red', markersize=5, markeredgecolor='darkred'),
        #     ax=ax
        # )
        
        # # Add scatter points for individual data points with jitter
        # for i, policy in enumerate(policies):
        #     policy_data = filtered_df[filtered_df['policy_name'] == policy]['execution_time']
        #     if len(policy_data) > 0:
        #         # Add jitter for better visibility
        #         x_jitter = np.random.normal(i, 0.05, len(policy_data))
        #         plt.scatter(x_jitter, policy_data,
        #                    color=policy_colors[transform_policy_name(policy)],
        #                    alpha=0.6, s=20, edgecolors='white', linewidth=1)
        
        # Add mean annotations
        for i, policy in enumerate(policies):
            policy_data = filtered_df[filtered_df['policy_name'] == policy]['execution_time']
            if len(policy_data) > 0:
                mean_val = policy_data.mean()
                std_val = policy_data.std()
                plt.text(i, mean_val, f'{mean_val:.2f}\n±{std_val:.2f}',
                        ha='center', va='bottom', fontweight='bold', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
        
        # Customize plot appearance
        # plt.title('Execution Time Distribution by Policy', fontsize=16, fontweight='bold', pad=20)
        # plt.xlabel('Policy', fontsize=14, fontweight='bold')
        plt.ylabel('Execution Time (seconds)', fontsize=14,
                   # fontweight='bold'
                   )
        ax.set_xticks(range(len(policies_sorted)))
        ax.set_xticklabels(policies_sorted, rotation=45, ha='right')
        # plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(fontsize=11)
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3, axis='y')
        
        # # Add filtered data info to the plot
        # plt.text(0.02, 0.98, f'Data filtered: ≤{max_execution_time}s\n({filtered_points}/{total_points} points)',
        #         transform=ax.transAxes, fontsize=9,
        #         bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8),
        #         verticalalignment='top')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def plot_execution_time_per_step_comparison(self, save_path: str = None, 
                                               figsize: Tuple[int, int] = (12, 8)):
        """Plot execution time per step comparison between policies"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Boxplot for execution time per step
        sns.boxplot(data=self.episode_df, x='policy_name', y='execution_time_per_step', ax=ax1)
        ax1.set_title('Execution Time per Step Comparison')
        ax1.set_xlabel('Policy')
        ax1.set_ylabel('Execution Time per Step (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add mean values as text
        policies = self.episode_df['policy_name'].unique()
        for i, policy in enumerate(policies):
            data = self.episode_df[self.episode_df['policy_name'] == policy]['execution_time_per_step']
            mean_val = data.mean()
            ax1.text(i, mean_val, f'{mean_val:.6f}s', 
                    ha='center', va='bottom', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Violin plot for more detailed distribution
        sns.violinplot(data=self.episode_df, x='policy_name', y='execution_time_per_step', ax=ax2)
        ax2.set_title('Execution Time per Step Distribution')
        ax2.set_xlabel('Policy')
        ax2.set_ylabel('Execution Time per Step (seconds)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add statistical information
        for i, policy in enumerate(policies):
            data = self.episode_df[self.episode_df['policy_name'] == policy]['execution_time_per_step']
            median_val = data.median()
            ax2.text(i, median_val, f'med: {median_val:.6f}s', 
                    ha='center', va='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_correlation_matrix(self, save_path: str = None, 
                              figsize: Tuple[int, int] = (10, 8)):
        """Plot correlation matrix of numerical features"""
        numerical_cols = ['delta_makespan', 'execution_time', 'execution_time_per_step', 'total_reward', 
                         'n_jobs', 'n_ops', 'makespan', 'estimated_makespan']
        
        # Filter columns that exist in the dataframe
        available_cols = [col for col in numerical_cols if col in self.episode_df.columns]
        
        corr_matrix = self.episode_df[available_cols].corr()
        
        plt.figure(figsize=figsize)
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f')
        
        plt.title('Correlation Matrix of Performance Metrics')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_improvement_analysis(self, baseline_policy: str = 'random',
                                save_path: str = None, figsize: Tuple[int, int] = (12, 8)):
        """Plot improvement analysis compared to baseline policy"""
        # Calculate improvement for each instance
        improvement_data = []
        
        instances = self.episode_df['instance_filename'].unique()
        policies = self.episode_df['policy_name'].unique()
        baseline_policies = [p for p in policies if p != baseline_policy]
        
        for instance in instances:
            instance_data = self.episode_df[self.episode_df['instance_filename'] == instance]
            
            baseline_performance = instance_data[
                instance_data['policy_name'] == baseline_policy
            ]['delta_makespan'].mean()
            
            for policy in baseline_policies:
                policy_performance = instance_data[
                    instance_data['policy_name'] == policy
                ]['delta_makespan'].mean()
                
                improvement = (baseline_performance - policy_performance) / baseline_performance * 100
                
                improvement_data.append({
                    'instance': instance,
                    'policy': policy,
                    'improvement_percent': improvement,
                    'n_jobs': instance_data['n_jobs'].iloc[0],
                    'n_ops': instance_data['n_ops'].iloc[0]
                })
        
        improvement_df = pd.DataFrame(improvement_data)
        
        # Create subplot for each non-baseline policy
        n_policies = len(baseline_policies)
        fig, axes = plt.subplots(1, n_policies, figsize=figsize, sharey=True)
        
        if n_policies == 1:
            axes = [axes]
        
        for i, policy in enumerate(baseline_policies):
            policy_data = improvement_df[improvement_df['policy'] == policy]
            
            # Scatter plot with job size as color
            scatter = axes[i].scatter(policy_data['n_ops'], policy_data['improvement_percent'],
                                    c=policy_data['n_jobs'], cmap='viridis', alpha=0.7, s=50)
            
            axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[i].set_xlabel('Number of Operations')
            axes[i].set_title(f'{policy.title()} vs {baseline_policy.title()}')
            axes[i].grid(True, alpha=0.3)
            
            # Add colorbar for job size
            plt.colorbar(scatter, ax=axes[i], label='Number of Jobs')
        
        axes[0].set_ylabel('Improvement (%)')
        
        plt.suptitle(f'Performance Improvement Analysis (Baseline: {baseline_policy.title()})')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return improvement_df
    
    def generate_comprehensive_report(self, output_dir: str = None):
        """Generate a comprehensive analysis report with all visualizations"""
        if output_dir is None:
            output_dir = "analysis_report"
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"Generating comprehensive analysis report in {output_path}")
        
        # Basic statistics
        basic_stats = self.get_basic_statistics()
        print("Basic Statistics:")
        for key, value in basic_stats.items():
            print(f"  {key}: {value}")
        
        # Policy comparison
        policy_comparison = self.compare_policies_overall()
        print("\nPolicy Comparison (Overall):")
        print(policy_comparison)
        
        # Save comparison to CSV
        policy_comparison.to_csv(output_path / "policy_comparison_overall.csv")
        
        # Statistical significance tests
        policies = self.episode_df['policy_name'].unique()
        significance_results = []
        significance_results_per_step = []
        
        for i in range(len(policies)):
            for j in range(i+1, len(policies)):
                result = self.statistical_significance_test(policies[i], policies[j])
                significance_results.append(result)
                
                # Test execution time per step as well
                result_per_step = self.statistical_significance_test(policies[i], policies[j], 'execution_time_per_step')
                significance_results_per_step.append(result_per_step)
        
        significance_df = pd.DataFrame(significance_results)
        significance_df.to_csv(output_path / "statistical_significance_tests.csv", index=False)
        
        significance_per_step_df = pd.DataFrame(significance_results_per_step)
        significance_per_step_df.to_csv(output_path / "statistical_significance_tests_per_step.csv", index=False)
        
        # print("\nStatistical Significance Tests (Delta Makespan):")
        # for result in significance_results:
        #     print(f"{result['policy1']} vs {result['policy2']}: "
        #           f"p-value = {result['t_test_p_value']:.4f}, "
        #           f"improvement = {result['improvement']:.2f}%")
        #
        # print("\nStatistical Significance Tests (Execution Time per Step):")
        # for result in significance_results_per_step:
        #     print(f"{result['policy1']} vs {result['policy2']}: "
        #           f"p-value = {result['t_test_p_value']:.4f}, "
        #           f"improvement = {result['improvement']:.2f}%")
        
        # Generate plots
        print("\nGenerating visualizations...")
        
        # 1. Policy comparison boxplot
        self.plot_policy_comparison_boxplot(
            save_path=output_path / "policy_comparison_boxplot.png"
        )
        
        # 2. Performance by instance size
        self.plot_performance_by_instance_size(
            save_path=output_path / "performance_by_instance_size.png"
        )
        
        # 3. Execution time comparison
        self.plot_execution_time_comparison(
            save_path=output_path / "execution_time_comparison.png"
        )
        
        # # 4. Execution time per step comparison
        # self.plot_execution_time_per_step_comparison(
        #     save_path=output_path / "execution_time_per_step_comparison.png"
        # )
        #
        # # 5. Correlation matrix
        # self.plot_correlation_matrix(
        #     save_path=output_path / "correlation_matrix.png"
        # )
        #
        # # 6. Improvement analysis
        # improvement_df = self.plot_improvement_analysis(
        #     save_path=output_path / "improvement_analysis.png"
        # )
        # improvement_df.to_csv(output_path / "improvement_analysis.csv", index=False)
        
        # Save detailed episode data
        self.episode_df.to_csv(output_path / "detailed_episode_results.csv", index=False)
        
        # Generate summary report
        self._generate_text_report(output_path, basic_stats, policy_comparison, significance_results, significance_results_per_step)
        
        print(f"\nComprehensive report generated successfully in {output_path}")
    
    def _generate_text_report(self, output_path: Path, basic_stats: Dict, 
                            policy_comparison: pd.DataFrame, significance_results: List[Dict],
                            significance_results_per_step: List[Dict]):
        """Generate a text summary report"""
        report_path = output_path / "analysis_report.txt"
        
        # Determine appropriate makespan column based on what's available
        if 'mean_makespan' in policy_comparison.columns:
            makespan_column = 'mean_makespan'
            analysis_type = "GLOBAL INSTANCE"
        elif 'mean_delta_makespan' in policy_comparison.columns:
            makespan_column = 'mean_delta_makespan'
            analysis_type = "LOCAL INSTANCE"
        else:
            # Fallback - try to find any makespan column
            makespan_columns = [col for col in policy_comparison.columns if 'makespan' in col and 'mean' in col]
            makespan_column = makespan_columns[0] if makespan_columns else 'mean_makespan'
            analysis_type = "SCHEDULING POLICY"
        
        with open(report_path, 'w') as f:
            f.write(f"{analysis_type} COMPARISON ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic Statistics
            f.write("1. BASIC STATISTICS\n")
            f.write("-" * 20 + "\n")
            for key, value in basic_stats.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Policy Performance
            f.write("2. POLICY PERFORMANCE SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(policy_comparison.to_string())
            f.write("\n\n")
            
            # Statistical Significance - Makespan
            metric_type = "MAKESPAN" if 'global' in analysis_type.lower() else "DELTA MAKESPAN"
            f.write(f"3. STATISTICAL SIGNIFICANCE TESTS - {metric_type}\n")
            f.write("-" * (30 + len(metric_type)) + "\n")
            for result in significance_results:
                f.write(f"\n{result['policy1'].upper()} vs {result['policy2'].upper()}:\n")
                f.write(f"  Mean Performance: {result['policy1_mean']:.3f} vs {result['policy2_mean']:.3f}\n")
                f.write(f"  Improvement: {result['improvement']:.2f}%\n")
                f.write(f"  T-test p-value: {result['t_test_p_value']:.6f}\n")
                f.write(f"  Mann-Whitney p-value: {result['mann_whitney_p']:.6f}\n")
                f.write(f"  Effect size (Cohen's d): {result['cohens_d']:.3f}\n")
                f.write(f"  Significant at α=0.05: {result['significant_at_0.05']}\n")
                f.write(f"  Significant at α=0.01: {result['significant_at_0.01']}\n")
            
            f.write("\n\n")
            
            # Statistical Significance - Execution Time per Step
            f.write("4. STATISTICAL SIGNIFICANCE TESTS - EXECUTION TIME PER STEP\n")
            f.write("-" * 55 + "\n")
            for result in significance_results_per_step:
                f.write(f"\n{result['policy1'].upper()} vs {result['policy2'].upper()}:\n")
                f.write(f"  Mean Performance: {result['policy1_mean']:.6f}s vs {result['policy2_mean']:.6f}s\n")
                f.write(f"  Improvement: {result['improvement']:.2f}%\n")
                f.write(f"  T-test p-value: {result['t_test_p_value']:.6f}\n")
                f.write(f"  Mann-Whitney p-value: {result['mann_whitney_p']:.6f}\n")
                f.write(f"  Effect size (Cohen's d): {result['cohens_d']:.3f}\n")
                f.write(f"  Significant at α=0.05: {result['significant_at_0.05']}\n")
                f.write(f"  Significant at α=0.01: {result['significant_at_0.01']}\n")
            
            f.write("\n\n")
            
            # Conclusions
            f.write("5. KEY FINDINGS\n")
            f.write("-" * 15 + "\n")
            
            # Find best performing policy for makespan
            best_policy_makespan = policy_comparison.sort_values(makespan_column).index[0]
            f.write(f"• Best performing policy (makespan): {best_policy_makespan.upper()}\n")
            
            # Find best performing policy for execution time per step
            best_policy_time = policy_comparison.sort_values('mean_execution_time_per_step').index[0]
            f.write(f"• Fastest policy (time per step): {best_policy_time.upper()}\n")
            
            # Find significant improvements for makespan
            significant_improvements_makespan = [r for r in significance_results if r['significant_at_0.05'] and r['improvement'] > 0]
            if significant_improvements_makespan:
                f.write(f"• Significant makespan improvements found: {len(significant_improvements_makespan)}\n")
                for imp in significant_improvements_makespan:
                    f.write(f"  - {imp['policy2']} improves over {imp['policy1']} by {imp['improvement']:.2f}%\n")
            else:
                f.write("• No statistically significant makespan improvements found\n")
                
            # Find significant improvements for execution time per step
            significant_improvements_time = [r for r in significance_results_per_step if r['significant_at_0.05'] and r['improvement'] > 0]
            if significant_improvements_time:
                f.write(f"• Significant execution time per step improvements found: {len(significant_improvements_time)}\n")
                for imp in significant_improvements_time:
                    f.write(f"  - {imp['policy2']} improves over {imp['policy1']} by {imp['improvement']:.2f}%\n")
            else:
                f.write("• No statistically significant execution time per step improvements found\n")
    
    def plot_instance_difficulty_analysis(self, save_path: str = None, 
                                        figsize: Tuple[int, int] = (14, 10)):
        """Analyze and plot instance difficulty patterns"""
        # Calculate difficulty metrics for each instance
        instance_analysis = []
        
        for instance in self.episode_df['instance_filename'].unique():
            instance_data = self.episode_df[self.episode_df['instance_filename'] == instance]
            
            analysis_row = {
                'instance': instance,
                'n_jobs': instance_data['n_jobs'].iloc[0],
                'n_ops': instance_data['n_ops'].iloc[0],
                'window_id': instance_data['window_id'].iloc[0],
            }
            
            # Calculate metrics for each policy
            for policy in instance_data['policy_name'].unique():
                policy_data = instance_data[instance_data['policy_name'] == policy]
                analysis_row[f'{policy}_mean_makespan'] = policy_data['delta_makespan'].mean()
                analysis_row[f'{policy}_std_makespan'] = policy_data['delta_makespan'].std()
                analysis_row[f'{policy}_truncated_rate'] = policy_data['is_truncated'].mean()
            
            instance_analysis.append(analysis_row)
        
        analysis_df = pd.DataFrame(instance_analysis)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Difficulty vs number of jobs
        axes[0, 0].scatter(analysis_df['n_jobs'], analysis_df.get('random_mean_makespan', []), 
                          alpha=0.6, label='Random')
        if 'heuristic_mean_makespan' in analysis_df.columns:
            axes[0, 0].scatter(analysis_df['n_jobs'], analysis_df['heuristic_mean_makespan'], 
                              alpha=0.6, label='Heuristic')
        if 'madrl_mean_makespan' in analysis_df.columns:
            axes[0, 0].scatter(analysis_df['n_jobs'], analysis_df['madrl_mean_makespan'], 
                              alpha=0.6, label='MADRL')
        
        axes[0, 0].set_xlabel('Number of Jobs')
        axes[0, 0].set_ylabel('Mean Makespan')
        axes[0, 0].set_title('Difficulty vs Number of Jobs')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Difficulty vs number of operations
        axes[0, 1].scatter(analysis_df['n_ops'], analysis_df.get('random_mean_makespan', []), 
                          alpha=0.6, label='Random')
        if 'heuristic_mean_makespan' in analysis_df.columns:
            axes[0, 1].scatter(analysis_df['n_ops'], analysis_df['heuristic_mean_makespan'], 
                              alpha=0.6, label='Heuristic')
        if 'madrl_mean_makespan' in analysis_df.columns:
            axes[0, 1].scatter(analysis_df['n_ops'], analysis_df['madrl_mean_makespan'], 
                              alpha=0.6, label='MADRL')
        
        axes[0, 1].set_xlabel('Number of Operations')
        axes[0, 1].set_ylabel('Mean Makespan')
        axes[0, 1].set_title('Difficulty vs Number of Operations')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Operations density (ops/jobs)
        analysis_df['ops_per_job'] = analysis_df['n_ops'] / analysis_df['n_jobs']
        axes[1, 0].scatter(analysis_df['ops_per_job'], analysis_df.get('random_mean_makespan', []), 
                          alpha=0.6, label='Random')
        if 'heuristic_mean_makespan' in analysis_df.columns:
            axes[1, 0].scatter(analysis_df['ops_per_job'], analysis_df['heuristic_mean_makespan'], 
                              alpha=0.6, label='Heuristic')
        if 'madrl_mean_makespan' in analysis_df.columns:
            axes[1, 0].scatter(analysis_df['ops_per_job'], analysis_df['madrl_mean_makespan'], 
                              alpha=0.6, label='MADRL')
        
        axes[1, 0].set_xlabel('Operations per Job')
        axes[1, 0].set_ylabel('Mean Makespan')
        axes[1, 0].set_title('Difficulty vs Operation Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Truncation rate analysis
        if 'random_truncated_rate' in analysis_df.columns:
            axes[1, 1].scatter(analysis_df['n_ops'], analysis_df['random_truncated_rate'], 
                              alpha=0.6, label='Random', s=60)
        if 'heuristic_truncated_rate' in analysis_df.columns:
            axes[1, 1].scatter(analysis_df['n_ops'], analysis_df['heuristic_truncated_rate'], 
                              alpha=0.6, label='Heuristic', s=60)
        if 'madrl_truncated_rate' in analysis_df.columns:
            axes[1, 1].scatter(analysis_df['n_ops'], analysis_df['madrl_truncated_rate'], 
                              alpha=0.6, label='MADRL', s=60)
        
        axes[1, 1].set_xlabel('Number of Operations')
        axes[1, 1].set_ylabel('Truncation Rate')
        axes[1, 1].set_title('Truncation Rate vs Instance Size')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return analysis_df


# Utility functions for quick analysis
def quick_analysis_from_csv(csv_path: str, output_dir: str = None) -> ResultAnalyzer:
    """Quick analysis from CSV file"""
    analyzer = ResultAnalyzer(csv_path=csv_path)
    
    if output_dir:
        analyzer.generate_comprehensive_report(output_dir)
    
    return analyzer


def compare_experiment_runs(csv_paths: List[str], labels: List[str] = None) -> None:
    """Compare results from multiple experiment runs"""
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(csv_paths))]
    
    plt.figure(figsize=(12, 8))
    
    for i, (csv_path, label) in enumerate(zip(csv_paths, labels)):
        df = pd.read_csv(csv_path)
        
        # Plot performance distribution for each run
        policy_means = df.groupby('policy_name')['delta_makespan'].mean()
        
        plt.subplot(2, 2, i+1)
        policy_means.plot(kind='bar', title=f'{label} - Policy Performance')
        plt.ylabel('Mean Delta Makespan')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show() 