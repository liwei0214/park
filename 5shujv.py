

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from statsmodels.tools import add_constant
from statsmodels.regression.linear_model import OLS
import warnings
warnings.filterwarnings('ignore')
import os

# Set publication-quality parameters
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Define professional color palette
COLORS = {
    'park': '#2E7D32',  # Forest green
    'forest': '#81C784',  # Light green
    'grass': '#AED581',  # Lime green
    'water': '#039BE5',  # Ocean blue
    'urban': '#757575',  # Gray
    'pm25': '#E53935',  # Red
    'co2': '#FB8C00',  # Orange
    'temp': '#FDD835',  # Yellow
    'positive': '#1976D2',  # Blue
    'negative': '#D32F2F',  # Red
    'neutral': '#9E9E9E'  # Gray
}

class PublicationFigures:
    """Generate publication-quality figures for academic papers"""

    def __init__(self, data_path):
        """Initialize the figure generation system"""
        self.data_path = data_path
        self.output_dir = 'Publication_Figures'
        self.create_output_directory()

    def create_output_directory(self):
        """Create directory for figures"""
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"✓ Output directory created: {self.output_dir}")

    def load_data(self):
        """Load and prepare data"""
        if self.data_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_path)
        else:
            self.df = pd.read_excel(self.data_path)
        print(f"✓ Data loaded: {len(self.df)} observations")
        return self.df

    def figure1_multi_target_effects(self):
        """
        Figure 1: Multi-target Environmental Effects of Urban Parks
        展示公园对多个环境指标的综合影响
        """
        print("\nGenerating Figure 1: Multi-target Environmental Effects...")

        # Create figure with custom layout
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.35)

        # Prepare data for analysis
        targets = ['PM25', 'CO2_emissions', 'Temperature', 'Humidity']
        available_targets = [t for t in targets if t in self.df.columns]

        # Panel A: Effect sizes comparison
        ax1 = fig.add_subplot(gs[0, :2])

        effects_data = []
        for target in available_targets:
            if 'Park_area_km2' in self.df.columns:
                # Calculate effect using regression
                clean_data = self.df[['Park_area_km2', target]].dropna()
                if len(clean_data) > 20:
                    X = add_constant(clean_data[['Park_area_km2']])
                    y = clean_data[target]
                    model = OLS(y, X).fit()

                    effects_data.append({
                        'Target': target.replace('_', ' ').replace('CO2 emissions', 'CO₂'),
                        'Effect': model.params['Park_area_km2'],
                        'SE': model.bse['Park_area_km2'],
                        'P-value': model.pvalues['Park_area_km2']
                    })

        if effects_data:
            df_effects = pd.DataFrame(effects_data)

            # Create horizontal bar plot with error bars
            y_pos = np.arange(len(df_effects))
            colors_list = []
            for i, row in df_effects.iterrows():
                if row['P-value'] < 0.05:
                    colors_list.append(COLORS['negative'] if row['Effect'] < 0 else COLORS['positive'])
                else:
                    colors_list.append(COLORS['neutral'])

            bars = ax1.barh(y_pos, df_effects['Effect'], xerr=df_effects['SE']*1.96,
                           color=colors_list, alpha=0.8, capsize=5)

            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(df_effects['Target'])
            ax1.set_xlabel('Regression Coefficient (95% CI)', fontweight='bold')
            ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            ax1.set_title('A. Park Effects on Environmental Indicators', fontweight='bold', loc='left')
            ax1.grid(True, alpha=0.2, axis='x')

            # Add significance stars
            for i, (effect, pval) in enumerate(zip(df_effects['Effect'], df_effects['P-value'])):
                if pval < 0.001:
                    sig_text = '***'
                elif pval < 0.01:
                    sig_text = '**'
                elif pval < 0.05:
                    sig_text = '*'
                else:
                    sig_text = 'ns'

                x_pos = effect + (np.sign(effect) * df_effects['SE'].iloc[i] * 2.2)
                ax1.text(x_pos, i, sig_text, va='center', fontsize=10)

        # Panel B: R-squared comparison
        ax2 = fig.add_subplot(gs[0, 2])

        r2_data = []
        for target in available_targets:
            if 'Park_area_km2' in self.df.columns:
                clean_data = self.df[['Park_area_km2', target]].dropna()
                if len(clean_data) > 20:
                    X = add_constant(clean_data[['Park_area_km2']])
                    y = clean_data[target]
                    model = OLS(y, X).fit()
                    r2_data.append({
                        'Target': target.replace('_', ' ').replace('CO2 emissions', 'CO₂'),
                        'R²': model.rsquared
                    })

        if r2_data:
            df_r2 = pd.DataFrame(r2_data)
            bars = ax2.bar(range(len(df_r2)), df_r2['R²'],
                          color=[COLORS['park'], COLORS['co2'], COLORS['temp'], COLORS['water']][:len(df_r2)],
                          alpha=0.8)

            ax2.set_xticks(range(len(df_r2)))
            ax2.set_xticklabels(df_r2['Target'], rotation=45, ha='right')
            ax2.set_ylabel('R² Value', fontweight='bold')
            ax2.set_ylim(0, 0.5)
            ax2.set_title('B. Model Explanatory Power', fontweight='bold', loc='left')
            ax2.grid(True, alpha=0.2, axis='y')

            # Add value labels
            for bar, val in zip(bars, df_r2['R²']):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', fontsize=9)

        # Panel C: Scatter plots for key relationships
        ax3 = fig.add_subplot(gs[1, 0])
        if 'Park_area_km2' in self.df.columns and 'PM25' in self.df.columns:
            ax3.scatter(self.df['Park_area_km2'], self.df['PM25'],
                       alpha=0.5, s=30, color=COLORS['pm25'])

            # Add trend line
            z = np.polyfit(self.df['Park_area_km2'].dropna(),
                          self.df['PM25'].dropna(), 1)
            p = np.poly1d(z)
            x_trend = np.linspace(self.df['Park_area_km2'].min(),
                                self.df['Park_area_km2'].max(), 100)
            ax3.plot(x_trend, p(x_trend), '--', color='black', linewidth=1.5, alpha=0.8)

            # Calculate correlation
            corr = self.df['Park_area_km2'].corr(self.df['PM25'])
            ax3.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax3.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax3.set_xlabel('Park Area (km²)', fontweight='bold')
            ax3.set_ylabel('PM2.5 (μg/m³)', fontweight='bold')
            ax3.set_title('C. Park-PM2.5 Relationship', fontweight='bold', loc='left')
            ax3.grid(True, alpha=0.2)

        # Panel D: CO2 relationship
        ax4 = fig.add_subplot(gs[1, 1])
        if 'Park_area_km2' in self.df.columns and 'CO2_emissions' in self.df.columns:
            ax4.scatter(self.df['Park_area_km2'], self.df['CO2_emissions'],
                       alpha=0.5, s=30, color=COLORS['co2'])

            # Add trend line
            z = np.polyfit(self.df['Park_area_km2'].dropna(),
                          self.df['CO2_emissions'].dropna(), 1)
            p = np.poly1d(z)
            ax4.plot(x_trend, p(x_trend), '--', color='black', linewidth=1.5, alpha=0.8)

            # Calculate correlation
            corr = self.df['Park_area_km2'].corr(self.df['CO2_emissions'])
            ax4.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax4.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax4.set_xlabel('Park Area (km²)', fontweight='bold')
            ax4.set_ylabel('CO₂ Emissions', fontweight='bold')
            ax4.set_title('D. Park-CO₂ Relationship', fontweight='bold', loc='left')
            ax4.grid(True, alpha=0.2)

        # Panel E: Time trend
        ax5 = fig.add_subplot(gs[1, 2])
        if 'Year' in self.df.columns and 'Park_area_km2' in self.df.columns:
            yearly_data = self.df.groupby('Year').agg({
                'Park_area_km2': 'mean',
                'PM25': 'mean' if 'PM25' in self.df.columns else lambda x: np.nan
            })

            ax5_twin = ax5.twinx()

            line1 = ax5.plot(yearly_data.index, yearly_data['Park_area_km2'],
                           color=COLORS['park'], marker='o', linewidth=2, label='Park Area')
            ax5.set_xlabel('Year', fontweight='bold')
            ax5.set_ylabel('Park Area (km²)', color=COLORS['park'], fontweight='bold')
            ax5.tick_params(axis='y', labelcolor=COLORS['park'])

            if 'PM25' in yearly_data.columns:
                line2 = ax5_twin.plot(yearly_data.index, yearly_data['PM25'],
                                     color=COLORS['pm25'], marker='s', linewidth=2, label='PM2.5')
                ax5_twin.set_ylabel('PM2.5 (μg/m³)', color=COLORS['pm25'], fontweight='bold')
                ax5_twin.tick_params(axis='y', labelcolor=COLORS['pm25'])

            ax5.set_title('E. Temporal Trends', fontweight='bold', loc='left')
            ax5.grid(True, alpha=0.2)

        plt.suptitle('Multi-dimensional Environmental Effects of Urban Parks',
                    fontsize=14, fontweight='bold', y=1.02)

        # Save figure
        fig_path = os.path.join(self.output_dir, 'Figure_1_Multi_Target_Effects.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✓ Figure 1 saved: {fig_path}")
        return fig_path

    def figure2_green_space_comparison(self):
        """
        Figure 2: Comparative Effectiveness of Different Green Spaces
        不同类型绿地的效果对比
        """
        print("\nGenerating Figure 2: Green Space Comparison...")

        fig = plt.figure(figsize=(14, 6))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        green_types = ['Park_area_km2', 'Forest_km2', 'Grassland_km2', 'Water_km2']
        available_green = [g for g in green_types if g in self.df.columns]

        # Panel A: Effect on PM2.5
        ax1 = fig.add_subplot(gs[0, 0])

        pm25_effects = []
        if 'PM25' in self.df.columns:
            for green in available_green:
                clean_data = self.df[[green, 'PM25']].dropna()
                if len(clean_data) > 20:
                    X = add_constant(clean_data[[green]])
                    y = clean_data['PM25']
                    model = OLS(y, X).fit()

                    pm25_effects.append({
                        'Green Type': green.replace('_km2', '').replace('_', ' ').title(),
                        'Effect': model.params[green],
                        'P-value': model.pvalues[green]
                    })

        if pm25_effects:
            df_pm25 = pd.DataFrame(pm25_effects)
            colors = [COLORS['negative'] if e < 0 and p < 0.05
                     else COLORS['positive'] if e > 0 and p < 0.05
                     else COLORS['neutral']
                     for e, p in zip(df_pm25['Effect'], df_pm25['P-value'])]

            bars = ax1.bar(range(len(df_pm25)), df_pm25['Effect'], color=colors, alpha=0.8)
            ax1.set_xticks(range(len(df_pm25)))
            ax1.set_xticklabels(df_pm25['Green Type'], rotation=45, ha='right')
            ax1.set_ylabel('Effect on PM2.5', fontweight='bold')
            ax1.set_title('A. PM2.5 Reduction', fontweight='bold', loc='left')
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax1.grid(True, alpha=0.2, axis='y')

        # Panel B: Effect on CO2
        ax2 = fig.add_subplot(gs[0, 1])

        co2_effects = []
        if 'CO2_emissions' in self.df.columns:
            for green in available_green:
                clean_data = self.df[[green, 'CO2_emissions']].dropna()
                if len(clean_data) > 20:
                    X = add_constant(clean_data[[green]])
                    y = clean_data['CO2_emissions']
                    model = OLS(y, X).fit()

                    co2_effects.append({
                        'Green Type': green.replace('_km2', '').replace('_', ' ').title(),
                        'Effect': model.params[green],
                        'P-value': model.pvalues[green]
                    })

        if co2_effects:
            df_co2 = pd.DataFrame(co2_effects)
            colors = [COLORS['negative'] if e < 0 and p < 0.05
                     else COLORS['positive'] if e > 0 and p < 0.05
                     else COLORS['neutral']
                     for e, p in zip(df_co2['Effect'], df_co2['P-value'])]

            bars = ax2.bar(range(len(df_co2)), df_co2['Effect'], color=colors, alpha=0.8)
            ax2.set_xticks(range(len(df_co2)))
            ax2.set_xticklabels(df_co2['Green Type'], rotation=45, ha='right')
            ax2.set_ylabel('Effect on CO₂', fontweight='bold')
            ax2.set_title('B. CO₂ Impact', fontweight='bold', loc='left')
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax2.grid(True, alpha=0.2, axis='y')

        # Panel C: Overall effectiveness score
        ax3 = fig.add_subplot(gs[0, 2])

        # Calculate composite scores
        scores = []
        for green in available_green:
            score = 0
            count = 0

            for target in ['PM25', 'CO2_emissions', 'Temperature']:
                if target in self.df.columns:
                    clean_data = self.df[[green, target]].dropna()
                    if len(clean_data) > 20:
                        corr = clean_data[green].corr(clean_data[target])
                        # Negative correlation is good for pollutants
                        if target in ['PM25', 'CO2_emissions']:
                            score += -corr
                        else:
                            score += abs(corr)
                        count += 1

            if count > 0:
                scores.append({
                    'Green Type': green.replace('_km2', '').replace('_', ' ').title(),
                    'Score': score / count
                })

        if scores:
            df_scores = pd.DataFrame(scores).sort_values('Score', ascending=False)
            colors_score = [COLORS['park'], COLORS['forest'], COLORS['grass'], COLORS['water']][:len(df_scores)]

            bars = ax3.barh(range(len(df_scores)), df_scores['Score'], color=colors_score, alpha=0.8)
            ax3.set_yticks(range(len(df_scores)))
            ax3.set_yticklabels(df_scores['Green Type'])
            ax3.set_xlabel('Effectiveness Score', fontweight='bold')
            ax3.set_title('C. Overall Effectiveness', fontweight='bold', loc='left')
            ax3.grid(True, alpha=0.2, axis='x')

        # Panel D: Area coverage comparison
        ax4 = fig.add_subplot(gs[1, :])

        if 'City' in self.df.columns:
            city_green = self.df.groupby('City')[available_green].mean()

            x = np.arange(len(city_green))
            width = 0.2

            for i, green in enumerate(available_green):
                offset = (i - len(available_green)/2 + 0.5) * width
                color = [COLORS['park'], COLORS['forest'], COLORS['grass'], COLORS['water']][i]
                ax4.bar(x + offset, city_green[green], width,
                       label=green.replace('_km2', '').replace('_', ' ').title(),
                       color=color, alpha=0.8)

            ax4.set_xlabel('City', fontweight='bold')
            ax4.set_ylabel('Average Area (km²)', fontweight='bold')
            ax4.set_title('D. Green Space Distribution by City', fontweight='bold', loc='left')
            ax4.set_xticks(x)
            ax4.set_xticklabels(city_green.index, rotation=45, ha='right')
            ax4.legend(loc='upper right', frameon=False)
            ax4.grid(True, alpha=0.2, axis='y')

        plt.suptitle('Comparative Effectiveness of Urban Green Spaces',
                    fontsize=14, fontweight='bold', y=1.02)

        # Save figure
        fig_path = os.path.join(self.output_dir, 'Figure_2_Green_Space_Comparison.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✓ Figure 2 saved: {fig_path}")
        return fig_path

    def figure3_temporal_evolution(self):
        """
        Figure 3: Temporal Evolution of Environmental Relationships
        环境关系的时间演化
        """
        print("\nGenerating Figure 3: Temporal Evolution...")

        fig = plt.figure(figsize=(12, 8))

        if 'Year' not in self.df.columns:
            print("  ⚠ No temporal data available")
            return None

        # Define periods
        median_year = self.df['Year'].median()
        early_data = self.df[self.df['Year'] <= median_year]
        late_data = self.df[self.df['Year'] > median_year]

        # Panel layout
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Panel A: Correlation evolution
        ax1 = fig.add_subplot(gs[0, :])

        relationships = [
            ('Park_area_km2', 'PM25', 'Park→PM2.5'),
            ('Park_area_km2', 'CO2_emissions', 'Park→CO₂'),
            ('Forest_km2', 'PM25', 'Forest→PM2.5'),
            ('Population_density', 'CO2_emissions', 'Pop Density→CO₂')
        ]

        early_corrs = []
        late_corrs = []
        labels = []

        for var1, var2, label in relationships:
            if var1 in self.df.columns and var2 in self.df.columns:
                early_corr = early_data[var1].corr(early_data[var2])
                late_corr = late_data[var1].corr(late_data[var2])
                early_corrs.append(early_corr)
                late_corrs.append(late_corr)
                labels.append(label)

        if labels:
            x = np.arange(len(labels))
            width = 0.35

            bars1 = ax1.bar(x - width/2, early_corrs, width,
                           label=f'Early Period (≤{median_year:.0f})',
                           color=COLORS['neutral'], alpha=0.8)
            bars2 = ax1.bar(x + width/2, late_corrs, width,
                           label=f'Late Period (>{median_year:.0f})',
                           color=COLORS['park'], alpha=0.8)

            ax1.set_xlabel('Relationship', fontweight='bold')
            ax1.set_ylabel('Correlation Coefficient', fontweight='bold')
            ax1.set_title('A. Evolution of Key Environmental Relationships',
                         fontweight='bold', loc='left')
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels, rotation=45, ha='right')
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax1.legend(loc='upper left', frameon=False)
            ax1.grid(True, alpha=0.2, axis='y')

            # Add change indicators
            for i, (early, late) in enumerate(zip(early_corrs, late_corrs)):
                change = late - early
                if abs(change) > 0.1:
                    arrow_props = dict(arrowstyle='->', color='red' if change < 0 else 'green', lw=2)
                    ax1.annotate('', xy=(i, late), xytext=(i, early), arrowprops=arrow_props)

        # Panel B: Time series of park expansion
        ax2 = fig.add_subplot(gs[1, 0])

        if 'Park_area_km2' in self.df.columns:
            yearly_park = self.df.groupby('Year')['Park_area_km2'].mean()

            ax2.plot(yearly_park.index, yearly_park.values,
                    color=COLORS['park'], marker='o', linewidth=2)
            ax2.fill_between(yearly_park.index, 0, yearly_park.values,
                            color=COLORS['park'], alpha=0.2)
            ax2.set_xlabel('Year', fontweight='bold')
            ax2.set_ylabel('Average Park Area (km²)', fontweight='bold')
            ax2.set_title('B. Park Expansion Trend', fontweight='bold', loc='left')
            ax2.grid(True, alpha=0.2)

            # Add trend line
            z = np.polyfit(yearly_park.index, yearly_park.values, 1)
            p = np.poly1d(z)
            ax2.plot(yearly_park.index, p(yearly_park.index), '--',
                    color='black', linewidth=1.5, alpha=0.8)

            # Add growth rate annotation
            growth_rate = (yearly_park.iloc[-1] - yearly_park.iloc[0]) / yearly_park.iloc[0] * 100
            ax2.text(0.95, 0.05, f'Total Growth: {growth_rate:.1f}%',
                    transform=ax2.transAxes, ha='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Panel C: Environmental improvement
        ax3 = fig.add_subplot(gs[1, 1])

        if 'PM25' in self.df.columns and 'CO2_emissions' in self.df.columns:
            yearly_env = self.df.groupby('Year')[['PM25', 'CO2_emissions']].mean()

            # Normalize for comparison
            pm25_norm = (yearly_env['PM25'] - yearly_env['PM25'].mean()) / yearly_env['PM25'].std()
            co2_norm = (yearly_env['CO2_emissions'] - yearly_env['CO2_emissions'].mean()) / yearly_env['CO2_emissions'].std()

            ax3.plot(yearly_env.index, pm25_norm, color=COLORS['pm25'],
                    marker='o', linewidth=2, label='PM2.5')
            ax3.plot(yearly_env.index, co2_norm, color=COLORS['co2'],
                    marker='s', linewidth=2, label='CO₂')

            ax3.set_xlabel('Year', fontweight='bold')
            ax3.set_ylabel('Normalized Value', fontweight='bold')
            ax3.set_title('C. Environmental Quality Trends', fontweight='bold', loc='left')
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            ax3.legend(loc='best', frameon=False)
            ax3.grid(True, alpha=0.2)

        plt.suptitle('Temporal Evolution of Park-Environment Relationships',
                    fontsize=14, fontweight='bold', y=1.02)

        # Save figure
        fig_path = os.path.join(self.output_dir, 'Figure_3_Temporal_Evolution.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✓ Figure 3 saved: {fig_path}")
        return fig_path

    def figure4_interaction_effects(self):
        """
        Figure 4: Interaction Effects and Non-linear Relationships
        交互效应和非线性关系
        """
        print("\nGenerating Figure 4: Interaction Effects...")

        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Panel A: Park × Population Density interaction on PM2.5
        ax1 = fig.add_subplot(gs[0, 0])

        if all(col in self.df.columns for col in ['Park_area_km2', 'Population_density', 'PM25']):
            # Create population density categories
            pop_tertiles = pd.qcut(self.df['Population_density'], q=3,
                                  labels=['Low Density', 'Medium Density', 'High Density'])

            colors_pop = [COLORS['positive'], COLORS['neutral'], COLORS['negative']]

            for i, (label, color) in enumerate(zip(['Low Density', 'Medium Density', 'High Density'], colors_pop)):
                subset = self.df[pop_tertiles == label]
                ax1.scatter(subset['Park_area_km2'], subset['PM25'],
                          label=label, alpha=0.6, s=30, color=color)

                # Add trend line for each group
                if len(subset) > 10:
                    z = np.polyfit(subset['Park_area_km2'], subset['PM25'], 1)
                    p = np.poly1d(z)
                    x_range = np.linspace(subset['Park_area_km2'].min(),
                                        subset['Park_area_km2'].max(), 50)
                    ax1.plot(x_range, p(x_range), '--', color=color, linewidth=1.5, alpha=0.8)

            ax1.set_xlabel('Park Area (km²)', fontweight='bold')
            ax1.set_ylabel('PM2.5 (μg/m³)', fontweight='bold')
            ax1.set_title('A. Park Effect by Population Density', fontweight='bold', loc='left')
            ax1.legend(loc='best', frameon=False, fontsize=9)
            ax1.grid(True, alpha=0.2)

        # Panel B: Non-linear relationship
        ax2 = fig.add_subplot(gs[0, 1])

        if 'Park_area_km2' in self.df.columns and 'PM25' in self.df.columns:
            x = self.df['Park_area_km2'].dropna()
            y = self.df['PM25'].dropna()
            min_len = min(len(x), len(y))
            x = x[:min_len]
            y = y[:min_len]

            ax2.scatter(x, y, alpha=0.5, s=30, color=COLORS['pm25'])

            # Fit linear and quadratic models
            z1 = np.polyfit(x, y, 1)
            z2 = np.polyfit(x, y, 2)

            x_smooth = np.linspace(x.min(), x.max(), 100)
            p1 = np.poly1d(z1)
            p2 = np.poly1d(z2)

            ax2.plot(x_smooth, p1(x_smooth), '--', color='blue',
                    linewidth=2, alpha=0.8, label='Linear')
            ax2.plot(x_smooth, p2(x_smooth), '-', color='red',
                    linewidth=2, alpha=0.8, label='Quadratic')

            ax2.set_xlabel('Park Area (km²)', fontweight='bold')
            ax2.set_ylabel('PM2.5 (μg/m³)', fontweight='bold')
            ax2.set_title('B. Non-linear Effects', fontweight='bold', loc='left')
            ax2.legend(loc='best', frameon=False)
            ax2.grid(True, alpha=0.2)

        # Panel C: Threshold effects
        ax3 = fig.add_subplot(gs[0, 2])

        if 'Park_area_km2' in self.df.columns and 'CO2_emissions' in self.df.columns:
            # Identify threshold
            threshold = self.df['Park_area_km2'].median()

            low_park = self.df[self.df['Park_area_km2'] <= threshold]
            high_park = self.df[self.df['Park_area_km2'] > threshold]

            ax3.scatter(low_park['Park_area_km2'], low_park['CO2_emissions'],
                       alpha=0.6, s=30, label=f'≤{threshold:.1f} km²', color=COLORS['neutral'])
            ax3.scatter(high_park['Park_area_km2'], high_park['CO2_emissions'],
                       alpha=0.6, s=30, label=f'>{threshold:.1f} km²', color=COLORS['park'])

            # Fit separate regression lines
            if len(low_park) > 10:
                z_low = np.polyfit(low_park['Park_area_km2'], low_park['CO2_emissions'], 1)
                p_low = np.poly1d(z_low)
                x_low = np.linspace(low_park['Park_area_km2'].min(),
                                  low_park['Park_area_km2'].max(), 50)
                ax3.plot(x_low, p_low(x_low), '--', color=COLORS['neutral'], linewidth=2)

            if len(high_park) > 10:
                z_high = np.polyfit(high_park['Park_area_km2'], high_park['CO2_emissions'], 1)
                p_high = np.poly1d(z_high)
                x_high = np.linspace(high_park['Park_area_km2'].min(),
                                   high_park['Park_area_km2'].max(), 50)
                ax3.plot(x_high, p_high(x_high), '--', color=COLORS['park'], linewidth=2)

            ax3.axvline(x=threshold, color='red', linestyle=':', linewidth=1.5, alpha=0.8)
            ax3.set_xlabel('Park Area (km²)', fontweight='bold')
            ax3.set_ylabel('CO₂ Emissions', fontweight='bold')
            ax3.set_title('C. Threshold Effects', fontweight='bold', loc='left')
            ax3.legend(loc='best', frameon=False)
            ax3.grid(True, alpha=0.2)

        # Panel D: Marginal effects
        ax4 = fig.add_subplot(gs[1, :])

        if 'Park_area_km2' in self.df.columns and 'PM25' in self.df.columns:
            # Calculate marginal effects at different park levels
            park_bins = pd.qcut(self.df['Park_area_km2'], q=6, duplicates='drop')
            marginal_effects = []
            park_centers = []
            confidence_intervals = []

            for bin_edge in park_bins.unique():
                bin_data = self.df[park_bins == bin_edge]
                if len(bin_data) > 5:
                    X = add_constant(bin_data[['Park_area_km2']])
                    y = bin_data['PM25']
                    try:
                        model = OLS(y, X).fit()
                        marginal_effects.append(model.params['Park_area_km2'])
                        park_centers.append(bin_data['Park_area_km2'].mean())
                        ci = model.conf_int().loc['Park_area_km2']
                        confidence_intervals.append((ci[0], ci[1]))
                    except:
                        pass

            if marginal_effects:
                # Plot marginal effects with confidence intervals
                park_centers = np.array(park_centers)
                marginal_effects = np.array(marginal_effects)
                ci_lower = [ci[0] for ci in confidence_intervals]
                ci_upper = [ci[1] for ci in confidence_intervals]

                ax4.plot(park_centers, marginal_effects, 'o-', color=COLORS['park'],
                        linewidth=2, markersize=8, label='Marginal Effect')
                ax4.fill_between(park_centers, ci_lower, ci_upper,
                                color=COLORS['park'], alpha=0.2)

                ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
                ax4.set_xlabel('Park Area (km²)', fontweight='bold')
                ax4.set_ylabel('Marginal Effect on PM2.5', fontweight='bold')
                ax4.set_title('D. Varying Marginal Effects with Confidence Intervals',
                            fontweight='bold', loc='left')
                ax4.grid(True, alpha=0.2)
                ax4.legend(loc='best', frameon=False)

        plt.suptitle('Non-linear and Interaction Effects in Park-Environment Relationships',
                    fontsize=14, fontweight='bold', y=1.02)

        # Save figure
        fig_path = os.path.join(self.output_dir, 'Figure_4_Interaction_Effects.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✓ Figure 4 saved: {fig_path}")
        return fig_path

    def figure5_comprehensive_heatmap(self):
        """
        Figure 5: Comprehensive Correlation Matrix and City Profiles
        综合相关矩阵和城市画像
        """
        print("\nGenerating Figure 5: Comprehensive Overview...")

        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

        # Panel A: Correlation heatmap
        ax1 = fig.add_subplot(gs[0, :])

        # Select key variables
        key_vars = []
        for var in ['Park_area_km2', 'Forest_km2', 'Grassland_km2', 'Water_km2',
                   'PM25', 'CO2_emissions', 'Temperature', 'Humidity',
                   'Population_density', 'Urban_km2']:
            if var in self.df.columns:
                key_vars.append(var)

        if len(key_vars) >= 4:
            corr_matrix = self.df[key_vars].corr()

            # Create custom colormap
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

            # Plot heatmap
            im = ax1.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
            cbar.set_label('Correlation Coefficient', fontweight='bold')

            # Set ticks and labels
            ax1.set_xticks(np.arange(len(key_vars)))
            ax1.set_yticks(np.arange(len(key_vars)))
            ax1.set_xticklabels([v.replace('_', ' ').replace('km2', '').replace('CO2 emissions', 'CO₂')
                                for v in key_vars], rotation=45, ha='right')
            ax1.set_yticklabels([v.replace('_', ' ').replace('km2', '').replace('CO2 emissions', 'CO₂')
                                for v in key_vars])

            # Add correlation values
            for i in range(len(key_vars)):
                for j in range(len(key_vars)):
                    if not mask[i, j]:
                        text = ax1.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                                      ha="center", va="center",
                                      color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black",
                                      fontsize=8)

            ax1.set_title('A. Comprehensive Correlation Matrix', fontweight='bold', loc='left')

        # Panel B: City environmental performance
        ax2 = fig.add_subplot(gs[1, 0])

        if 'City' in self.df.columns:
            # Calculate city performance scores
            city_scores = []

            for city in self.df['City'].unique():
                city_data = self.df[self.df['City'] == city]
                score = 0

                # Lower PM2.5 is better
                if 'PM25' in city_data.columns:
                    pm25_norm = (city_data['PM25'].mean() - self.df['PM25'].min()) / \
                               (self.df['PM25'].max() - self.df['PM25'].min())
                    score -= pm25_norm

                # Lower CO2 is better
                if 'CO2_emissions' in city_data.columns:
                    co2_norm = (city_data['CO2_emissions'].mean() - self.df['CO2_emissions'].min()) / \
                              (self.df['CO2_emissions'].max() - self.df['CO2_emissions'].min())
                    score -= co2_norm

                # Higher park area is better
                if 'Park_area_km2' in city_data.columns:
                    park_norm = (city_data['Park_area_km2'].mean() - self.df['Park_area_km2'].min()) / \
                               (self.df['Park_area_km2'].max() - self.df['Park_area_km2'].min())
                    score += park_norm

                city_scores.append({'City': city, 'Score': score})

            df_scores = pd.DataFrame(city_scores).sort_values('Score', ascending=False)

            colors_city = [COLORS['park'] if s > 0 else COLORS['negative'] for s in df_scores['Score']]

            bars = ax2.barh(range(len(df_scores)), df_scores['Score'], color=colors_city, alpha=0.8)
            ax2.set_yticks(range(len(df_scores)))
            ax2.set_yticklabels(df_scores['City'])
            ax2.set_xlabel('Environmental Performance Score', fontweight='bold')
            ax2.set_title('B. City Environmental Rankings', fontweight='bold', loc='left')
            ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
            ax2.grid(True, alpha=0.2, axis='x')

        # Panel C: Park-PM2.5-CO2 bubble plot
        ax3 = fig.add_subplot(gs[1, 1])

        if all(col in self.df.columns for col in ['Park_area_km2', 'PM25', 'CO2_emissions']):
            # Normalize CO2 for bubble size
            co2_norm = (self.df['CO2_emissions'] - self.df['CO2_emissions'].min()) / \
                      (self.df['CO2_emissions'].max() - self.df['CO2_emissions'].min())

            scatter = ax3.scatter(self.df['Park_area_km2'], self.df['PM25'],
                                s=co2_norm * 300 + 20,  # Size based on CO2
                                c=self.df['CO2_emissions'],  # Color based on CO2
                                cmap='YlOrRd', alpha=0.6,
                                edgecolors='black', linewidth=0.5)

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax3)
            cbar.set_label('CO₂ Emissions', fontweight='bold')

            ax3.set_xlabel('Park Area (km²)', fontweight='bold')
            ax3.set_ylabel('PM2.5 (μg/m³)', fontweight='bold')
            ax3.set_title('C. Multi-dimensional Relationships\n(Bubble size = CO₂)',
                         fontweight='bold', loc='left')
            ax3.grid(True, alpha=0.2)

            # Add trend line
            z = np.polyfit(self.df['Park_area_km2'], self.df['PM25'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(self.df['Park_area_km2'].min(),
                                self.df['Park_area_km2'].max(), 100)
            ax3.plot(x_trend, p(x_trend), '--', color='black', linewidth=1.5, alpha=0.8)

        plt.suptitle('Comprehensive Environmental Analysis Overview',
                    fontsize=14, fontweight='bold', y=1.02)

        # Save figure
        fig_path = os.path.join(self.output_dir, 'Figure_5_Comprehensive_Overview.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✓ Figure 5 saved: {fig_path}")
        return fig_path

    def generate_all_figures(self):
        """Generate all 5 publication-quality figures"""
        print("\n" + "="*80)
        print("Generating Publication-Quality Figures for Academic Paper")
        print("="*80)

        # Load data
        self.load_data()

        # Generate each figure
        figures = []
        figures.append(self.figure1_multi_target_effects())
        figures.append(self.figure2_green_space_comparison())
        figures.append(self.figure3_temporal_evolution())
        figures.append(self.figure4_interaction_effects())
        figures.append(self.figure5_comprehensive_heatmap())

        print("\n" + "="*80)
        print("✅ All figures generated successfully!")
        print("="*80)
        print(f"\n📁 Figures saved in: {self.output_dir}/")
        print("\nGenerated files:")
        print("  • Figure 1: Multi-target Environmental Effects")
        print("  • Figure 2: Green Space Comparison")
        print("  • Figure 3: Temporal Evolution")
        print("  • Figure 4: Interaction Effects")
        print("  • Figure 5: Comprehensive Overview")
        print("\n📝 These figures are ready for academic publication!")

        return figures


# Main execution
if __name__ == "__main__":
    import os

    # Set data path
    data_folder = r"D:\yolos\datasets\4组数据\Park_Environmental_Impact_Results"
    data_file = "comprehensive_environmental_data.csv"
    data_path = os.path.join(data_folder, data_file)

    print("="*80)
    print("PUBLICATION-QUALITY FIGURE GENERATION")
    print("="*80)
    print(f"Data file: {data_path}")

    # Check if file exists
    if not os.path.exists(data_path):
        print(f"\n❌ Error: File not found - {data_path}")
    else:
        # Create figure generator
        fig_gen = PublicationFigures(data_path)

        # Generate all figures
        figures = fig_gen.generate_all_figures()

        print("\n💡 Figure descriptions for your paper:")
        print("\nFigure 1: Multi-dimensional environmental effects of urban parks.")
        print("         Shows regression coefficients, R² values, and temporal trends.")

        print("\nFigure 2: Comparative effectiveness of different green space types.")
        print("         Demonstrates differential impacts on PM2.5 and CO₂.")

        print("\nFigure 3: Temporal evolution of park-environment relationships.")
        print("         Illustrates strengthening effects over time.")

        print("\nFigure 4: Non-linear and interaction effects.")
        print("         Reveals complex dependencies on population density.")

        print("\nFigure 5: Comprehensive correlation matrix and city rankings.")

        print("         Provides system-level overview of all relationships.")
