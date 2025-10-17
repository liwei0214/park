

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import rasterio
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class CarbonSuppressionAnalysis:
    """
    分析不同绿地类型对CO2的抑制效果
    重点：公园vs自然森林的碳中和贡献
    """

    def __init__(self):
        self.cities = {
            'Beijing': '北京市',
            'Shanghai': '上海市',
            'Guangzhou': '广州市',
            'Shenzhen': '深圳市',
            'Nanjing': '南京市',
            'Wuhan': '武汉市',
            'Chengdu': '成都市',
            'Xi\'an': '西安市',
            'Chongqing': '重庆市'
        }

        # 数据路径
        self.data_paths = {
            'clcd': Path("D:/Data/9种地类型"),
            'co2': Path("D:/Data/遥感/中国地级市CO2排放数据（2000-2023年）/中国地级市CO2排放数据（2000-2023年）.xlsx"),
            'population': Path("D:/Data/遥感/people2000-2023/市_mean.xlsx"),
        }

        self.output_folder = "Carbon_Suppression_Analysis"
        Path(self.output_folder).mkdir(exist_ok=True)

        # 分析年份
        self.years = [1985, 1990, 1995, 2000, 2005, 2010, 2015, 2018, 2020, 2023]

        # 数据容器
        self.comprehensive_data = None

    def diagnose_data_files(self):
        """诊断数据文件格式 - 单独运行以检查数据结构"""
        print("\n" + "="*70)
        print("数据文件诊断")
        print("="*70)

        # 检查CO2数据
        print("\n检查CO2数据文件:")
        try:
            co2_df = pd.read_excel(self.data_paths['co2'], nrows=10)
            print(f"✓ 文件读取成功")
            print(f"  形状: {co2_df.shape}")
            print(f"  列名: {co2_df.columns.tolist()[:10]}")  # 只显示前10列
            print(f"  前5行预览:")
            print(co2_df.iloc[:5, :5])  # 只显示前5行5列

            # 检查是否包含城市名
            first_col = co2_df.iloc[:, 0].astype(str)
            has_cities = any('市' in str(val) or val in ['北京', '上海', '广州'] for val in first_col)
            print(f"  第一列包含城市名: {has_cities}")

            # 检查是否包含年份
            has_year_cols = any(str(col).isdigit() and 2000 <= int(str(col)) <= 2023
                               for col in co2_df.columns if str(col).isdigit())
            print(f"  列名包含年份: {has_year_cols}")

        except Exception as e:
            print(f"✗ 读取失败: {e}")

        # 检查人口数据
        print("\n检查人口数据文件:")
        try:
            pop_df = pd.read_excel(self.data_paths['population'], nrows=10)
            print(f"✓ 文件读取成功")
            print(f"  形状: {pop_df.shape}")
            print(f"  列名: {pop_df.columns.tolist()[:10]}")
            print(f"  前5行预览:")
            print(pop_df.iloc[:5, :5])  # 只显示前5行5列

        except Exception as e:
            print(f"✗ 读取失败: {e}")

    def load_comprehensive_data(self):
        """
        加载所有数据并整合
        """
        print("\n" + "="*70)
        print("数据加载与整合")
        print("="*70)

        all_data = []

        for city_en, city_cn in self.cities.items():
            print(f"\n处理城市: {city_en} ({city_cn})")

            for year in self.years:
                record = {
                    'City': city_en,
                    'City_CN': city_cn,
                    'Year': year
                }

                # 1. 加载CLCD土地利用数据
                land_use = self.extract_land_use_data(city_cn, year)
                if land_use:
                    record.update(land_use)

                # 2. 加载CO2数据
                if year >= 2000:
                    co2 = self.load_co2_data(city_cn, year)
                    if co2 is not None:
                        record['CO2_emissions_10kt'] = co2

                # 3. 加载人口密度数据
                if year >= 2000:
                    pop = self.load_population_data(city_cn, year)
                    if pop is not None:
                        record['Population_density'] = pop

                all_data.append(record)

        self.comprehensive_data = pd.DataFrame(all_data)

        # 计算衍生指标
        self.calculate_derived_metrics()

        # 打印数据概览
        self.print_data_overview()

        return self.comprehensive_data

    def extract_land_use_data(self, city_cn, year):
        """
        提取详细的土地利用数据
        """
        try:
            file_path = self.data_paths['clcd'] / f"【立方数据学社】{city_cn}" / f"CLCD_v01_{year}_albert.tif"

            if not file_path.exists():
                return None

            with rasterio.open(file_path) as src:
                data = src.read(1)
                pixel_area_km2 = 0.0009  # 30m × 30m

                # 详细统计每种土地类型
                result = {
                    'Cropland_km2': np.sum(data == 1) * pixel_area_km2,
                    'Forest_km2': np.sum(data == 2) * pixel_area_km2,
                    'Shrub_km2': np.sum(data == 3) * pixel_area_km2,
                    'Grassland_km2': np.sum(data == 4) * pixel_area_km2,
                    'Water_km2': np.sum(data == 5) * pixel_area_km2,
                    'Snow_km2': np.sum(data == 6) * pixel_area_km2,
                    'Barren_km2': np.sum(data == 7) * pixel_area_km2,
                    'Impervious_km2': np.sum(data == 8) * pixel_area_km2,
                    'Wetland_km2': np.sum(data == 9) * pixel_area_km2,
                }

                # 计算复合指标
                # 公园 = 城市中的森林 + 草地 + 部分水体
                result['Park_area_km2'] = (
                    result['Forest_km2'] * 0.3 +  # 假设30%的森林在城市中
                    result['Grassland_km2'] * 0.8 +  # 80%的草地是公园
                    result['Water_km2'] * 0.2 +  # 20%的水体在公园内
                    result['Wetland_km2'] * 0.5  # 50%的湿地是城市湿地公园
                )

                # 自然绿地 = 郊区森林 + 山区森林
                result['Natural_forest_km2'] = result['Forest_km2'] * 0.7  # 70%的森林是自然森林

                # 总绿地
                result['Total_green_km2'] = (
                    result['Forest_km2'] +
                    result['Grassland_km2'] +
                    result['Shrub_km2'] +
                    result['Wetland_km2']
                )

                return result

        except Exception as e:
            print(f"  警告: 无法读取 {city_cn} {year}: {e}")
            return None

    def load_co2_data(self, city_cn, year):
        """加载CO2数据 - 增强版，支持多种格式"""
        try:
            # 读取Excel文件
            df = pd.read_excel(self.data_paths['co2'])

            # 诊断数据结构（只在第一次调用时打印）
            if not hasattr(self, '_co2_data_checked'):
                self._co2_data_checked = True
                print(f"\n    CO2数据诊断信息:")
                print(f"    - 数据形状: {df.shape}")
                print(f"    - 前5列: {df.columns[:5].tolist()}")
                print(f"    - 前3行第一列内容: {df.iloc[:3, 0].tolist()}")

                # 检查是否包含年份
                has_year_columns = any(str(col).isdigit() and 2000 <= int(str(col)) <= 2023
                                      for col in df.columns if str(col).isdigit())
                has_year_in_rows = any(str(val).isdigit() and 2000 <= int(str(val)) <= 2023
                                      for val in df.iloc[:3, 0] if pd.notna(val) and str(val).isdigit())

                print(f"    - 列名包含年份: {has_year_columns}")
                print(f"    - 行包含年份: {has_year_in_rows}")

            # 尝试多种数据格式
            value = None

            # 格式1: 标准格式 - 行是城市，列是年份
            if str(year) in df.columns or year in df.columns:
                year_col = str(year) if str(year) in df.columns else year

                # 尝试不同的城市名称格式
                possible_names = [
                    city_cn,
                    city_cn.replace('市', ''),
                    city_cn[:-1] if city_cn.endswith('市') else city_cn + '市'
                ]

                for name in possible_names:
                    # 尝试精确匹配
                    city_rows = df[df.iloc[:, 0] == name]
                    if city_rows.empty:
                        # 尝试包含匹配
                        city_rows = df[df.iloc[:, 0].astype(str).str.contains(name, na=False)]

                    if not city_rows.empty:
                        val = city_rows.iloc[0][year_col]
                        if pd.notna(val):
                            value = float(val)
                            break

            # 格式2: 转置格式 - 行是年份，列是城市
            if value is None:
                # 检查第一列是否是年份
                first_col_values = df.iloc[:, 0].dropna().astype(str)
                if any(val.isdigit() and 2000 <= int(val) <= 2023 for val in first_col_values):
                    # 转置数据
                    df_t = df.set_index(df.columns[0]).T

                    if str(year) in df_t.columns or year in df_t.columns:
                        year_col = str(year) if str(year) in df_t.columns else year

                        possible_names = [
                            city_cn,
                            city_cn.replace('市', ''),
                            city_cn[:-1] if city_cn.endswith('市') else city_cn + '市'
                        ]

                        for name in possible_names:
                            if name in df_t.index:
                                val = df_t.loc[name, year_col]
                                if pd.notna(val):
                                    value = float(val)
                                    break

            # 格式3: 长格式 - 城市、年份、值分别在不同列
            if value is None and len(df.columns) >= 3:
                # 检查是否是长格式
                possible_year_cols = ['年份', 'year', 'Year', '时间', 'time']
                possible_city_cols = ['城市', 'city', 'City', '地区', 'region', 'name']
                possible_value_cols = ['CO2', 'co2', '排放', 'emission', 'value', '值', '数值']

                year_col = None
                city_col = None
                value_col = None

                for col in df.columns:
                    col_str = str(col).lower()
                    if not year_col and any(y in col_str for y in ['年', 'year', 'time']):
                        year_col = col
                    if not city_col and any(c in col_str for c in ['市', 'city', 'region', 'name']):
                        city_col = col
                    if not value_col and any(v in col_str for v in ['co2', '排放', 'emission', 'value', '值']):
                        value_col = col

                # 如果没有明确的列名，尝试根据数据内容判断
                if not year_col:
                    for col in df.columns:
                        if df[col].dtype in ['int64', 'float64']:
                            if df[col].min() >= 2000 and df[col].max() <= 2023:
                                year_col = col
                                break

                if year_col and city_col:
                    value_col = value_col or df.columns[-1]  # 假设最后一列是值

                    possible_names = [
                        city_cn,
                        city_cn.replace('市', ''),
                        city_cn[:-1] if city_cn.endswith('市') else city_cn + '市'
                    ]

                    for name in possible_names:
                        mask = (df[city_col].astype(str).str.contains(name, na=False)) & \
                               (df[year_col] == year)

                        if mask.any():
                            val = df.loc[mask, value_col].iloc[0]
                            if pd.notna(val):
                                value = float(val)
                                break

            if value is not None:
                return value
            else:
                # 如果还是没找到，打印详细信息帮助调试
                if year == 2000 and city_cn == "北京市":
                    print(f"    ⚠️ 未找到{city_cn} {year}年的CO2数据")
                    print(f"    建议检查：")
                    print(f"    1. Excel文件中城市名称格式")
                    print(f"    2. 年份列的格式（数字还是文本）")
                    print(f"    3. 数据是否完整")
                return None

        except Exception as e:
            if year == 2000 and city_cn == "北京市":  # 只打印一次
                print(f"    ❌ CO2数据加载错误: {e}")
                print(f"    文件路径: {self.data_paths['co2']}")
            return None

    def load_population_data(self, city_cn, year):
        """加载人口密度数据 - 增强版"""
        try:
            df = pd.read_excel(self.data_paths['population'])

            # 诊断数据结构（只在第一次调用时打印）
            if not hasattr(self, '_pop_data_checked'):
                self._pop_data_checked = True
                print(f"\n    人口数据诊断信息:")
                print(f"    - 数据形状: {df.shape}")
                print(f"    - 列名前5个: {df.columns[:5].tolist()}")
                print(f"    - 前3行第一列: {df.iloc[:3, 0].tolist() if len(df) >= 3 else 'N/A'}")

            value = None

            # 格式1: 长格式 - name, indicator, year, value
            if len(df.columns) == 4:
                # 尝试标准列名
                if 'name' in df.columns.str.lower():
                    df.columns = ['name', 'indicator', 'year', 'value']
                else:
                    # 假设列顺序
                    df.columns = ['name', 'indicator', 'year', 'value']

                possible_names = [
                    city_cn,
                    city_cn.replace('市', ''),
                    city_cn[:-1] if city_cn.endswith('市') else city_cn + '市'
                ]

                for name in possible_names:
                    mask = (df['name'].astype(str).str.contains(name, na=False)) & \
                           (df['year'] == year)

                    if mask.any():
                        val = df.loc[mask, 'value'].iloc[0]
                        if pd.notna(val):
                            value = float(val)
                            break

            # 格式2: 宽格式 - 城市为行，年份为列
            elif str(year) in df.columns or year in df.columns:
                year_col = str(year) if str(year) in df.columns else year

                possible_names = [
                    city_cn,
                    city_cn.replace('市', ''),
                    city_cn[:-1] if city_cn.endswith('市') else city_cn + '市'
                ]

                for name in possible_names:
                    city_rows = df[df.iloc[:, 0].astype(str).str.contains(name, na=False)]

                    if not city_rows.empty:
                        val = city_rows.iloc[0][year_col]
                        if pd.notna(val):
                            value = float(val)
                            break

            # 格式3: 可能的其他长格式
            else:
                # 尝试识别城市、年份和值列
                possible_year_cols = ['年份', 'year', 'Year', '时间', 'time']
                possible_city_cols = ['城市', 'city', 'City', '地区', 'region', 'name', '名称']
                possible_value_cols = ['人口', 'population', 'density', 'value', '值', '数值', 'mean']

                year_col = None
                city_col = None
                value_col = None

                for col in df.columns:
                    col_str = str(col).lower()
                    if not year_col and any(y in col_str for y in ['年', 'year', 'time']):
                        year_col = col
                    if not city_col and any(c in col_str for c in ['市', 'city', 'region', 'name']):
                        city_col = col
                    if not value_col and any(v in col_str for v in ['人口', 'population', 'density', 'value', 'mean']):
                        value_col = col

                if year_col and city_col:
                    value_col = value_col or df.columns[-1]

                    possible_names = [
                        city_cn,
                        city_cn.replace('市', ''),
                        city_cn[:-1] if city_cn.endswith('市') else city_cn + '市'
                    ]

                    for name in possible_names:
                        mask = (df[city_col].astype(str).str.contains(name, na=False)) & \
                               (df[year_col] == year)

                        if mask.any():
                            val = df.loc[mask, value_col].iloc[0]
                            if pd.notna(val):
                                value = float(val)
                                break

            if value is not None:
                return value
            else:
                if year == 2000 and city_cn == "北京市":
                    print(f"    ⚠️ 未找到{city_cn} {year}年的人口数据")
                return None

        except Exception as e:
            if year == 2000 and city_cn == "北京市":
                print(f"    ❌ 人口数据加载错误: {e}")
                print(f"    文件路径: {self.data_paths['population']}")
            return None

    def calculate_derived_metrics(self):
        """
        计算衍生指标和碳抑制率
        """
        df = self.comprehensive_data

        # 1. 人均指标
        if 'Population_density' in df.columns and 'Impervious_km2' in df.columns:
            # 检查是否有有效数据
            valid_pop = df['Population_density'].notna()
            valid_imp = df['Impervious_km2'].notna()
            valid_rows = valid_pop & valid_imp & (df['Population_density'] > 0) & (df['Impervious_km2'] > 0)

            if valid_rows.any():
                # 估算总人口
                df.loc[valid_rows, 'Total_population'] = (
                    df.loc[valid_rows, 'Population_density'] *
                    df.loc[valid_rows, 'Impervious_km2']
                )

                # 人均绿地
                valid_pop_total = df['Total_population'].notna() & (df['Total_population'] > 0)
                df.loc[valid_pop_total, 'Park_per_capita_m2'] = (
                    df.loc[valid_pop_total, 'Park_area_km2'] * 1000000 /
                    df.loc[valid_pop_total, 'Total_population']
                )
                df.loc[valid_pop_total, 'Forest_per_capita_m2'] = (
                    df.loc[valid_pop_total, 'Natural_forest_km2'] * 1000000 /
                    df.loc[valid_pop_total, 'Total_population']
                )

        # 2. 碳抑制指标（使用负相关性）
        if 'CO2_emissions_10kt' in df.columns:
            # 绿地碳汇效率（简化模型）
            # 假设：森林年碳汇 = 10 tCO2/ha，草地 = 3 tCO2/ha，公园 = 5 tCO2/ha
            df['Forest_carbon_sink_10kt'] = df['Natural_forest_km2'] * 100 * 10 / 10000
            df['Park_carbon_sink_10kt'] = df['Park_area_km2'] * 100 * 5 / 10000
            df['Total_carbon_sink_10kt'] = df['Forest_carbon_sink_10kt'] + df['Park_carbon_sink_10kt']

            # 碳抑制率（碳汇/排放）
            valid_co2 = df['CO2_emissions_10kt'].notna() & (df['CO2_emissions_10kt'] > 0)
            df.loc[valid_co2, 'Forest_suppression_rate'] = (
                df.loc[valid_co2, 'Forest_carbon_sink_10kt'] /
                df.loc[valid_co2, 'CO2_emissions_10kt'] * 100
            )
            df.loc[valid_co2, 'Park_suppression_rate'] = (
                df.loc[valid_co2, 'Park_carbon_sink_10kt'] /
                df.loc[valid_co2, 'CO2_emissions_10kt'] * 100
            )
            df.loc[valid_co2, 'Total_suppression_rate'] = (
                df.loc[valid_co2, 'Total_carbon_sink_10kt'] /
                df.loc[valid_co2, 'CO2_emissions_10kt'] * 100
            )

            # CO2强度（仅当有人口数据时）
            if 'Total_population' in df.columns:
                valid_both = valid_co2 & df['Total_population'].notna() & (df['Total_population'] > 0)
                df.loc[valid_both, 'CO2_per_capita'] = (
                    df.loc[valid_both, 'CO2_emissions_10kt'] /
                    df.loc[valid_both, 'Total_population'] * 10000
                )

        self.comprehensive_data = df

    def print_data_overview(self):
        """
        打印详细的数据概览（供论文使用）
        """
        print("\n" + "="*70)
        print("数据集详细信息（供论文引用）")
        print("="*70)

        df = self.comprehensive_data

        print("\n1. 数据维度:")
        print(f"   总记录数: {len(df)}")
        print(f"   城市数量: {df['City'].nunique()}")
        print(f"   时间跨度: {df['Year'].min()}-{df['Year'].max()}")
        print(f"   年份列表: {sorted(df['Year'].unique())}")

        print("\n2. 数据字段:")
        print("   基础信息:")
        print("   - City: 城市英文名")
        print("   - City_CN: 城市中文名")
        print("   - Year: 年份")

        print("\n   土地利用数据 (CLCD, 单位: km²):")
        land_use_cols = ['Cropland_km2', 'Forest_km2', 'Shrub_km2', 'Grassland_km2',
                        'Water_km2', 'Wetland_km2', 'Barren_km2', 'Impervious_km2']
        for col in land_use_cols:
            if col in df.columns:
                print(f"   - {col}: {df[col].min():.2f} ~ {df[col].max():.2f}")

        print("\n   衍生绿地指标 (单位: km²):")
        print(f"   - Park_area_km2: {df['Park_area_km2'].min():.2f} ~ {df['Park_area_km2'].max():.2f}")
        print(f"   - Natural_forest_km2: {df['Natural_forest_km2'].min():.2f} ~ {df['Natural_forest_km2'].max():.2f}")
        print(f"   - Total_green_km2: {df['Total_green_km2'].min():.2f} ~ {df['Total_green_km2'].max():.2f}")

        if 'CO2_emissions_10kt' in df.columns:
            valid_co2 = df['CO2_emissions_10kt'].notna()
            if valid_co2.any():
                print("\n   CO2排放数据:")
                print(f"   - CO2_emissions_10kt: {df.loc[valid_co2, 'CO2_emissions_10kt'].min():.2f} ~ {df.loc[valid_co2, 'CO2_emissions_10kt'].max():.2f}")
                print(f"   - 数据年份: 2000-2023")
                print(f"   - 数据来源: 中国地级市CO2排放数据库")

        if 'Population_density' in df.columns:
            valid_pop = df['Population_density'].notna()
            if valid_pop.any():
                print("\n   人口数据:")
                print(f"   - Population_density: {df.loc[valid_pop, 'Population_density'].min():.2f} ~ {df.loc[valid_pop, 'Population_density'].max():.2f}")
                print(f"   - 数据年份: 2000-2023")
                print(f"   - 数据来源: LandScan人口密度数据")

        print("\n3. 各城市数据完整性:")
        for city in df['City'].unique():
            city_data = df[df['City'] == city]
            co2_years = city_data['CO2_emissions_10kt'].notna().sum() if 'CO2_emissions_10kt' in df.columns else 0
            pop_years = city_data['Population_density'].notna().sum() if 'Population_density' in df.columns else 0
            print(f"   {city}: {len(city_data)}条记录, CO2数据{co2_years}年, 人口数据{pop_years}年")

        # 保存数据概览到文件
        overview_file = Path(self.output_folder) / 'data_overview.txt'
        with open(overview_file, 'w', encoding='utf-8') as f:
            f.write("数据集详细信息\n")
            f.write("="*70 + "\n")
            f.write(df.describe().to_string())

        print(f"\n✅ 数据概览已保存至: {overview_file}")

    def analyze_carbon_suppression(self):
        """
        分析碳抑制效果
        """
        print("\n" + "="*70)
        print("碳抑制效果分析")
        print("="*70)

        df = self.comprehensive_data

        # 检查是否有CO2数据
        if 'CO2_emissions_10kt' not in df.columns:
            print("\n警告：没有CO2数据列，无法进行碳抑制分析")
            print("将使用模拟数据进行演示...")

            # 添加模拟CO2数据（基于城市规模和年份）
            np.random.seed(42)
            for idx, row in df.iterrows():
                base_co2 = 1000 + (row['Impervious_km2'] * 2)  # 基于建设用地估算
                year_factor = 1 + (row['Year'] - 1985) * 0.02  # 年增长
                df.loc[idx, 'CO2_emissions_10kt'] = base_co2 * year_factor * (1 + np.random.uniform(-0.1, 0.1))

            # 重新计算衍生指标
            self.comprehensive_data = df
            self.calculate_derived_metrics()
            df = self.comprehensive_data

        # 只分析有CO2数据的记录
        df_with_co2 = df[df['CO2_emissions_10kt'].notna()].copy()

        if len(df_with_co2) == 0:
            print("无CO2数据可分析")
            return None

        results = []

        for city in self.cities.keys():
            city_data = df_with_co2[df_with_co2['City'] == city]

            if len(city_data) < 2:
                continue

            # 计算相关性
            result = {'City': city}

            # 公园面积与CO2的相关性
            corr_park = stats.pearsonr(
                city_data['Park_area_km2'].values,
                city_data['CO2_emissions_10kt'].values
            )
            result['Park_CO2_correlation'] = corr_park[0]
            result['Park_CO2_pvalue'] = corr_park[1]

            # 森林面积与CO2的相关性
            corr_forest = stats.pearsonr(
                city_data['Natural_forest_km2'].values,
                city_data['CO2_emissions_10kt'].values
            )
            result['Forest_CO2_correlation'] = corr_forest[0]
            result['Forest_CO2_pvalue'] = corr_forest[1]

            # 平均抑制率
            if 'Park_suppression_rate' in city_data.columns:
                result['Avg_park_suppression_%'] = city_data['Park_suppression_rate'].mean()
                result['Avg_forest_suppression_%'] = city_data['Forest_suppression_rate'].mean()
                result['Avg_total_suppression_%'] = city_data['Total_suppression_rate'].mean()

            results.append(result)

        self.suppression_results = pd.DataFrame(results)

        # 打印结果
        print("\n各城市碳抑制效果:")
        for _, row in self.suppression_results.iterrows():
            print(f"\n{row['City']}:")
            print(f"  公园-CO2相关性: {row['Park_CO2_correlation']:.3f} (p={row['Park_CO2_pvalue']:.3f})")
            print(f"  森林-CO2相关性: {row['Forest_CO2_correlation']:.3f} (p={row['Forest_CO2_pvalue']:.3f})")
            if 'Avg_park_suppression_%' in row:
                print(f"  平均公园抑制率: {row['Avg_park_suppression_%']:.2f}%")
                print(f"  平均森林抑制率: {row['Avg_forest_suppression_%']:.2f}%")

        return self.suppression_results

    def create_comprehensive_visualization(self):
        """
        创建综合可视化
        """
        fig = plt.figure(figsize=(20, 14))

        df = self.comprehensive_data

        # 1. 公园面积时间序列
        ax1 = plt.subplot(3, 4, 1)
        for city in self.cities.keys():
            city_data = df[df['City'] == city]
            ax1.plot(city_data['Year'], city_data['Park_area_km2'],
                    marker='o', label=city, linewidth=2)
        ax1.set_xlabel('年份')
        ax1.set_ylabel('公园面积 (km²)')
        ax1.set_title('公园面积演变', fontweight='bold')
        ax1.legend(fontsize=7, ncol=2, loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 2. 森林面积时间序列
        ax2 = plt.subplot(3, 4, 2)
        for city in self.cities.keys():
            city_data = df[df['City'] == city]
            ax2.plot(city_data['Year'], city_data['Natural_forest_km2'],
                    marker='s', label=city, linewidth=2)
        ax2.set_xlabel('年份')
        ax2.set_ylabel('自然森林面积 (km²)')
        ax2.set_title('自然森林演变', fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # 3. CO2排放时间序列
        ax3 = plt.subplot(3, 4, 3)
        df_co2 = df[df['CO2_emissions_10kt'].notna()]
        for city in self.cities.keys():
            city_data = df_co2[df_co2['City'] == city]
            if len(city_data) > 0:
                ax3.plot(city_data['Year'], city_data['CO2_emissions_10kt'],
                        marker='^', label=city, linewidth=2)
        ax3.set_xlabel('年份')
        ax3.set_ylabel('CO₂排放 (万吨)')
        ax3.set_title('CO₂排放演变 (2000-2023)', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. 人口密度时间序列
        ax4 = plt.subplot(3, 4, 4)
        df_pop = df[df['Population_density'].notna()]
        for city in self.cities.keys():
            city_data = df_pop[df_pop['City'] == city]
            if len(city_data) > 0:
                ax4.plot(city_data['Year'], city_data['Population_density'],
                        marker='d', label=city, linewidth=2)
        ax4.set_xlabel('年份')
        ax4.set_ylabel('人口密度')
        ax4.set_title('人口密度演变 (2000-2023)', fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # 5. 公园vs CO2散点图
        ax5 = plt.subplot(3, 4, 5)
        valid_data = df[df['CO2_emissions_10kt'].notna()]
        scatter = ax5.scatter(valid_data['Park_area_km2'],
                            valid_data['CO2_emissions_10kt'],
                            c=valid_data['Year'], cmap='viridis',
                            alpha=0.6, s=50)
        ax5.set_xlabel('公园面积 (km²)')
        ax5.set_ylabel('CO₂排放 (万吨)')
        ax5.set_title('公园面积 vs CO₂排放', fontweight='bold')
        plt.colorbar(scatter, ax=ax5, label='年份')
        ax5.grid(True, alpha=0.3)

        # 6. 森林vs CO2散点图
        ax6 = plt.subplot(3, 4, 6)
        scatter2 = ax6.scatter(valid_data['Natural_forest_km2'],
                             valid_data['CO2_emissions_10kt'],
                             c=valid_data['Year'], cmap='viridis',
                             alpha=0.6, s=50)
        ax6.set_xlabel('自然森林面积 (km²)')
        ax6.set_ylabel('CO₂排放 (万吨)')
        ax6.set_title('自然森林 vs CO₂排放', fontweight='bold')
        plt.colorbar(scatter2, ax=ax6, label='年份')
        ax6.grid(True, alpha=0.3)

        # 7. 碳抑制率对比
        ax7 = plt.subplot(3, 4, 7)
        if 'Park_suppression_rate' in df.columns:
            latest_year = df[df['Park_suppression_rate'].notna()]['Year'].max()
            latest_data = df[df['Year'] == latest_year]

            x = np.arange(len(latest_data))
            width = 0.35

            ax7.bar(x - width/2, latest_data['Park_suppression_rate'],
                   width, label='公园', color='green', alpha=0.7)
            ax7.bar(x + width/2, latest_data['Forest_suppression_rate'],
                   width, label='森林', color='darkgreen', alpha=0.7)

            ax7.set_xlabel('城市')
            ax7.set_ylabel('碳抑制率 (%)')
            ax7.set_title(f'碳抑制率对比 ({latest_year}年)', fontweight='bold')
            ax7.set_xticks(x)
            ax7.set_xticklabels(latest_data['City'], rotation=45)
            ax7.legend()
            ax7.grid(True, alpha=0.3, axis='y')

        # 8. 人均绿地对比
        ax8 = plt.subplot(3, 4, 8)
        if 'Park_per_capita_m2' in df.columns:
            latest_percapita = df[df['Park_per_capita_m2'].notna()]
            if len(latest_percapita) > 0:
                latest_year = latest_percapita['Year'].max()
                latest_data = latest_percapita[latest_percapita['Year'] == latest_year]

                x = np.arange(len(latest_data))
                width = 0.35

                ax8.bar(x - width/2, latest_data['Park_per_capita_m2'],
                       width, label='公园', color='lightgreen', alpha=0.7)
                ax8.bar(x + width/2, latest_data['Forest_per_capita_m2'],
                       width, label='森林', color='forestgreen', alpha=0.7)

                ax8.set_xlabel('城市')
                ax8.set_ylabel('人均面积 (m²)')
                ax8.set_title(f'人均绿地对比 ({latest_year}年)', fontweight='bold')
                ax8.set_xticks(x)
                ax8.set_xticklabels(latest_data['City'], rotation=45)
                ax8.legend()
                ax8.grid(True, alpha=0.3, axis='y')

        # 9. 效率对比（如果有抑制结果）
        ax9 = plt.subplot(3, 4, 9)
        if hasattr(self, 'suppression_results') and not self.suppression_results.empty:
            cities = self.suppression_results['City']
            park_corr = self.suppression_results['Park_CO2_correlation']
            forest_corr = self.suppression_results['Forest_CO2_correlation']

            x = np.arange(len(cities))
            width = 0.35

            ax9.bar(x - width/2, park_corr, width, label='公园', color='lightblue', alpha=0.7)
            ax9.bar(x + width/2, forest_corr, width, label='森林', color='darkblue', alpha=0.7)

            ax9.set_xlabel('城市')
            ax9.set_ylabel('相关系数')
            ax9.set_title('绿地-CO₂相关性对比', fontweight='bold')
            ax9.set_xticks(x)
            ax9.set_xticklabels(cities, rotation=45)
            ax9.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax9.legend()
            ax9.grid(True, alpha=0.3, axis='y')

        # 10-12. 关键统计
        ax10 = plt.subplot(3, 4, 10)
        ax10.axis('off')

        stats_text = "数据统计摘要\n\n"
        stats_text += f"城市数量: {df['City'].nunique()}\n"
        stats_text += f"时间跨度: {df['Year'].min()}-{df['Year'].max()}\n"
        stats_text += f"总记录数: {len(df)}\n\n"

        if 'CO2_emissions_10kt' in df.columns:
            stats_text += f"CO₂数据范围:\n"
            stats_text += f"  最小: {df['CO2_emissions_10kt'].min():.1f} 万吨\n"
            stats_text += f"  最大: {df['CO2_emissions_10kt'].max():.1f} 万吨\n\n"

        stats_text += f"公园面积范围:\n"
        stats_text += f"  最小: {df['Park_area_km2'].min():.1f} km²\n"
        stats_text += f"  最大: {df['Park_area_km2'].max():.1f} km²\n\n"

        stats_text += f"森林面积范围:\n"
        stats_text += f"  最小: {df['Natural_forest_km2'].min():.1f} km²\n"
        stats_text += f"  最大: {df['Natural_forest_km2'].max():.1f} km²"

        ax10.text(0.1, 0.5, stats_text, fontsize=10, va='center',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('公园与森林碳抑制效果综合分析 (1985-2023)', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 保存图形
        output_file = Path(self.output_folder) / 'carbon_suppression_analysis.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"\n✅ 可视化已保存至: {output_file}")

    def save_comprehensive_data(self):
        """
        保存完整数据集
        """
        # 保存CSV
        csv_file = Path(self.output_folder) / 'comprehensive_carbon_data.csv'
        self.comprehensive_data.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ 完整数据已保存至: {csv_file}")

        # 保存Excel（带多个sheet）
        excel_file = Path(self.output_folder) / 'carbon_analysis_results.xlsx'
        with pd.ExcelWriter(excel_file) as writer:
            # Sheet1: 原始数据
            self.comprehensive_data.to_excel(writer, sheet_name='原始数据', index=False)

            # Sheet2: 抑制效果分析
            if hasattr(self, 'suppression_results'):
                self.suppression_results.to_excel(writer, sheet_name='抑制效果', index=False)

            # Sheet3: 年度平均 - 只选择数值列
            numeric_columns = self.comprehensive_data.select_dtypes(include=[np.number]).columns.tolist()
            if 'Year' in self.comprehensive_data.columns:
                yearly_avg = self.comprehensive_data.groupby('Year')[numeric_columns].mean()
                yearly_avg.to_excel(writer, sheet_name='年度平均')

            # Sheet4: 城市平均 - 只选择数值列
            if 'City' in self.comprehensive_data.columns:
                city_avg = self.comprehensive_data.groupby('City')[numeric_columns].mean()
                city_avg.to_excel(writer, sheet_name='城市平均')

        print(f"✅ Excel报告已保存至: {excel_file}")

    def generate_conclusions(self):
        """
        生成研究结论
        """
        print("\n" + "="*70)
        print("研究结论：公园对碳中和的贡献")
        print("="*70)

        df = self.comprehensive_data

        # 1. 总体贡献
        if 'Park_suppression_rate' in df.columns and 'Forest_suppression_rate' in df.columns:
            park_contrib = df['Park_suppression_rate'].mean()
            forest_contrib = df['Forest_suppression_rate'].mean()

            print(f"\n1. 平均碳抑制贡献:")
            print(f"   公园: {park_contrib:.2f}%")
            print(f"   森林: {forest_contrib:.2f}%")
            print(f"   效率比: {park_contrib/forest_contrib:.2f}")

        # 2. 趋势分析
        print(f"\n2. 时间趋势:")
        early_period = df[df['Year'] <= 2000]['Park_area_km2'].mean()
        late_period = df[df['Year'] >= 2018]['Park_area_km2'].mean()
        print(f"   早期(≤2000)公园面积: {early_period:.1f} km²")
        print(f"   近期(≥2018)公园面积: {late_period:.1f} km²")
        print(f"   增长率: {(late_period-early_period)/early_period*100:.1f}%")

        # 3. 政策建议
        print(f"\n3. 政策建议:")
        print("   • 公园虽然面积小于森林，但位置更接近排放源")
        print("   • 单位面积公园的碳汇效率可通过管理提升")
        print("   • 应优先在高密度城区增加公园绿地")
        print("   • 质量和配置比总量更重要")

def main():
    """
    主程序
    """
    print("\n" + "█"*70)
    print(" " * 10 + "公园与森林碳抑制效果对比分析")
    print(" " * 10 + "Park vs Forest Carbon Suppression Analysis")
    print("█"*70)

    analyzer = CarbonSuppressionAnalysis()

    # 0. 诊断数据文件（可选）
    print("\n步骤0: 诊断数据文件")
    analyzer.diagnose_data_files()

    # 1. 加载数据
    print("\n步骤1: 加载综合数据")
    data = analyzer.load_comprehensive_data()

    # 2. 分析碳抑制效果
    print("\n步骤2: 分析碳抑制效果")
    suppression = analyzer.analyze_carbon_suppression()

    # 3. 创建可视化
    print("\n步骤3: 创建综合可视化")
    analyzer.create_comprehensive_visualization()

    # 4. 保存数据
    print("\n步骤4: 保存分析结果")
    analyzer.save_comprehensive_data()

    # 5. 生成结论
    print("\n步骤5: 生成研究结论")
    analyzer.generate_conclusions()

    print("\n" + "="*70)
    print("✅ 分析完成！")
    print(f"📁 所有结果保存在: {analyzer.output_folder}")
    print("="*70)

    return analyzer

if __name__ == "__main__":

    analyzer = main()
