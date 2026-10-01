import pandas as pd
import numpy as np
from config import DATA_FILE, CLEANED_DATA_FILE

def load_data():
    df = pd.read_csv(DATA_FILE)
    print('1. ĐỌC DỮ LIỆU')
    print('Kích thước dữ liệu ban đầu:', df.shape)
    print('\n5 dòng đầu tiên:')
    print(df.head())
    return df

def check_data(df):
    print('2. KIỂM TRA CHẤT LƯỢNG DỮ LIỆU')
    print('\nDANH SÁCH CÁC CỘT:')
    print(df.columns.tolist())
    print('\nKIỂU DỮ LIỆU:')
    print(df.dtypes)
    print('\nSỐ LƯỢNG DỮ LIỆU THIẾU:')
    print(df.isnull().sum())
    missing_percent = (df.isnull().sum() / len(df) * 100).round(2)
    print('\nTỶ LỆ DỮ LIỆU THIẾU (%):')
    print(missing_percent)
    print('\nSỐ DÒNG TRÙNG:')
    print(df.duplicated().sum())
    print('\nTHỐNG KÊ MÔ TẢ:')
    print(df.describe(include='all').transpose())

def filter_data(df):
    print('3. LỌC DỮ LIỆU')
    df = df.copy()
    df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
    df = df[df['province_name'].astype(str).str.strip() == 'Hà Nội']
    print('\nSau khi lọc Hà Nội:')
    print('Số dòng:', len(df))
    df = df[(df['published_at'] >= '2025-01-01') & (df['published_at'] < '2025-07-01')]
    print('\nSau khi lọc 6 tháng đầu năm 2025:')
    print('Số dòng:', len(df))
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['area'] = pd.to_numeric(df['area'], errors='coerce')
    df = df[(df['price'] > 0) & (df['area'] > 0)]
    df = df.dropna(subset=['price', 'area', 'district_name', 'property_type_name', 'published_at'])
    print('\nSau khi loại bỏ dữ liệu quan trọng không hợp lệ:')
    print('Số dòng:', len(df))
    return df

def remove_duplicates(df):
    print('4. XỬ LÝ DỮ LIỆU TRÙNG')
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    print('Số dòng trùng đã xóa:', before - after)
    print('Số dòng còn lại:', after)
    return df

def remove_high_missing_columns(df):
    print('5. XỬ LÝ CÁC CỘT CÓ NHIỀU DỮ LIỆU THIẾU')
    columns_to_drop = ['project_name', 'floor_count', 'house_depth', 'road_width', 'house_direction', 'balcony_direction']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=columns_to_drop)
    print('\nCác cột đã loại bỏ:')
    for col in columns_to_drop:
        print('-', col)
    print('\nCác cột được giữ lại để xử lý dữ liệu thiếu:')
    keep_columns = ['ward_name', 'street_name', 'bedroom_count', 'bathroom_count', 'frontage_width']
    for col in keep_columns:
        if col in df.columns:
            print('-', col)
    return df

def handle_missing_data(df):
    print('6. XỬ LÝ DỮ LIỆU THIẾU')
    df = df.copy()
    numeric_columns = ['frontage_width', 'bedroom_count', 'bathroom_count']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'frontage_width' in df.columns:
        median_value = df['frontage_width'].median()
        df['frontage_width'] = df['frontage_width'].fillna(median_value)
        print('frontage_width: đã giữ lại và điền thiếu bằng median =', median_value)
    if 'bedroom_count' in df.columns:
        median_value = df['bedroom_count'].median()
        df['bedroom_count'] = df['bedroom_count'].fillna(median_value)
        print('bedroom_count: đã giữ lại và điền thiếu bằng median =', median_value)
    if 'bathroom_count' in df.columns:
        median_value = df['bathroom_count'].median()
        df['bathroom_count'] = df['bathroom_count'].fillna(median_value)
        print('bathroom_count: đã giữ lại và điền thiếu bằng median =', median_value)
    if 'ward_name' in df.columns:
        df['ward_name'] = df['ward_name'].fillna('Không rõ')
        print("ward_name: đã giữ lại và thay dữ liệu thiếu bằng 'Không rõ'")
    if 'street_name' in df.columns:
        df['street_name'] = df['street_name'].fillna('Không rõ')
        print("street_name: đã giữ lại và thay dữ liệu thiếu bằng 'Không rõ'")
    return df

def create_price_features(df):
    print('7. TẠO CÁC BIẾN GIÁ')
    df = df.copy()
    df['price_original'] = df['price']
    df['price_vnd'] = df['price']
    df['price_per_m2'] = df['price_vnd'] / df['area']
    df['price_per_m2'] = df['price_per_m2'].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['price_per_m2'])
    print('Đã tạo:')
    print('- price_original')
    print('- price_vnd')
    print('- price_per_m2')
    return df

def create_time_features(df):
    print('8. XỬ LÝ DỮ LIỆU THỜI GIAN')
    df = df.copy()
    df = df.sort_values(by='published_at').reset_index(drop=True)
    df['year'] = df['published_at'].dt.year
    df['month'] = df['published_at'].dt.month
    df['quarter'] = df['published_at'].dt.quarter
    df['day_of_week'] = df['published_at'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    print('\nĐã tạo:')
    print('- year')
    print('- month')
    print('- quarter')
    print('- day_of_week')
    print('- is_weekend')
    print('\nSỐ LƯỢNG TIN ĐĂNG THEO THÁNG:')
    monthly_count = df['month'].value_counts().sort_index()
    print(monthly_count)
    return df

def reorder_columns(df):
    print('9. SẮP XẾP CÁC CỘT')
    preferred_columns = ['name', 'description', 'property_type_name', 'province_name', 'district_name', 'ward_name', 'street_name', 'price_original', 'price_vnd', 'area', 'price_per_m2', 'frontage_width', 'bedroom_count', 'bathroom_count', 'published_at', 'year', 'month', 'quarter', 'day_of_week', 'is_weekend']
    existing_columns = [col for col in preferred_columns if col in df.columns]
    remaining_columns = [col for col in df.columns if col not in existing_columns]
    df = df[existing_columns + remaining_columns]
    print('\nThứ tự các cột sau khi sắp xếp:')
    print(df.columns.tolist())
    return df

def check_after_cleaning(df):
    print('10. KIỂM TRA SAU KHI LÀM SẠCH')
    print('\nKÍCH THƯỚC DỮ LIỆU:')
    print(df.shape)
    print('\nSỐ LƯỢNG DỮ LIỆU THIẾU:')
    print(df.isnull().sum())
    missing_percent = (df.isnull().sum() / len(df) * 100).round(2)
    print('\nTỶ LỆ DỮ LIỆU THIẾU (%):')
    print(missing_percent)
    print('\nSỐ DÒNG TRÙNG:')
    print(df.duplicated().sum())
    print('\nTHỐNG KÊ GIÁ BẤT ĐỘNG SẢN - VND:')
    price_stats = df['price_vnd'].describe()
    for name, value in price_stats.items():
        if name == 'count':
            print(f'{name:>6}: {value:,.0f}')
        else:
            print(f'{name:>6}: {value:,.0f}')
    print('\nTHỐNG KÊ DIỆN TÍCH - m²:')
    area_stats = df['area'].describe()
    for name, value in area_stats.items():
        print(f'{name:>6}: {value:,.2f}')
    print('\nTHỐNG KÊ GIÁ/M² - VND/m²:')
    price_m2_stats = df['price_per_m2'].describe()
    for name, value in price_m2_stats.items():
        if name == 'count':
            print(f'{name:>6}: {value:,.0f}')
        else:
            print(f'{name:>6}: {value:,.0f}')
    print('\nKIỂM TRA 5 TRƯỜNG ĐƯỢC GIỮ LẠI:')
    keep_columns = ['ward_name', 'street_name', 'bedroom_count', 'bathroom_count', 'frontage_width']
    for col in keep_columns:
        if col in df.columns:
            missing = df[col].isnull().sum()
            print(f'- {col}: còn {missing} giá trị thiếu')
    print('\n10 DÒNG DỮ LIỆU SAU KHI LÀM SẠCH:')
    display_columns = ['name', 'district_name', 'ward_name', 'street_name', 'price_vnd', 'area', 'frontage_width', 'bedroom_count', 'bathroom_count', 'price_per_m2', 'published_at']
    display_columns = [col for col in display_columns if col in df.columns]
    display_data = df[display_columns].head(10).copy()
    if 'price_vnd' in display_data.columns:
        display_data['price_vnd'] = display_data['price_vnd'].map(lambda x: f'{x:,.0f}')
    if 'area' in display_data.columns:
        display_data['area'] = display_data['area'].map(lambda x: f'{x:,.2f}')
    if 'frontage_width' in display_data.columns:
        display_data['frontage_width'] = display_data['frontage_width'].map(lambda x: f'{x:,.2f}')
    if 'price_per_m2' in display_data.columns:
        display_data['price_per_m2'] = display_data['price_per_m2'].map(lambda x: f'{x:,.0f}')
    print(display_data)

def save_data(df):
    df.to_csv(CLEANED_DATA_FILE, index=False, encoding='utf-8-sig')
    print('11. LƯU DỮ LIỆU')
    print('\nĐã tạo file dữ liệu sạch:')
    print(CLEANED_DATA_FILE)
if __name__ == '__main__':
    df = load_data()
    check_data(df)
    df = filter_data(df)
    df = remove_duplicates(df)
    df = remove_high_missing_columns(df)
    df = handle_missing_data(df)
    df = create_price_features(df)
    df = create_time_features(df)
    df = reorder_columns(df)
    check_after_cleaning(df)
    save_data(df)
    print('HOÀN THÀNH PART 1')
