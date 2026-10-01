import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
from config import CLEANED_DATA_FILE
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Anh')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
data_path = CLEANED_DATA_FILE if os.path.isabs(CLEANED_DATA_FILE) else os.path.join(os.path.dirname(os.path.abspath(__file__)), CLEANED_DATA_FILE)
df = pd.read_csv(data_path)
district_price = df.groupby('district_name')['price_vnd'].mean().sort_values(ascending=False)
plt.figure(figsize=(12, 7))
bars = plt.bar(district_price.index, district_price / 1000000000)
plt.title('Giá bất động sản trung bình theo quận/huyện tại Hà Nội')
plt.xlabel('Quận/Huyện')
plt.ylabel('Giá trung bình (tỷ đồng)')
plt.xticks(rotation=45, ha='right')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.1f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'bieu_do_1_gia_trung_binh_theo_quan_huyen.png'), dpi=300, bbox_inches='tight')
price_histogram = df.loc[df['price_vnd'] <= 100000000000, 'price_vnd'] / 1000000000
plt.figure(figsize=(12, 7))
counts, bins, patches = plt.hist(price_histogram, bins=20)
plt.title('Phân bố giá bất động sản tại Hà Nội (0 - 100 tỷ đồng)')
plt.xlabel('Giá bất động sản (tỷ đồng)')
plt.ylabel('Số lượng bất động sản')
for count, patch in zip(counts, patches):
    if count > 0:
        plt.text(patch.get_x() + patch.get_width() / 2, count, f'{int(count)}', ha='center', va='bottom', fontsize=8)
plt.xticks(range(0, 101, 5), rotation=45)
plt.xlim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'bieu_do_2_phan_bo_gia_bat_dong_san.png'), dpi=300, bbox_inches='tight')
r = df['area'].corr(df['price_vnd'])
plt.figure(figsize=(11, 7))
plt.scatter(df['area'], df['price_vnd'] / 1000000000, s=23, alpha=0.4)
plt.title(f'Diện tích và giá bất động sản tại Hà Nội\nTháng 6/2025 | n = {len(df):,} | Pearson r = {r:.3f}')
plt.xlabel('Diện tích (m²)')
plt.ylabel('Giá bất động sản (tỷ đồng)')
plt.xlim(0, 600)
plt.ylim(0, 250)
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'bieu_do_3_dien_tich_va_gia.png'), dpi=300, bbox_inches='tight')
order = df.groupby('property_type_name')['price_vnd'].median().sort_values().index
values = [df.loc[df['property_type_name'] == name, 'price_vnd'] / 1000000000.0 for name in order]
labels = [f'{name}\n(n = {len(data):,})' for name, data in zip(order, values)]
fig, ax = plt.subplots(figsize=(12, 7))
bp = ax.boxplot(values, orientation='horizontal', patch_artist=True, widths=0.55, medianprops={'linewidth': 2}, flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.4})
for patch in bp['boxes']:
    patch.set_facecolor('#8ECAD0')
ax.set_yticks(range(1, len(labels) + 1))
ax.set_yticklabels(labels)
ax.set_xlabel('Giá bất động sản (tỷ đồng)')
ax.set_title('Phân bố giá theo loại bất động sản\nHà Nội, tháng 6/2025 | Sắp xếp theo trung vị', pad=15)
ax.grid(axis='x', alpha=0.2)
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_FOLDER, 'bieu_do_4_gia_theo_loai_bat_dong_san.png'), dpi=300, bbox_inches='tight')
columns = ['price_vnd', 'area', 'frontage_width', 'bedroom_count', 'bathroom_count']
labels = ['Giá', 'Diện tích', 'Mặt tiền', 'Phòng ngủ', 'Phòng tắm']
corr = df[columns].corr(method='pearson')
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30, ha='right')
ax.set_yticklabels(labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        value = corr.iloc[i, j]
        ax.text(j, i, f'{value:.2f}', ha='center', va='center', color='white' if abs(value) > 0.6 else '#172B3A', fontsize=12)
fig.colorbar(im, ax=ax, shrink=0.82, label='Hệ số tương quan Pearson')
ax.set_title(f'Tương quan giữa các biến số\nHà Nội, tháng 6/2025 | n = {len(df):,}', pad=15)
fig.text(0.5, 0.025, 'Mặt tiền, phòng ngủ và phòng tắm đã được điền thiếu bằng trung vị.\nTương quan không thể hiện quan hệ nhân quả.', ha='center', fontsize=10)
fig.tight_layout(rect=(0, 0.07, 1, 1))
plt.savefig(os.path.join(OUTPUT_FOLDER, 'bieu_do_5_ma_tran_tuong_quan.png'), dpi=300, bbox_inches='tight')
