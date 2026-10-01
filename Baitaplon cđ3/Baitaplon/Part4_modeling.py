import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import CLEANED_DATA_FILE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from pathlib import Path
import hashlib
import json
import platform
import shutil
import joblib
import sklearn
from matplotlib.ticker import FuncFormatter, NullFormatter
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.inspection import permutation_importance
BASE = Path(__file__).resolve().parent
FEATURES = ['area', 'frontage_width', 'bedroom_count', 'bathroom_count']
LABELS = ['Diện tích', 'Mặt tiền', 'Phòng ngủ', 'Phòng tắm']
SEED = 42
pd.set_option('display.float_format', lambda value: f'{value:,.4f}')
np.set_printoptions(suppress=True)

def thong_bao(so, ten):
    print(f'\nBƯỚC {so}/9 — {ten}', flush=True)

def metrics(y, prediction):
    return {'MAE_VND': float(mean_absolute_error(y, prediction)), 'RMSE_VND': float(np.sqrt(mean_squared_error(y, prediction))), 'MAPE_percent': float(mean_absolute_percentage_error(y, prediction) * 100), 'R2': float(r2_score(y, prediction))}

def savefig(fig, path):
    formatter = FuncFormatter(lambda value, position: f'{value:,.4f}'.rstrip('0').rstrip('.'))
    for ax in fig.axes:
        if ax.get_title().startswith('Feature Importance'):
            ax.xaxis.set_major_formatter(formatter)
        else:
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)

def buoc_4_tien_xu_ly(X, y):
    thong_bao(4, 'Kiểm tra 4 biến số trong dữ liệu sạch')
    if not np.isfinite(X.to_numpy(dtype=float)).all() or not np.isfinite(y.to_numpy()).all():
        raise ValueError('X/y còn thiếu hoặc vô cực. Cần thống nhất xử lý ở phần chung của nhóm.')
    if (y <= 0).any() or (X['area'] <= 0).any():
        raise ValueError('Giá và diện tích phải dương.')
    print('Không mã hóa quận/loại nhà; không điền thiếu hoặc lọc riêng thêm ở Random Forest.')
    return 'passthrough'

def buoc_5_huan_luyen(preprocessor, X_train, y_train):
    thong_bao(5, 'Huấn luyện Random Forest và chọn cấu hình')
    pipe = Pipeline([('preprocess', preprocessor), ('model', RandomForestRegressor(n_estimators=150, criterion='squared_error', bootstrap=True, max_features=0.8, random_state=SEED, n_jobs=2))])
    grid = {'model__max_depth': [None, 12, 20], 'model__min_samples_leaf': [1, 3]}
    search = GridSearchCV(pipe, grid, scoring='neg_mean_absolute_error', cv=TimeSeriesSplit(n_splits=3), n_jobs=1, refit=True, error_score='raise')
    print('Đang chọn cấu hình bằng MAE trên kiểm định chéo 3 folds...', flush=True)
    search.fit(X_train, y_train)
    model = search.best_estimator_
    print('Cấu hình tốt nhất trong 6 cấu hình thử:', search.best_params_)
    print(f'MAE CV: {-search.best_score_ / 1000000000.0:.3f} tỷ đồng.')
    return (search, model)

def buoc_6_danh_gia(model, search, X_train, X_test, y_train, y_test, raw, df, data_path):
    thong_bao(6, 'Dự báo và đánh giá mô hình')
    pred = model.predict(X_test)
    assert np.isfinite(pred).all() and len(pred) == len(y_test)
    cv = pd.DataFrame(search.cv_results_)
    cv['mean_MAE_VND'] = -cv['mean_test_score']
    baseline = DummyRegressor(strategy='median').fit(np.zeros((len(y_train), 1)), y_train)
    base_pred = baseline.predict(np.zeros((len(y_test), 1)))
    report = {'source_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(), 'cleaned_rows': len(raw), 'eligible_rows': len(df), 'train_rows': len(y_train), 'test_rows': len(y_test), 'date_min': str(df.published_at.min()), 'date_max': str(df.published_at.max()), 'random_state': SEED, 'features': FEATURES, 'selection': 'Minimum 3-fold TimeSeriesSplit CV MAE on training split only', 'best_cv_MAE_VND': float(-search.best_score_), 'best_params': model.named_steps['model'].get_params(), 'train_metrics': metrics(y_train, model.predict(X_train)), 'test_metrics': metrics(y_test, pred), 'median_baseline_test': metrics(y_test, base_pred), 'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__}}
    report['selected_grid_params'] = search.best_params_
    print(pd.DataFrame({'Train': report['train_metrics'], 'Test': report['test_metrics'], 'Mốc trung vị (test)': report['median_baseline_test']}).to_string())
    print('MAE/RMSE: đồng; MAPE: phần trăm; R² không phải độ chính xác.')
    return (pred, report, cv)

def buoc_7_phan_tich_sai_so(df, test_ids, y_test, pred, charts):
    thong_bao(7, 'Vẽ biểu đồ và phân tích sai số')
    detail = df.iloc[test_ids][['sorted_row', 'name', 'published_at'] + FEATURES].copy()
    detail['actual_vnd'] = y_test.to_numpy()
    detail['predicted_vnd'] = pred
    detail['residual_vnd'] = detail.actual_vnd - pred
    detail['absolute_error_vnd'] = detail.residual_vnd.abs()
    detail['absolute_percentage_error'] = detail.absolute_error_vnd / detail.actual_vnd * 100
    detail['price_band'] = pd.cut(detail.actual_vnd / 1000000000.0, bins=[0, 5, 10, 20, 100, np.inf], labels=['<5 tỷ', '5–<10 tỷ', '10–<20 tỷ', '20–<100 tỷ', '>=100 tỷ'], right=False)
    groups = detail.groupby('price_band', observed=True).agg(count=('actual_vnd', 'size'), MAE_VND=('absolute_error_vnd', 'mean'), MAPE_percent=('absolute_percentage_error', 'mean'))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    actual = y_test.to_numpy() / 1000000000.0
    predicted = pred / 1000000000.0
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax in axes:
        ax.scatter(actual, predicted, s=20, alpha=0.55, color='#227A91')
        low, high = (min(actual.min(), predicted.min()), max(actual.max(), predicted.max()))
        ax.plot([low, high], [low, high], '--', color='#B54A43', label='Dự báo = thực tế')
        ax.set_xlabel('Giá thực tế (tỷ đồng)')
        ax.set_ylabel('Giá dự báo (tỷ đồng)')
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_title('Thang tuyến tính — toàn bộ tập kiểm tra')
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_title('Thang log — toàn bộ tập kiểm tra')
    fig.suptitle(f'Random Forest: Actual vs Predicted | n = {len(test_ids)}')
    savefig(fig, charts / '01_actual_vs_predicted.png')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(predicted, detail.residual_vnd / 1000000000.0, alpha=0.55, s=22, color='#227A91')
    ax.axhline(0, color='#B54A43', linestyle='--')
    ax.set_xscale('log')
    ax.set(xlabel='Giá dự báo (tỷ đồng, thang log)', ylabel='Thực tế − dự báo (tỷ đồng)', title='Sai số trên tập kiểm tra — dương nghĩa là dự báo thấp')
    ax.grid(alpha=0.2)
    savefig(fig, charts / '02_residuals.png')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(detail.absolute_percentage_error, bins=35, color='#227A91', edgecolor='white')
    ax.set_yscale('log')
    ax.set(xlabel='Sai số phần trăm tuyệt đối (%)', ylabel='Số tin (thang log)', title='Phân bố sai số phần trăm — không loại giá trị lớn')
    savefig(fig, charts / '03_percentage_errors.png')
    print('Đã lưu 3 hình: Actual vs Predicted, phần dư và sai số phần trăm.')
    display_groups = groups.rename(columns={'count': 'Số tin', 'MAPE_percent': 'MAPE (%)'}).copy()
    display_groups['Sai số MAE (tỷ đồng)'] = display_groups.pop('MAE_VND') / 1000000000
    display_groups.index.name = 'Nhóm giá thực tế'
    print('Sai số theo nhóm giá THỰC TẾ; MAE là sai số, không phải giá nhà:')
    print(display_groups[['Số tin', 'Sai số MAE (tỷ đồng)', 'MAPE (%)']].to_string())
    return (detail, groups)

def buoc_8_feature_importance(model, X_test, y_test, charts):
    thong_bao(8, 'Phân tích mức độ quan trọng của đặc trưng')
    print('Đang phân tích mức độ quan trọng của đặc trưng...', flush=True)
    imp = permutation_importance(model, X_test, y_test, scoring='neg_mean_absolute_error', n_repeats=8, random_state=SEED, n_jobs=1)
    importance = pd.DataFrame({'feature': FEATURES, 'label': LABELS, 'MAE_increase_VND': imp.importances_mean, 'std_VND': imp.importances_std})
    ordered = importance.sort_values('MAE_increase_VND')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(ordered.label, ordered.MAE_increase_VND / 1000000000.0, xerr=ordered.std_VND / 1000000000.0, color='#227A91', capsize=4)
    ax.axvline(0, color='gray', linewidth=1)
    ax.set(xlabel='Mức tăng MAE khi hoán vị (tỷ đồng)', title='Feature Importance trên tập kiểm tra\nThanh sai số: độ lệch chuẩn qua 8 lần hoán vị')
    savefig(fig, charts / '04_feature_importance.png')
    print(importance.sort_values('MAE_increase_VND', ascending=False).to_string(index=False))
    return importance

def buoc_9_luu_ket_qua(out, model, report, cv, manifest, detail, groups, importance, X_test, pred):
    thong_bao(9, 'Lưu kết quả và nhận xét')
    manifest.to_csv(out / 'train_test_split.csv', index=False, encoding='utf-8-sig', float_format='%.10f')
    cv.to_csv(out / 'cv_results.csv', index=False, encoding='utf-8-sig', float_format='%.10f')
    (out / 'model_metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    detail.to_csv(out / 'test_predictions.csv', index=False, encoding='utf-8-sig', float_format='%.10f')
    detail.nlargest(20, 'absolute_error_vnd').to_csv(out / 'top20_errors.csv', index=False, encoding='utf-8-sig', float_format='%.10f')
    detail.nlargest(20, 'absolute_percentage_error').to_csv(out / 'top20_percentage_errors.csv', index=False, encoding='utf-8-sig', float_format='%.10f')
    groups.to_csv(out / 'errors_by_price_band.csv', encoding='utf-8-sig', float_format='%.10f')
    importance.to_csv(out / 'feature_importance.csv', index=False, encoding='utf-8-sig', float_format='%.10f')
    joblib.dump(model, out / 'random_forest_pipeline.joblib', compress=3)
    reloaded = joblib.load(out / 'random_forest_pipeline.joblib')
    np.testing.assert_allclose(pred, reloaded.predict(X_test))
    m = report['test_metrics']
    b = report['median_baseline_test']
    summary = f"KẾT QUẢ RANDOM FOREST\nDữ liệu hợp lệ: {report['eligible_rows']}; train: {report['train_rows']}; test: {report['test_rows']}.\nMAE kiểm định chéo theo thời gian tốt nhất: {report['best_cv_MAE_VND'] / 1000000000.0:.3f} tỷ đồng.\nCấu hình được chọn: {report['selected_grid_params']}; 150 cây, max_features=0.8, bootstrap=True, squared_error.\nTập kiểm tra: MAE = {m['MAE_VND'] / 1000000000.0:.3f} tỷ; RMSE = {m['RMSE_VND'] / 1000000000.0:.3f} tỷ; MAPE = {m['MAPE_percent']:.2f}%; R² = {m['R2']:.4f}.\nMốc dự báo trung vị từ train: MAE = {b['MAE_VND'] / 1000000000.0:.3f} tỷ; RMSE = {b['RMSE_VND'] / 1000000000.0:.3f} tỷ.\nMAPE có thể rất lớn khi giá thực tế nhỏ; không diễn giải 100% − MAPE thành độ chính xác.\nRMSE phạt mạnh các sai số lớn. Xem top20_errors.csv và errors_by_price_band.csv để phân tích.\nFeature Importance là mức tăng MAE khi xáo trộn từng biến trên test, không phải quan hệ nhân quả hay phần trăm ảnh hưởng tới giá. Các biến tương quan có thể chia sẻ thông tin. Không dùng kết quả test này để chọn lại cấu hình.\nTrain/test cùng nằm trong tháng 6/2025; đánh giá đoạn cuối sau đoạn đầu, chưa kiểm chứng khả năng dự báo các tháng sau hay giá giao dịch thực tế. Không tự xóa giá ngoại lai để làm đẹp kết quả.\nDùng nguyên CSV sạch của nhóm. Median đã được tính trước khi chia tập: đây là hạn chế rò rỉ tiền xử lý chung của cả hai mô hình. Muốn khắc phục cần sửa phần chuẩn bị chung cho cả hai, không chỉ Random Forest.\n"
    linear = report['linear_regression_test']
    summary += f"\nLinear Regression trên cùng test: MAE = {linear['MAE_VND'] / 1000000000.0:.3f} tỷ; RMSE = {linear['RMSE_VND'] / 1000000000.0:.3f} tỷ; MAPE = {linear['MAPE_percent']:.2f}%; R² = {linear['R2']:.4f}.\n"
    summary += 'Xem model_comparison.csv; không sử dụng các chỉ số của bản chia ngẫu nhiên cũ để so sánh.\n'
    (out / 'Nhan_xet_ket_qua.txt').write_text(summary, encoding='utf-8')
    print(summary, flush=True)
    print(f'Hoàn thành Part 4: Linear Regression và Random Forest. Kết quả: {out.resolve()}')

def main():
    IMAGE_FOLDER = BASE / 'Anh'
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    data_path = Path(CLEANED_DATA_FILE)
    if not data_path.is_absolute():
        data_path = BASE / data_path
    raw = pd.read_csv(data_path)
    df = raw.copy()
    out = BASE / 'results_modeling'
    charts = out / 'charts'
    charts.mkdir(parents=True, exist_ok=True)
    df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
    df = df.dropna(subset=['published_at'])
    df = df.sort_values('published_at')
    df = df.set_index('published_at')
    print('PART 4 - XÂY DỰNG MÔ HÌNH')
    print('Kích thước dữ liệu:', df.shape)
    features = ['area', 'frontage_width', 'bedroom_count', 'bathroom_count']
    target = 'price_vnd'
    for col in features + [target]:
        if col not in df.columns:
            raise ValueError(f'Không tìm thấy cột: {col}')
    X = df[features].copy()
    y = df[target].copy()
    print('KIỂM TRA DỮ LIỆU')
    print('\nGiá trị thiếu:')
    print(X.isnull().sum())
    if X.isnull().sum().sum() > 0:
        raise ValueError('X vẫn còn dữ liệu thiếu. Hãy xử lý dữ liệu thiếu ở Part 1 trước.')
    if y.isnull().sum() > 0:
        raise ValueError('price_vnd vẫn còn dữ liệu thiếu.')
    if not np.isfinite(X).all().all():
        raise ValueError('X có giá trị không hợp lệ.')
    if not np.isfinite(y).all():
        raise ValueError('price_vnd có giá trị không hợp lệ.')
    if (y <= 0).any():
        raise ValueError('price_vnd phải lớn hơn 0.')
    if (X['area'] <= 0).any():
        raise ValueError('area phải lớn hơn 0.')
    split = int(len(df) * 0.8)
    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]
    print('CHIA DỮ LIỆU TRAIN / TEST')
    print('Train:', len(X_train))
    print('Test :', len(X_test))
    print('LINEAR REGRESSION')
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_pred = linear_model.predict(X_test)
    linear_mae = mean_absolute_error(y_test, linear_pred)
    linear_rmse = np.sqrt(mean_squared_error(y_test, linear_pred))
    linear_mape = mean_absolute_percentage_error(y_test, linear_pred) * 100
    linear_r2 = r2_score(y_test, linear_pred)
    print('RANDOM FOREST')
    search, random_forest_model = buoc_5_huan_luyen('passthrough', X_train, y_train)
    rf_pred = random_forest_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_mape = mean_absolute_percentage_error(y_test, rf_pred) * 100
    rf_r2 = r2_score(y_test, rf_pred)
    comparison = pd.DataFrame({'Model': ['Linear Regression', 'Random Forest'], 'MAE_VND': [linear_mae, rf_mae], 'RMSE_VND': [linear_rmse, rf_rmse], 'MAPE_percent': [linear_mape, rf_mape], 'R2': [linear_r2, rf_r2]})
    print('KẾT QUẢ SO SÁNH')
    comparison_display = comparison.copy()
    comparison_display['MAE_VND'] = comparison_display['MAE_VND'].apply(lambda x: f'{x:,.0f}')
    comparison_display['RMSE_VND'] = comparison_display['RMSE_VND'].apply(lambda x: f'{x:,.0f}')
    comparison_display['MAPE_percent'] = comparison_display['MAPE_percent'].apply(lambda x: f'{x:.2f}')
    comparison_display['R2'] = comparison_display['R2'].apply(lambda x: f'{x:.4f}')
    print(comparison_display.to_string(index=False))
    prediction_result = pd.DataFrame({'published_at': X_test.index, 'actual_price_vnd': y_test.values, 'linear_regression_prediction': linear_pred, 'random_forest_prediction': rf_pred})
    prediction_result['actual_price_vnd'] = prediction_result['actual_price_vnd'].round(0)
    prediction_result['linear_regression_prediction'] = prediction_result['linear_regression_prediction'].round(0)
    prediction_result['random_forest_prediction'] = prediction_result['random_forest_prediction'].round(0)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test / 1000000000.0, rf_pred / 1000000000.0, alpha=0.5)
    plt.xlabel('Giá thực tế (tỷ đồng)')
    plt.ylabel('Giá dự đoán (tỷ đồng)')
    plt.title('Giá thực tế và giá dự đoán - Random Forest')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_FOLDER, 'part4_thuc_te_du_doan.png'), dpi=300)
    plt.close()
    n = min(100, len(y_test))
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values[:n] / 1000000000.0, label='Giá thực tế')
    plt.plot(linear_pred[:n] / 1000000000.0, label='Linear Regression')
    plt.plot(rf_pred[:n] / 1000000000.0, label='Random Forest')
    plt.xlabel('Thứ tự mẫu')
    plt.ylabel('Giá (tỷ đồng)')
    plt.title('So sánh giá thực tế và giá dự đoán')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_FOLDER, 'part4_so_sanh_du_doan.png'), dpi=300)
    plt.close()
    plt.figure(figsize=(7, 5))
    plt.bar(comparison['Model'], comparison['MAPE_percent'])
    plt.xlabel('Mô hình')
    plt.ylabel('MAPE (%)')
    plt.title('So sánh MAPE giữa hai mô hình')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_FOLDER, 'part4_mape.png'), dpi=300)
    plt.close()
    plt.figure(figsize=(7, 5))
    plt.barh(comparison['Model'], comparison['R2'])
    plt.xlabel('R2')
    plt.ylabel('Mô hình')
    plt.title('So sánh R2 giữa hai mô hình')
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_FOLDER, 'part4_r2.png'), dpi=300)
    plt.close()
    comparison.to_csv(BASE / 'model_comparison.csv', index=False, encoding='utf-8-sig', float_format='%.4f')
    prediction_result.to_csv(BASE / 'model_predictions.csv', index=False, encoding='utf-8-sig', float_format='%.0f')
    df_result = df.reset_index().copy()
    df_result['sorted_row'] = np.arange(len(df_result))
    if 'name' not in df_result:
        df_result['name'] = ''
    test_ids = np.arange(split, len(df_result))
    _, report, cv = buoc_6_danh_gia(random_forest_model, search, X_train, X_test, y_train, y_test, raw, df_result, data_path)
    report['linear_regression_test'] = metrics(y_test, linear_pred)
    report['linear_regression_train'] = metrics(y_train, linear_model.predict(X_train))
    report['split_method'] = 'Shared chronological 80/20 split for both models'
    report['preprocessing_limitation'] = 'CSV median imputation occurred before splitting; affects both models.'
    report['time_ranges'] = {'train': [str(X_train.index.min()), str(X_train.index.max())], 'test': [str(X_test.index.min()), str(X_test.index.max())]}
    manifest = df_result[['sorted_row', 'published_at']].copy()
    manifest['split'] = np.where(manifest['sorted_row'] < split, 'train', 'test')
    detail, groups = buoc_7_phan_tich_sai_so(df_result, test_ids, y_test, rf_pred, charts)
    detail['linear_predicted_vnd'] = linear_pred
    importance = buoc_8_feature_importance(random_forest_model, X_test, y_test, charts)
    comparison.to_csv(out / 'model_comparison.csv', index=False, encoding='utf-8-sig', float_format='%.10f')
    joblib.dump(linear_model, out / 'linear_regression.joblib', compress=3)
    np.testing.assert_allclose(joblib.load(out / 'linear_regression.joblib').predict(X_test), linear_pred)
    buoc_9_luu_ket_qua(out, random_forest_model, report, cv, manifest, detail, groups, importance, X_test, rf_pred)
    for picture in charts.glob('*.png'):
        shutil.copy2(picture, IMAGE_FOLDER / picture.name)
    print('Đã lưu 4 biểu đồ chung và 4 biểu đồ phân tích Random Forest vào:', IMAGE_FOLDER)
    print('Bảng kết quả chi tiết và mô hình:', out)
if __name__ == '__main__':
    main()
