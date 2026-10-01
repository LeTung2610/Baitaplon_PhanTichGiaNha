import pandas as pd
from pathlib import Path
pd.set_option('display.float_format', lambda value: f'{value:,.4f}')
comparison = pd.read_csv(Path(__file__).resolve().parent / 'model_comparison.csv')
print('PART 5 - INSIGHTS')
print('\nKẾT QUẢ MÔ HÌNH:')
print(comparison.to_string(index=False))
linear = comparison[comparison['Model'] == 'Linear Regression'].iloc[0]
rf = comparison[comparison['Model'] == 'Random Forest'].iloc[0]
print('INSIGHTS CHÍNH')
if rf['MAE_VND'] < linear['MAE_VND']:
    print('\n1. Random Forest có MAE thấp hơn Linear Regression.')
else:
    print('\n1. Linear Regression có MAE thấp hơn Random Forest.')
if rf['RMSE_VND'] < linear['RMSE_VND']:
    print('2. Random Forest có RMSE thấp hơn Linear Regression.')
else:
    print('2. Linear Regression có RMSE thấp hơn Random Forest.')
if rf['MAPE_percent'] < linear['MAPE_percent']:
    print('3. Random Forest có MAPE thấp hơn Linear Regression.')
else:
    print('3. Linear Regression có MAPE thấp hơn Random Forest.')
if rf['R2'] > linear['R2']:
    print('4. Random Forest có R² cao hơn Linear Regression.')
else:
    print('4. Linear Regression có R² cao hơn Random Forest.')
print('\n5. Giá bất động sản chịu ảnh hưởng bởi nhiều yếu tố.')
print('   Mô hình hiện tại mới sử dụng diện tích, mặt tiền,')
print('   số phòng ngủ và số phòng tắm.')
print('\n6. Kết quả chỉ phản ánh dữ liệu bất động sản tại Hà Nội')
print('   trong khoảng thời gian thực tế có trong dữ liệu; xem model_metrics.json.')
print('\n7. Có thể cải thiện mô hình bằng cách bổ sung thêm')
print('   các yếu tố về vị trí, loại bất động sản và đặc điểm nhà.')
print('HOÀN THÀNH PART 5')
