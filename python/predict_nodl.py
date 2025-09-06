import numpy as np
import pandas as pd
import pickle
from keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体
try:
    # 尝试使用STHeiti Medium字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    # 如果STHeiti不可用，尝试安装字体
    try:
        import os
        if not os.path.exists('/System/Library/Fonts/STHeiti Medium.ttc'):
            # 安装STHeiti字体
            os.system('cp /System/Library/Fonts/Supplemental/STHeiti\ Medium.ttc /System/Library/Fonts/')
        plt.rcParams['font.sans-serif'] = ['STHeiti Medium']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        print("警告：无法安装STHeiti字体，图表可能无法正确显示中文")

def load_trained_model():
    """加载训练好的模型和scaler"""
    model = load_model('models/nodl_model.h5')
    with open('models/nodl_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def predict_future(model, scaler, last_data, days=30):
    """预测未来涨跌幅"""
    predictions = []
    current_seq = last_data.copy()
    
    
    for _ in range(days):
        # 准备输入数据 (需要符合LSTM输入形状)
        input_data = current_seq.reshape(1, current_seq.shape[0], current_seq.shape[1])
        pred = model.predict(input_data)  # Keep as decimal (0.01 = 1%)
        print(f"Model output shape: {pred.shape}")
        print(f"Model output: {pred}")
        
        # 保存预测结果 (直接取标量值)
        predictions.append(pred[0][0])
        
        # 更新序列
        current_seq = np.roll(current_seq, -1, axis=0)
        current_seq[-1, -1] = pred[0][0]  # 用预测值更新涨跌幅
        
    return np.array(predictions)

def inverse_transform(predictions, scaler, last_data):
    """将预测结果反归一化"""
    # 创建假数据用于反归一化
    dummy = np.zeros((len(predictions), last_data.shape[1]))
    dummy[:, 0] = predictions  # 只使用收盘价列
    
    # 反归一化
    return scaler.inverse_transform(dummy)[:, 0]

def save_results(predictions, dates):
    """保存预测结果"""
    print(f"Predictions length: {len(predictions)}")
    print(f"Dates length: {len(dates)}")
    print(f"Predictions: {predictions}")
    print(f"Dates: {dates}")
    
    df = pd.DataFrame({
        '日期': dates,
        '预测涨跌幅': predictions * 100  # Convert to percentage
    })
    df.to_csv('predict_predictions.csv', index=False)
    print("预测结果已保存到 predict_predictions.csv")

def on_zoom(event):
    """动态调整网格密度"""
    ax = event.inaxes
    if ax is None:
        return
    
    # 根据x轴范围调整网格密度
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    
    # 设置主要网格间隔
    if x_range <= 7:  # 小于等于7天
        ax.xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=1))
    elif x_range <= 30:  # 小于等于30天
        ax.xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=2))
    else:  # 大于30天
        ax.xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=5))
    
    # 设置次要网格间隔
    ax.xaxis.set_minor_locator(plt.matplotlib.dates.HourLocator(interval=12))
    
    # 重绘图表
    plt.draw()

def plot_results(predictions, dates, historical_changes=None):
    """绘制预测涨跌幅"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 如果有历史数据，绘制实际涨跌幅
    if historical_changes is not None:
        hist_dates = pd.date_range(end=dates[0] - pd.Timedelta(days=1), periods=len(historical_changes))
        ax.plot(hist_dates, historical_changes, 'go-', label='实际涨跌幅')
    
    # 直接绘制预测涨跌幅
    line, = ax.plot(dates, predictions*100, 'bo-', label='预测涨跌幅')
    
    # 在每个点上添加涨跌幅标签
    for i, (date, change) in enumerate(zip(dates, predictions)):
        ax.text(date, change*100, f'{change*100:.2f}%', 
                fontsize=8, color='blue',
                ha='center', va='bottom')
    
    ax.set_title('未来30天涨跌幅预测')
    ax.set_xlabel('日期')
    ax.set_ylabel('涨跌幅 (%)')
    
    # 设置x轴日期格式
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=1))  # 每天一个主刻度
    ax.xaxis.set_minor_locator(plt.matplotlib.dates.HourLocator(interval=6))  # 每6小时一个次刻度
    plt.xticks(rotation=45)
    
    # 自动调整y轴范围
    all_changes = np.concatenate([predictions, historical_changes]) if historical_changes is not None else predictions
    y_min = np.min(all_changes)
    y_max = np.max(all_changes)
    y_range = y_max - y_min
    
    # 设置y轴范围，上下各留10%的padding
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
    
    # 设置y轴刻度间隔
    y_interval = max(0.1, round(y_range / 20, 1))  # 更精细的刻度间隔
    ax.yaxis.set_major_locator(plt.MultipleLocator(y_interval))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(y_interval/2))  # 添加次刻度
    
    # 设置网格线
    ax.grid(True, which='major', linestyle='-', linewidth=0.5)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.5)
    
    # 网格配置
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.5)
    
    # 添加缩放事件监听
    fig.canvas.mpl_connect('button_release_event', on_zoom)
    
    plt.legend()
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.2)
    plt.savefig('predict_predictions.png', dpi=300)
    plt.close()
    print("预测图表已保存为 predict_predictions.png")

def main():
    # 加载模型
    model, scaler = load_trained_model()
    
    # 加载历史数据
    df = pd.read_csv('python/nodl_historical_data.csv')
    
    # 预处理最后30天的数据
    features = ['收盘', '开盘', '高', '低', '交易量', '涨跌幅']
    last_data = df[features].tail(30).copy()
    
    # 处理交易量（去掉K/M单位并转换为数值）
    last_data['交易量'] = last_data['交易量'].str.replace('K', '').str.replace('M', '')
    last_data['交易量'] = np.where(
        last_data['交易量'].str.contains('K'), 
        last_data['交易量'].astype(float) * 1e3,
        last_data['交易量'].astype(float) * 1e6
    )
    
    # 处理涨跌幅（去掉%并转换为数值）
    last_data['涨跌幅'] = last_data['涨跌幅'].str.replace('%', '').astype(float) / 100
    
    # 使用特征名称进行转换
    last_data_scaled = scaler.transform(pd.DataFrame(last_data.values, columns=scaler.feature_names))
    
    # 预测未来30天
    predictions_scaled = predict_future(model, scaler, last_data_scaled)
    
    # 反归一化预测结果
    predictions = inverse_transform(predictions_scaled, scaler, last_data)
    
    # 生成日期 (从历史数据的最后日期开始)
    last_date = pd.to_datetime(df['日期'].iloc[0])
    dates = pd.date_range(last_date, periods=len(predictions))
    
    # 获取历史涨跌幅并转换为float
    historical_changes = df['涨跌幅'].tail(30).str.replace('%', '').astype(float).values / 100
    
    # 保存和可视化结果
    save_results(predictions, dates)
    plot_results(predictions, dates, historical_changes)

if __name__ == '__main__':
    main()
