import pandas as pd
import matplotlib.pyplot as plt

# Загружаем данные
forecast = pd.read_csv("forecast_61days.csv", sep=';', parse_dates=['Date'])
expanded = pd.read_csv("expanded.csv", sep=';', parse_dates=['Date'])

# Фильтруем по нужному артикулу
article_code = 'TALTHA-BP0026'
forecast_item = forecast[forecast['Article'] == article_code]
expanded_item = expanded[expanded['Article'] == article_code]

# Сумма по департаментам
sum_by_dept = forecast_item.groupby('Department')['Predicted'].sum()
total_sum = forecast_item['Predicted'].sum()
print("Сумма продаж по департаментам:")
print(sum_by_dept)
print("\nОбщая сумма продаж за месяц:", total_sum)

# --- График ---
for dept in forecast_item['Department'].unique():
    plt.figure(figsize=(50, 8))  # шире график

    # Исторические данные из expanded.csv
    hist = expanded_item[expanded_item['Department'] == dept].copy()
    hist['Sold'] = hist['Sold'].fillna(0)
    plt.plot(hist['Date'], hist['Sold'], label='Historical Sold', color='green', marker='s', markersize=4)

    # Actual из forecast_30days.csv
    fc = forecast_item[forecast_item['Department'] == dept].copy()
    plt.plot(fc['Date'], fc['Actual'], label='Actual', color='blue', marker='x', markersize=4)

    # Predicted из forecast_30days.csv
    plt.plot(fc['Date'], fc['Predicted'], label='Predicted', color='orange', marker='o', markersize=4)

    plt.title(f'Продажи и прогноз - {dept}', fontsize=16)
    plt.xlabel('Дата', fontsize=14)
    plt.ylabel('Количество', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(f'forecast_vs_actual_historical_wide_{dept.replace(" ", "_")}.png')
    plt.close()
