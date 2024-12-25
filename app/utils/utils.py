import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error



def metrics(test, prediction)->pd.DataFrame:

    print('def metrics')

    mae = mean_absolute_error(test, prediction)
    mse = mean_squared_error(test, prediction)
    r2 = r2_score(test, prediction)
    mape = mean_absolute_percentage_error(test, prediction) * 100

    data = {
            'eval':['MAE','MSE','R2', 'MAPE'], 
            'score':[round(mae, 2), round(mse, 2), round(r2, 2), round(mape, 2)]
            }
    
    results = pd.DataFrame(data)
    return results, mae, mse, r2, mape


def plot_TSMixer_results(predicted_prices_df, date, actual, predicted):

    print('plot TSMixer')

    plt.figure(figsize=(12, 6))
    plt.plot(predicted_prices_df[date], predicted_prices_df[actual], label='Actual Close Price')
    plt.plot(predicted_prices_df[date], predicted_prices_df[predicted], label='Predicted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Actual vs Predicted Close Prices')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()