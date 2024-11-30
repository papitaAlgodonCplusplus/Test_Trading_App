
def run_simulation(data, initial_capital=10000, reverse=False, risk_amount=None, expected_profit=None, log=False):
    """Run a trading simulation."""
    capital, units_hold, profit_vault = initial_capital, 0, 0
    profit_over_time = []
    vault_values = []
    data.loc[:, 'Action'] = data['Action'].mask(data['Action'] == data['Action'].shift(), 'Hold')

    for i, row in data.iterrows():
        action, price = row['Action'], row['Close']
        if reverse:
            action = 'Buy' if action == 'Sell' else 'Sell' if action == 'Buy' else 'Hold'

        if action == 'Buy' and capital > 0:
            invest = min(risk_amount or capital, capital)
            units_hold = invest / price
            capital -= invest
            data.loc[i, 'Action'] = 'Buy'
        elif action == 'Sell' and units_hold > 0:
            capital += units_hold * price
            units_hold = 0
            data.loc[i, 'Action'] = 'Sell'
        else:
            data.loc[i, 'Action'] = 'Hold'

        profit_over_time.append(capital - initial_capital)
        if log:
           print(f"Date: {row['Date']}, Capital: {capital:.2f}, Units: {units_hold:.2f}, Profit: {profit_over_time[-1]:.2f}, Action: {action}, Price: {price:.2f}")

        if profit_over_time[-1] > 0:
            profit_vault += profit_over_time[-1]
            capital = initial_capital

        vault_values.append(profit_vault)    

        if expected_profit and profit_vault >= expected_profit:
            capital += units_hold * price
            units_hold = 0
            return data, capital, profit_vault

    if units_hold > 0:
        capital += units_hold * data['Close'].iloc[-1]
        profit_over_time[-1] = capital - initial_capital

    data.loc[:, 'Profit'] = profit_over_time
    data.loc[:, 'Vault'] = vault_values
    return data, capital, profit_vault
