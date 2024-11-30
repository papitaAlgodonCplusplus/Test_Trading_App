
from analyzers.indicators import get_conditions
import traceback
def define_actions(data, probabilities, threshold=0.5, analysis_level=0, zigzag=False, context_window=None):
    """Define trading actions based on indicators."""
    
    data_backup = data.copy()
    probabilities_backup = probabilities.copy()
    if context_window is not None:
        try:
            data_backup = data_backup.iloc[:context_window]
            probabilities_backup = probabilities[:context_window]
        except:
            traceback.print_exc()
            print("Error occurred while defining actions.")

    bullish, bearish = get_conditions(data_backup, analysis_level)

    data_backup['Action'] = 'Hold'
    data_backup.loc[bullish, 'Action'] = 'Buy'
    data_backup.loc[bearish, 'Action'] = 'Sell'

    if threshold > 0.0:
        data_backup.loc[(data_backup['Action'] == 'Hold') & (probabilities_backup > 1 - threshold), 'Action'] = 'Buy'
        data_backup.loc[(data_backup['Action'] == 'Hold') & (probabilities_backup < threshold), 'Action'] = 'Sell'

    if zigzag:
        data_backup['Action'] = data_backup['Action'].mask(data_backup['Action'] == data_backup['Action'].shift(), 'Hold')
    
    try:
        if 'Action' not in data.columns or data['Action'].empty:
            data['Action'] = 'Hold'
        data.loc[data_backup.index, 'Action'] = data_backup['Action']
    except:
        traceback.print_exc()
        print("Error occurred while defining actions.")
    
    return data