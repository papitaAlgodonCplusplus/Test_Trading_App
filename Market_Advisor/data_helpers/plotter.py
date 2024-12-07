from dash import Dash, dcc, html, Output, Input
import plotly.graph_objects as go
import plotly.subplots as sp
import threading
import logging
import numpy as np
logging.getLogger('werkzeug').setLevel(logging.WARNING)

class Plotter:
    def __init__(self):
        self.app = Dash(__name__)
        self.figure = go.Figure()
        self.predictions = None
        self.title_suffix = ""
        self.data_backtest = None

        # Define layout with Interval component
        self.app.layout = html.Div([
            html.H1(f"Trading Results and Predictions ({self.title_suffix})"),
            dcc.Graph(id='trading-graph', figure=self.figure),
            dcc.Interval(id='interval-update', interval=1000, n_intervals=0)  # Interval in milliseconds
        ])

        # Register callback
        self.app.callback(
            Output('trading-graph', 'figure'),
            Input('interval-update', 'n_intervals')  # Triggers callback every interval
        )(self.update_figure)

    def update_figure(self, n_intervals):
        if self.data_backtest is not None:
            # Create subplots: 3 rows, 1 column
            fig = sp.make_subplots(
                rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                subplot_titles=(
                    f"Price Action with Trading Signals ({self.title_suffix})",
                    "Profit and Vault Over Time",
                    "Predicted Trends (Growth/Fall)"
                )
            )

            # Subplot 1: Price Action and Trading Signals
            fig.add_trace(go.Scatter(
                x=self.data_backtest['Date'],
                y=self.data_backtest['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color='blue', width=2),
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=self.data_backtest['Date'],
                y=self.data_backtest['Take Profit Long'],
                mode='lines',
                name='Take Profit Long',
                line=dict(color='green', width=1, dash='dot'),
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=self.data_backtest['Date'],
                y=self.data_backtest['Stop Loss Long'],
                mode='lines',
                name='Stop Loss Long',
                line=dict(color='red', width=1, dash='dot'),
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=self.data_backtest['Date'],
                y=self.data_backtest['Take Profit Short'],
                mode='lines',
                name='Take Profit Short',
                line=dict(color='cyan', width=1, dash='dot'),
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=self.data_backtest['Date'],
                y=self.data_backtest['Stop Loss Short'],
                mode='lines',
                name='Stop Loss Short',
                line=dict(color='orange', width=1, dash='dot'),
            ), row=1, col=1)
            
            action_colors = {'Buy': 'red', 'Sell': 'green', 'Sell Short': 'purple', 'Buy to Cover': 'orange'}
            action_labels = ['Buy Long', 'Sell Long', 'Sell Short', 'Buy to Cover']
            for action, color in action_colors.items():
                fig.add_trace(go.Scatter(
                    x=self.data_backtest['Date'][self.data_backtest['Action'] == action],
                    y=self.data_backtest['Close'][self.data_backtest['Action'] == action],
                    mode='markers',
                    name=action_labels[list(action_colors.keys()).index(action)],
                    marker=dict(color=color, size=10),
                ), row=1, col=1)

            # Subplot 2: Profit and Vault
            fig.add_trace(go.Scatter(
                x=self.data_backtest['Date'],
                y=self.data_backtest['Profit'],
                mode='lines',
                name='Revenue',
                line=dict(color='green', width=2)
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x=self.data_backtest['Date'],
                y=self.data_backtest['Vault'],
                mode='lines',
                name='Vault',
                line=dict(color='yellow', width=2)
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x=self.data_backtest['Date'],
                y=self.data_backtest['Risk/Reward Ratios'],
                mode='lines',
                name='Risk/Reward Ratios',
                line=dict(color='purple', width=2)
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x=self.data_backtest['Date'],
                y=self.data_backtest['Profit Summatory'],
                mode='lines',
                name='Profit Summatory',
                line=dict(color='red', width=2)
            ), row=2, col=1)

            fig.add_trace(go.Scatter(
                x=self.data_backtest['Date'],
                y=self.data_backtest['Win/Loss Ratio'],
                mode='lines',
                name='Win/Loss Ratio',
                line=dict(color='orange', width=2)
            ), row=2, col=1)
                
            # Subplot 3: Predicted Trends
            if self.predictions is not None:
                self.predictions = np.pad(self.predictions, (0, len(self.data_backtest) - len(self.predictions)), mode='constant', constant_values=0)
                fig.add_trace(go.Scatter(
                    x=self.data_backtest['Date'],
                    y=self.data_backtest['Close'],
                    mode='lines',
                    name='Close Price (Predictions)',
                    line=dict(color='gray', width=1, dash='dot'),
                ), row=3, col=1)

                fig.add_trace(go.Scatter(
                    x=self.data_backtest['Date'][self.predictions == 0],
                    y=self.data_backtest['Close'][self.predictions == 0],
                    mode='markers',
                    name='Predicted Growth',
                    marker=dict(color='purple', size=8),
                ), row=3, col=1)

                fig.add_trace(go.Scatter(
                    x=self.data_backtest['Date'][self.predictions == 1],
                    y=self.data_backtest['Close'][self.predictions == 1],
                    mode='markers',
                    name='Predicted Fall',
                    marker=dict(color='orange', size=8),
                ), row=3, col=1)

                # Update layout
                fig.update_layout(
                    height=1000,
                    title=f"Trading Results and Predictions ({self.title_suffix})",
                    xaxis=dict(title="Date"),
                    yaxis=dict(title="Close Price (Signals)"),
                    yaxis2=dict(title="Profit and Vault"),
                    yaxis3=dict(title="Close Price (Predictions)"),
                    legend_title="Actions",
                    template="plotly_white",
                    hovermode="x unified"
                )

            return fig
        return self.figure

    def start(self):
        thread = threading.Thread(target=self.app.run_server, kwargs={'port': 9000, 'debug': False, 'use_reloader': False})
        thread.daemon = True
        thread.start()
