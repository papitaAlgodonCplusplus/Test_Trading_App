import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
from datetime import datetime, timedelta
from queue_manager import data_queue

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[
                "https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css"])
app.title = "Real-Time Trading Dashboard"

# Retrieve all available data from the queue
dates = []
prices = []
actions = []
data_points = []
current_portfolio_value = None
monthly_portfolio_value = None
total_gains = None
total_losses = None
monthly_gains = []
monthly_losses = []
vertical_lines = []
month_annotations = []

# Layout
app.layout = html.Div(
    style={
        "backgroundColor": "#000000",  # Pure black background
        "backgroundImage": "url('https://i.imgur.com/N996dMz.gif')",
        "backgroundSize": "cover",
        "color": "white",
        "fontFamily": "Orbitron, sans-serif",
        "minHeight": "100vh",
        "padding": "20px",
    },
    children=[
        html.H1(
            "Real-Time Trading Performance",
            className="text-center my-3",
            style={
                "color": "#00FF00",  # Neon green for title
                "textShadow": "0px 0px 10px #00FF00",
                "marginBottom": "30px",
            },
        ),
        dcc.Graph(id="trading-graph"),
        dcc.Graph(id="monthly-table"),  # Add a new graph for the table
        dcc.Interval(id="update-interval", interval=1000, n_intervals=0),
        dcc.Store(id="monthly-data-store",
                  data={"month_starts": [], "monthly_gains": {}, "monthly_losses": {}}),
    ],
)

processed_month_starts = set()

def obtain_data():
    global monthly_gains, monthly_losses

    while not data_queue.empty():
        data_points.append(data_queue.get())

    if not data_points:
        # Return empty/default response
        return {
            "x_vals": [],
            "y_vals": [],
            "hold_signals": [],
            "buy_signals": [],
            "sell_signals": [],
            "month_starts": [],
            "portfolio_info": {
                "initial_portfolio_value": 0,
                "current_portfolio_value": 0,
                "monthly_portfolio_value": 0,
                "total_gains": 0,
                "total_losses": 0,
                "monthly_gains": 0,
                "monthly_losses": 0,
            },
        }

    for dp in data_points:
        if dp.get("date") and dp.get("price") is not None:
            dates.append(dp.get("date"))
            prices.append(dp.get("price", 0))
            actions.append(dp.get("action", 0))

            # Retrieve portfolio information if available
            portfolio_info = dp.get("portfolio_info")
            if portfolio_info:
                initial_portfolio_value = portfolio_info.get(
                    "initial_portfolio_value", 0)
                current_portfolio_value = portfolio_info.get(
                    "current_portfolio_value", 0)
                monthly_portfolio_value = portfolio_info.get(
                    "monthly_portfolio_value", 0)
                total_gains = portfolio_info.get("total_gains", 0)
                total_losses = portfolio_info.get("total_losses", 0)
                monthly_gains.append(portfolio_info.get("monthly_gains", 0))
                monthly_losses.append(portfolio_info.get("monthly_losses", 0))

    # Sort the data by date
    sorted_data = sorted(zip(dates, prices, actions), key=lambda x: x[0])
    dates[:], prices[:], actions[:] = zip(*sorted_data)

    # Identify new month starts
    month_starts = []
    for i in range(1, len(dates)):
        if dates[i] and dates[i - 1] and dates[i].month != dates[i - 1].month:
            if dates[i] not in processed_month_starts:
                month_starts.append(dates[i])
                processed_month_starts.add(dates[i])

    # Identify signals based on actions
    hold_signals = [(dates[i], prices[i])
                    for i in range(len(dates)) if actions[i] == 0]
    buy_signals = [(dates[i], prices[i])
                   for i in range(len(dates)) if actions[i] == 1]
    sell_signals = [(dates[i], prices[i])
                    for i in range(len(dates)) if actions[i] == 2]

    return {
        "x_vals": dates,
        "y_vals": prices,
        "hold_signals": hold_signals,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "month_starts": month_starts,  # Only new month starts
        "portfolio_info": {
            "initial_portfolio_value": initial_portfolio_value,
            "current_portfolio_value": current_portfolio_value,
            "monthly_portfolio_value": monthly_portfolio_value,
            "total_gains": total_gains,
            "total_losses": total_losses,
            "monthly_gains": monthly_gains[-1] if monthly_gains else 0,
            "monthly_losses": monthly_losses[-1] if monthly_losses else 0,
        },
    }

@app.callback(
    [Output("trading-graph", "figure"), Output("monthly-table", "figure"), Output("monthly-data-store", "data")],
    [Input("update-interval", "n_intervals")],
    [State("monthly-data-store", "data")],
)
def update_dashboard_and_table(n_intervals, stored_data):
    updated_data = obtain_data()
    portfolio_info = updated_data["portfolio_info"]

    # Build Plotly figure for trading graph
    fig = go.Figure()

    # Stock price line in white
    fig.add_trace(
        go.Scatter(
            x=updated_data["x_vals"],
            y=updated_data["y_vals"],
            mode="lines+markers",
            name="Stock Price",
            line=dict(color="white", width=2),
            marker=dict(size=8, color="cyan",
                        line=dict(color="white", width=1)),
        )
    )

    # Signals with neon colors
    for signals, label, color in zip(
        [updated_data["hold_signals"], updated_data["buy_signals"],
            updated_data["sell_signals"]],
        ["Hold", "Buy", "Sell"],
        ["yellow", "lime", "red"],
    ):
        if signals:
            signal_dates, signal_prices = zip(*signals)
            fig.add_trace(
                go.Scatter(
                    x=signal_dates,
                    y=signal_prices,
                    mode="markers",
                    name=label,
                    marker=dict(
                        color=color,
                        size=12,
                        opacity=0.8,
                        line=dict(color="white", width=2),
                    ),
                )
            )

    # Add new vertical lines and annotations for new month starts
    for month_start in updated_data["month_starts"]:
        if month_start not in [line["x0"] for line in vertical_lines]:
            # Determine the line color based on portfolio info
            line_color = (
                "lime" if portfolio_info["monthly_gains"] > portfolio_info["monthly_losses"] else "red"
            )

            # Create the line as a shape
            new_line = dict(
                type="line",
                x0=month_start,
                x1=month_start,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color=line_color, width=3, dash="dash"),
            )
            vertical_lines.append(new_line)

            # Create an annotation for the month
            new_annotation = dict(
                x=month_start,
                y=1.05,
                xref="x",
                yref="paper",
                text=f"<b>{month_start.strftime('%B')}</b>",
                showarrow=False,
                font=dict(size=12, color=line_color),
                bgcolor="rgba(0, 0, 0, 0.8)",
                bordercolor=line_color,
                borderwidth=1,
            )
            month_annotations.append(new_annotation)

    # Apply all vertical lines to the figure
    fig.update_layout(
        shapes=vertical_lines,
        paper_bgcolor="rgba(0, 0, 0, 0)",  # Transparent to show animated background
        plot_bgcolor="rgba(0, 0, 0, 0)",  # Transparent to show animated background
    )

    # Apply all annotations to the figure
    for annotation in month_annotations:
        fig.add_annotation(annotation)

    # Add the cyan box with portfolio details in the bottom-right corner
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1.0,  # Align to the right edge
        y=0.0,  # Align to the bottom edge
        text=(
            f"<b>Portfolio</b><br>"
            f"Original: ${portfolio_info['initial_portfolio_value']:.2f}<br>"
            f"Current: ${portfolio_info['current_portfolio_value']:.2f}<br>"
            f"Monthly: ${portfolio_info['monthly_portfolio_value']:.2f}<br>"
            f"Profit: ${portfolio_info['total_gains']:.2f}<br>"
            f"Loss: ${portfolio_info['total_losses']:.2f}"
        ),
        showarrow=False,
        align="right",
        font=dict(color="cyan", size=12, family="Courier New"),
        bgcolor="rgba(0, 0, 0, 0.7)",  # Adjusted alpha for semi-transparency
        bordercolor="cyan",
        borderwidth=2,
    )

    fig.update_layout(
        xaxis=dict(title="Date", showgrid=False, zeroline=False),
        yaxis=dict(title="Value ($)", showgrid=False, zeroline=False),
        template="plotly_dark",
        # Transparent to show animated background
        paper_bgcolor="rgba(0, 0, 0, 0)",
        # Transparent to show animated background
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font=dict(color="white", size=14),
        # Increased bottom margin for the table
        margin=dict(t=50, b=150, l=50, r=50),
    )

    # Retrieve stored data
    existing_month_starts = stored_data.get("month_starts", [])
    existing_monthly_gains = stored_data.get("monthly_gains", {})
    existing_monthly_losses = stored_data.get("monthly_losses", {})

    # Sort the month starts for consistency
    existing_month_starts.sort()
    existing_month_starts = [
        datetime.strptime(month, '%Y-%m-%dT%H:%M:%S') if isinstance(month, str) else month
        for month in existing_month_starts
    ]

    # Ensure all new months are added to the store
    for month_start in updated_data["month_starts"]:
        month_key = month_start.strftime('%Y-%m')
        if month_key not in [m.strftime('%Y-%m') for m in existing_month_starts]:
            existing_month_starts.append(month_start)
            existing_monthly_gains[month_key] = portfolio_info["monthly_gains"]
            existing_monthly_losses[month_key] = portfolio_info["monthly_losses"]


    # Generate table data
    month_names = [month.strftime('%B %Y') for month in existing_month_starts]
    monthly_revenue = [existing_monthly_gains.get(
        month.strftime('%Y-%m'), 0) for month in existing_month_starts]
    monthly_losses = [existing_monthly_losses.get(
        month.strftime('%Y-%m'), 0) for month in existing_month_starts]

    monthly_profit = [revenue - loss for revenue, loss in zip(monthly_revenue, monthly_losses)]

    table_fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Month", "Revenue ($)", "Loss ($)", "Profit ($)"],
                    fill_color="black",
                    font=dict(color="lime", size=14),
                    align="center",
                ),
                cells=dict(
                    values=[
                        month_names,
                        [round(value, 2) for value in monthly_revenue],
                        [round(value, 2) for value in monthly_losses],
                        [round(value, 2) for value in monthly_profit],
                    ],
                    fill_color="black",
                    font=dict(color="white", size=12),
                    align="center",
                )
            )
        ]
    )

    table_fig.update_layout(
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(t=20, b=20, l=20, r=20),
    )

    # Return updated figures and stored data
    return fig, table_fig, {
        "month_starts": [month.strftime('%Y-%m-%dT%H:%M:%S') for month in existing_month_starts],
        "monthly_gains": existing_monthly_gains,
        "monthly_losses": existing_monthly_losses,
    }


if __name__ == "__main__":
    app.run_server(debug=True)