# src/lpbf/viz/dashboard.py
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from lpbf.state import SimulationState
import torch

# Minimal scaffold
def create_app():
    app = dash.Dash(__name__)
    
    app.layout = html.Div([
        html.H1("LPBF Thermal Simulator Dashboard"),
        html.Div([
            html.B("Status: "), "Ready"
        ]),
        html.Hr(),
        html.Div([
            dcc.Graph(id='heatmap-graph'),
        ]),
        html.Div([
            html.Label("Time Step:"),
            dcc.Slider(
                id='time-slider',
                min=0, max=100, step=1, value=0,
                marks={0: '0', 100: '100'}
            )
        ])
    ])
    
    # Callback placeholders (needs data source)
    @app.callback(
        Output('heatmap-graph', 'figure'),
        Input('time-slider', 'value')
    )
    def update_graph(step_idx):
        # TODO: Load real data from artifacts
        fig = go.Figure()
        fig.update_layout(title=f"Placeholder Step {step_idx}")
        return fig
        
    return app

if __name__ == '__main__':
    app = create_app()
    app.run_server(debug=True)
