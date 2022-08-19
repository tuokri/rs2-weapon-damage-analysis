import dash
import dash_bootstrap_components as dbc
import plotly.express as px
from dash import dcc
from dash import html

dash.register_page(__name__, path="/")

fig = px.line(x=[0, 1], y=[1, 2])

layout = dbc.Container([
    # html.H1(
    #     children="Rising Storm 2: Vietnam Weapon Simulator",
    # ),
    html.Div(
        children="Work in progress.",
    ),
    dcc.Graph(
        id="test-graph",
        figure=fig,
    ),
])
