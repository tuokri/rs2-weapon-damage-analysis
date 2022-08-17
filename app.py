import json
import os
from typing import Dict
from typing import List

import dash
import dash_bootstrap_components as dbc
import plotly.express as px
from dash import dcc
from dash import html
from dash_bootstrap_templates import load_figure_template

load_figure_template("vapor")


def load_external_scripts(var: str) -> List[Dict]:
    scripts = os.environ.get(var)
    if not scripts:
        return []
    return json.loads(scripts)


external_scripts = load_external_scripts("EXTERNAL_SCRIPTS")

app = dash.Dash(
    __name__,
    external_scripts=external_scripts,
    external_stylesheets=[dbc.themes.VAPOR],
    title="rs2sim",
)
server = app.server

app.css.config.serve_locally = False
app.scripts.config.serve_locally = False

fig = px.line(x=[0, 1], y=[1, 2])

app.layout = html.Div(
    children=[
        html.H1(
            children="Rising Storm 2: Vietnam Weapon Simulator",
        ),
        html.Div(
            children="Work in progress."
        ),
        dcc.Graph(
            id="test-graph",
            figure=fig
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(
        debug=True,
        threaded=True,
        dev_tools_hot_reload=True
    )
