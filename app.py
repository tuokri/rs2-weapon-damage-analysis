import json
import os
from typing import Dict
from typing import List

import dash
import plotly.express as px
from dash import dcc
from dash import html


def load_external_scripts(var: str) -> List[Dict]:
    scripts = os.environ.get(var)
    if not scripts:
        return []
    return json.loads(scripts)


external_scripts = load_external_scripts("EXTERNAL_SCRIPTS")

app = dash.Dash(
    __name__,
    external_scripts=external_scripts,
    title="rs2sim",
)
server = app.server

fig = px.line(x=[0, 1], y=[1, 2])

app.layout = html.Div(
    className="container",
    children=[
        html.H1(
            children="Rising Storm 2: Vietnam Weapon Simulator",
        ),
        html.Div(
            className="container",
            children="Work in progress."
        ),
        dcc.Graph(
            className="container",
            id="test-graph",
            figure=fig
        ),
    ]
)

if __name__ == "__main__":
    app.run_server(debug=False)
