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


external_scripts = []
external_scripts.extend(load_external_scripts("EXTERNAL_SCRIPTS"))

app = dash.Dash(
    __name__,
    external_scripts=external_scripts,
    external_stylesheets=[dbc.themes.VAPOR],
    title="rs2sim",
)
server = app.server

app.css.config.serve_locally = False
app.scripts.config.serve_locally = False

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        <!-- Google Tag Manager -->
        <script>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
        new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
        j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
        'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
        })(window,document,'script','dataLayer','GTM-N97VFK2');</script>
        <!-- End Google Tag Manager -->
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        <!-- Google Tag Manager (noscript) -->
        <noscript><iframe src="https://www.googletagmanager.com/ns.html?id=GTM-N97VFK2"
        height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>
        <!-- End Google Tag Manager (noscript) -->
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

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
