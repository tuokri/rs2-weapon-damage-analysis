import dash
import dash_bootstrap_components as dbc

from dash import html

dash.register_page(__name__, path="/privacypolicy")

layout = dbc.Container(
    [
        html.H2(children="TODO: PRIVACY POLICY"),

        html.Div(children="""
        TODO CONTENT.
    """),
    ]
)
