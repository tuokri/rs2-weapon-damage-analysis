import dash
import dash_bootstrap_components as dbc
import plotly.express as px
from dash import dcc
from dash import html

dash.register_page(
    __name__,
    path="/charts",
)

fig = px.line(x=[0, 1], y=[1, 2])

fig.update_layout()

weapon_selector = html.Div(
    dbc.Accordion(
        [
            dbc.AccordionItem(
                "This is the content of the first section",
                title="Item 1",
            ),
            dbc.AccordionItem(
                "This is the content of the second section",
                title="Item 2",
            ),
            dbc.AccordionItem(
                "This is the content of the third section",
                title="Item 3",
            ),
        ],
        start_collapsed=True,
        always_open=True,
        class_name="my-2",
    ),
)

layout = dbc.Container(
    [
        weapon_selector,

        dcc.Graph(
            id="test-graph",
            figure=fig,
            className="mb-5",
        ),
    ],
)
