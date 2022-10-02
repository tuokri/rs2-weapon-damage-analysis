import dash
import dash_bootstrap_components as dbc
import plotly.express as px
from dash import Input
from dash import Output
from dash import callback
from dash import dcc
from dash import html

dash.register_page(
    __name__,
    path="/charts",
)

fig = px.line(x=[0, 1], y=[1, 2])

fig.update_layout()

weapon_selector = html.Div(
    [
        dcc.Dropdown(
            id="weapon-selector",
        ),
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
            id="selected-weapons",
        ),
    ]
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
    class_name="mt-3",
)


@callback(
    Output("selected-weapons", "children"),
    Input("weapon-selector", "value"),
)
def add_weapon() -> dbc.AccordionItem:
    return dbc.AccordionItem(
        "test content",
        title="test title",
        id="test id",
    )
