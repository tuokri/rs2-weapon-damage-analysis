import dash
from dash import callback
from dash import dcc
from dash import html
from dash.dependencies import Input
from dash.dependencies import Output

dash.register_page(__name__, path="/statistics")

# layout = dbc.Container()
layout = html.Div([
    dcc.Location(id='url'),
    html.Div(id='layout-div'),
    html.Div(id='content'),
    html.Div(id='output'),
])


# @callback(Output('content', 'children'), Input('url', 'pathname'))
# def display_page(pathname):
#     return html.Div([
#         dcc.Input(id='input', value='hello world'),
#         html.Div(id='output')
#     ])
#
#
# @callback(Output('output', 'children'), Input('input', 'value'))
# def update_output(value):
#     print('>>> update_output')
#     return value
#
#
# @callback(Output('layout-div', 'children'), Input('input', 'value'))
# def update_layout_div(value):
#     print('>>> update_layout_div')
#     return value
