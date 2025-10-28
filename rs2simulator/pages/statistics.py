# Copyright (C) 2022-2025 Tuomo Kriikkula
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import dash
from dash import dcc
from dash import html

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
