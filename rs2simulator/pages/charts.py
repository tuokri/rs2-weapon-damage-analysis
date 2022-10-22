from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
from aio import template_from_url
from dash import ALL
from dash import Input
from dash import Output
from dash import callback
from dash import ctx
from dash import dcc
from dash import html
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

from components.aio import ThemeChangerAIOCustom
from rs2simulator import db

dash.register_page(
    __name__,
    path="/charts",
)


def get_weapon_selector_elements() -> Iterable[Dict]:
    elements = []
    weapons = db.api.get_weapons()

    for wep in weapons:
        wep_name = wep.name
        short_name = wep.short_display_name

        # TODO: fine tune drop down elements' vertical alignment.
        elements.append({
            "label": dbc.Container(
                [
                    html.P(
                        wep_name,
                        # className="",
                    ),
                    html.P(
                        short_name,
                        className="d-none d-md-block",
                        style={
                            "margin-right": "1.5em",
                        },
                    ),
                ],
                class_name="d-flex justify-content-between align-items-center",
                fluid=True,
            ),
            "value": wep_name,
            "search": short_name,
        })
    return elements


SELECT_WEP_PLACEHOLDER = "Select weapon..."
weapon_selector = html.Div(
    [
        dcc.Dropdown(
            get_weapon_selector_elements(),
            id="weapon-selector",
            placeholder=SELECT_WEP_PLACEHOLDER,
            clearable=False,
        ),
        dbc.Accordion(
            [],
            start_collapsed=True,
            always_open=True,
            class_name="my-2",
            id="selected-weapons",
        ),
    ],
    className="my-4",
)

placeholder_fig = px.line(
    # x=[0, 1],
    # y=[1, 2],
    template=template_from_url(dbc.themes.VAPOR),
)
layout = dbc.Container(
    [
        dcc.Graph(
            id="graph",
            figure=placeholder_fig,
        ),

        weapon_selector,
    ],
    class_name="my-3",
)


@callback(
    [
        Output("selected-weapons", "children"),
        Output("weapon-selector", "value"),
        Output("selected-weapons", "active_item"),
    ],
    [
        Input("weapon-selector", "value"),
        Input("selected-weapons", "children"),
        Input("selected-weapons", "active_item"),
        Input(
            component_id={
                "type": "dynamic-weapon-remove-button",
                "weapon": ALL,
                "index": ALL,
            },
            component_property="n_clicks",
        ),
    ]
)
def modify_selected_weapons(
        value: str,
        children: List[Any],
        active_items: List[Any],
        *_,
) -> Tuple[List[dbc.AccordionItem], str, Any]:
    ret = children
    if not value:
        raise PreventUpdate
        # return ret, SELECT_WEP_PLACEHOLDER, active_items

    triggered_id = ctx.triggered_id

    if triggered_id == "weapon-selector":
        index = len(children)
        ret = children + [
            dbc.AccordionItem(
                [
                    html.Div(
                        [
                            html.Div([
                                html.P(f"this is the content for {value}"),
                                html.P("asd"),
                            ]),
                            html.Div(
                                dbc.Button(
                                    "Remove",
                                    class_name="btn-danger",
                                    id={
                                        "type": "dynamic-weapon-remove-button",
                                        "weapon": value,
                                        "index": index,
                                    },
                                ),
                            ),
                        ],
                        className="d-flex justify-content-between",
                    ),
                ],
                title=value,
                id={
                    "type": "dynamic-weapon-accordion-item",
                    "weapon": value,
                    "index": index,
                },
                item_id=str(index),
            ),
        ]
    elif triggered_id == "selected-weapons":
        raise PreventUpdate
    else:
        to_del = None
        del_index = None
        try:
            trig_wep = triggered_id["weapon"]
            trig_idx = triggered_id["index"]
            for i, r in enumerate(ret):
                props = r["props"]
                r_id = props["id"]
                r_wep = r_id["weapon"]
                r_idx = r_id["index"]
                if r_wep == trig_wep and r_idx == trig_idx:
                    to_del = i
                    del_index = r_idx
                    break
        except KeyError as e:
            print(type(e).__name__, e)

        if (to_del is not None) and (del_index is not None):
            ret.pop(to_del)
            active_items.remove(str(del_index))

    return ret, SELECT_WEP_PLACEHOLDER, active_items


@callback(
    Output("graph", "figure"),
    Input("selected-weapons", "children"),
    Input(ThemeChangerAIOCustom.ids.radio("theme-changer"), "value"),
)
def update_graph_theme(weapons: List[dict], theme: str) -> go.Figure:
    if not weapons:
        return placeholder_fig

    print(ctx.triggered_id)

    fig = make_subplots(rows=1, cols=1)
    # alo_x = np.array([0, 1])
    # alo_y = np.array([0, 1])
    print(len(weapons))
    for weapon in weapons:
        name = weapon["props"]["id"]["weapon"]
        wep = db.api.get_weapon(name)
        # noinspection PyTypeChecker
        alo: db.models.AmmoLoadout = wep.ammo_loadouts[0]
        # for alo in wep.ammo_loadouts:
        print(f"plotting: {name}")
        x, y = alo.bullet.dmg_falloff_np_tuple()

        sim = db.api.get_weapon_sim(
            weapon_name=name,
            bullet_name=alo.bullet.name,
            angle=0,
        )
        print(len(sim))

        fig.add_scatter(
            x=sim["distance"],
            y=sim["damage"],
            name=wep.short_display_name,
            row=1,
            col=1,
        )

    fig.update_layout(template=template_from_url(theme))
    return fig
