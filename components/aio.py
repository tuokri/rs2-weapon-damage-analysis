from dash import Input
from dash import MATCH
from dash import Output
from dash import clientside_callback
from dash_bootstrap_templates import ThemeChangerAIO

from utils import read_asset_text

theme_changer_js = read_asset_text("theme-changer.js")


class ThemeChangerAIOCustom(ThemeChangerAIO):
    clientside_callback(
        theme_changer_js,
        Output(ThemeChangerAIO.ids.dummy_div(MATCH), "style"),
        Input(ThemeChangerAIO.ids.button(MATCH), "n_clicks"),
    )
