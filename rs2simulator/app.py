import os
from typing import Optional
from urllib.parse import urlparse
from urllib.parse import urlunparse

import dash
import dash_bootstrap_components as dbc
from dash import Input
from dash import Output
from dash import State
from dash import html
from dash_bootstrap_templates import load_figure_template
from flask import redirect
from flask import request
from werkzeug import Response

from rs2simulator.components.aio import ThemeChangerAIOCustom
from rs2simulator.utils import read_asset_text

gtag_manager_string = """
<!-- Google Tag Manager -->
<script>(function(w,d,s,l,i){w[l]=w[l]||[];w[l].push({'gtm.start':
new Date().getTime(),event:'gtm.js'});var f=d.getElementsByTagName(s)[0],
j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
})(window,document,'script','dataLayer','GTM-N97VFK2');</script>
<!-- End Google Tag Manager -->
"""

gtag_manager_noscript_string = """
<!-- Google Tag Manager (noscript) -->
<noscript><iframe src="https://www.googletagmanager.com/ns.html?id=GTM-N97VFK2"
height="0" width="0" style="display:none;visibility:hidden"></iframe></noscript>
<!-- End Google Tag Manager (noscript) -->
"""

index_string = """
<!DOCTYPE html>
<html>
    <head>
        <script>window['gtag_enable_tcf_support'] = true;</script>
        {gtag_manager}
        {metas}
        <title>{title}</title>
        {favicon}
        {css}
    </head>
    <body class="site">
        {gtag_manager_noscript}
        <main class="site-content">
            {app_entry}
        </main>
        {footer}
    </body>
</html>
"""

footer_string = """
<footer>
    {config}
    {scripts}
    {renderer}
    {footer_content}
</footer>
"""


class CustomDash(dash.Dash):
    def interpolate_index(self, **kwargs) -> str:
        footer_content = read_asset_text("footer.html")
        footer = footer_string.format(
            config=kwargs.pop("config"),
            scripts=kwargs.pop("scripts"),
            renderer=kwargs.pop("renderer"),
            footer_content=footer_content,
        )
        return index_string.format(
            # TODO: set these up later.
            gtag_manager="",
            gtag_manager_noscript="",
            footer=footer,
            **kwargs,
        )


load_figure_template("vapor")

external_stylesheets = [
    {
        "href": "https://cdn.jsdelivr.net/gh/orestbida/cookieconsent@v2.8.5/dist/cookieconsent.css",
        "rel": "stylesheet",
        "media": "print",
        "onload": "this.media='all'",
    },
    dbc.themes.VAPOR,
    "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.1/dbc.min.css",
]

external_scripts = [
    {
        # Load cookie consent plugin in async mode.
        # "async src": "https://cdn.jsdelivr.net/gh/orestbida/cookieconsent@v2.8.5/dist/cookieconsent.js",
    },
    {
        # TODO: main script here that inits cookieconsent and loads others on demand.
        # "src": "https://cdn.jsdelivr.net/gh/tuokri/solid-bassoon@master/plenty-leave.js"
    },
    {
        # Google Analytics. Can be loaded async, but should be enabled with
        # restricted functionality until consent is given.
        # "type": "text/plain",
        # "async src": "https://www.googletagmanager.com/gtag/js?id=G-3TNF8134RD",
        # "data-cookiecategory": "analytics",
    },
    {
        # Adsense. Should only be loaded after consent given or immediately, but
        # only with non-personalized ads.
        # "type": "text/plain",
        # "defer src": "https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-3291929185204781",
        # "crossorigin": "anonymous",
        # "data-cookiecategory": "analytics",
    },
]

app_extra_kwargs = {
    "serve_locally": False,
}
if __name__ == "__main__":
    app_extra_kwargs["serve_locally"] = True
    print("using app_extra_kwargs:", app_extra_kwargs)

app = CustomDash(
    __name__,
    external_scripts=external_scripts,
    external_stylesheets=external_stylesheets,
    title="rs2sim",
    use_pages=True,
    assets_folder=os.environ.get("ASSETS_FOLDER"),
    suppress_callback_exceptions=True,
    **app_extra_kwargs,
)
server = app.server


def upgrade_to_https(url: str) -> str:
    # noinspection HttpUrlsUsage
    return url.replace("http://", "https://", 1)


if "FLY_APP_NAME" in os.environ:
    @server.before_request
    def before_request_prod_env(*_) -> Optional[Response]:
        url = request.url
        if not request.is_secure:
            return redirect(
                location=upgrade_to_https(upgrade_to_https(url)),
                code=301,
            )
else:
    @server.before_request
    def before_request_dev_env(*_) -> Optional[Response]:
        url_parts = urlparse(request.url)
        if url_parts.path in ("", "/"):
            url_parts = url_parts._replace(path=charts_page["path"])
            return redirect(
                location=urlunparse(url_parts),
                code=301,
            )

theme_changer = ThemeChangerAIOCustom(
    aio_id="theme-changer",
    radio_props={
        "value": dbc.themes.VAPOR,
        "label_class_name": "badge",
        # "persistence": True, TODO: callbacks needed.
    },
    button_props={
        "class_name": "btn-dark my-md-0 my-2",
        "style": {
            "textTransform": "uppercase",
        },
    },
    offcanvas_props={
        "placement": "end",
        "title": "Select theme",
        "labelledby": "offcanvas-title-label",
        "style": {
            "width": "235px",
            "textTransform": "uppercase",
        },
    },
)

charts_page = dash.page_registry["pages.charts"]
statistics_page = dash.page_registry["pages.statistics"]

nav_items = [
    dbc.Nav(
        [
            dbc.NavItem(dbc.NavLink(
                "Charts",
                href=charts_page["path"],
            )),
            dbc.NavItem(dbc.NavLink(
                "Statistics",
                href=statistics_page["path"],
                disabled=True
            )),
            dbc.NavItem(dbc.NavLink(
                "Simulator",
                href="#",
                disabled=True,
            )),
            dbc.NavItem(dbc.NavLink(
                "Calculator",
                href="#",
                disabled=True,
            )),
            dbc.DropdownMenu(
                [
                    dbc.DropdownMenuItem("Discord"),
                    dbc.DropdownMenuItem("GitHub"),
                ],
                label="More",
                nav=True,
                disabled=True,
            ),
        ],
        navbar=True,
        class_name="justify-content-center",
        style={
            "textTransform": "uppercase",
            "width": "66%",
        },
    ),
    dbc.Nav(
        [
            dbc.NavItem(theme_changer),
        ],
        navbar=True,
        class_name="ml-auto justify-content-end",
        style={
            "width": "33%",
        },
    ),
]

navbar_top = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand(
                "rs2sim (WIP)",
                # href=charts_page["path"],
                class_name="d-flex mr-auto",
                style={
                    "width": "25%",
                    "textTransform": "uppercase",
                },
            ),
            dbc.NavbarToggler(
                id="navbar-toggler",
                n_clicks=0,
            ),
            dbc.Collapse(
                nav_items,
                id="navbar-collapse",
                is_open=False,
                navbar=True,
                style={
                    "width": "75%",
                }
            ),
        ],
        fluid=True,
    ),
    expand="lg",
)

app.layout = html.Div(
    [
        navbar_top,
        dash.page_container,
    ]
)


@app.callback(
    Output("navbar-collapse", "is_open"),
    Input("navbar-toggler", "n_clicks"),
    State("navbar-collapse", "is_open"),
)
def toggle_navbar_collapse(
        n_clicks: Optional[int],
        is_open: Optional[bool]) -> Optional[bool]:
    if n_clicks:
        return not is_open
    return is_open


if __name__ == "__main__":
    app.run_server(
        debug=True,
        threaded=True,
        dev_tools_hot_reload=True,
    )
