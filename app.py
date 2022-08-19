import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
from dash_bootstrap_templates import ThemeSwitchAIO
from dash_bootstrap_templates import load_figure_template

load_figure_template("vapor")

external_stylesheets = [
    {
        "href": "https://cdn.jsdelivr.net/gh/orestbida/cookieconsent@v2.8.5/dist/cookieconsent.min.css",
        "rel": "stylesheet",
        "media": "print",
        "onload": "this.media='all'",
    },
    dbc.themes.VAPOR,
]

external_scripts = [
    {
        # Load cookie consent plugin in async mode.
        "async src": "https://cdn.jsdelivr.net/gh/orestbida/cookieconsent@v2.8.5/dist/cookieconsent.min.js",
    },
    {
        # TODO: main script here that inits cookieconsent and loads others on demand.
        "src": "https://cdn.jsdelivr.net/gh/tuokri/solid-bassoon@master/plenty-leave.js"
    },
    {
        # Google Analytics. Can be loaded async, but should be enabled with
        # restricted functionality until consent is given.
        "type": "text/plain",
        "async src": "https://www.googletagmanager.com/gtag/js?id=G-3TNF8134RD",
        "data-cookiecategory": "analytics",
    },
    {
        # Adsense. Should only be loaded after consent given or immediately, but
        # only with non-personalized ads.
        "type": "text/plain",
        "defer src": "https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-3291929185204781",
        "crossorigin": "anonymous",
        "data-cookiecategory": "analytics",
    },
]

app = dash.Dash(
    __name__,
    external_scripts=external_scripts,
    external_stylesheets=external_stylesheets,
    title="rs2sim",
    use_pages=True,
)
server = app.server

# app.css.config.serve_locally = False
# app.scripts.config.serve_locally = False

app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        <script>window['gtag_enable_tcf_support'] = true;</script>
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

home_page = dash.page_registry["pages.home"]

navbar = html.Div()

theme_toggle = ThemeSwitchAIO(
    aio_id="theme",
    themes=[dbc.themes.VAPOR, dbc.themes.SIMPLEX],
    icons={"left": "fa fa-sun", "right": "fa fa-moon"},
)

app.layout = dbc.Container([
    navbar,

    theme_toggle,

    html.H1(
        # children="Rising Storm 2: Vietnam Weapon Simulator",
        dcc.Link("Rising Storm 2: Vietnam Weapon Simulator",
                 href=home_page["relative_path"]),
    ),

    dash.page_container,
])

if __name__ == "__main__":
    app.run_server(
        debug=True,
        threaded=True,
        dev_tools_hot_reload=True
    )
