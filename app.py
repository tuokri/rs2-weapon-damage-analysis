import json
import os
from typing import Dict
from typing import List

import dash
import plotly.express as px


def load_external_scripts(var: str) -> List[Dict]:
    scripts = os.environ.get(var)
    if not scripts:
        return []
    return json.loads(scripts)


external_scripts = load_external_scripts("EXTERNAL_SCRIPTS")

app = dash.Dash(
    "RS2 Weapon Simulator",
    external_scripts=external_scripts,
)
server = app.server

fig = px.line()

if __name__ == "__main__":
    app.run_server(debug=False)
