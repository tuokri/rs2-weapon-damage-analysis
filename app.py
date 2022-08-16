import dash
import plotly.express as px

app = dash.Dash("RS2 Weapons")

fig = px.line()

if __name__ == "__main__":
    app.run_server(debug=True)
