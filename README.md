# rs2simulator

# ALPHA VERSION - WORK IN PROGRESS

Data visualization app for the Rising Storm 2: Vietnam video game.
Features accurate simulation of in-engine physics projectiles in the browser,
weapon statistics comparison tools and data visualization based on automatically
collected data from the game's source files.

## Built with

### Main (web) data app

[Dash & Plotly](https://dash.plotly.com/)

### In-game object simulation and data collection

[rs2simlib](https://github.com/tuokri/rs2simlib)

[NumPy](https://numpy.org/)

[Numba](https://numba.pydata.org/)

[pandas](https://pandas.pydata.org/)

### Database access

[SQLAlchemy](https://www.sqlalchemy.org/)

[Psycopg 3](https://www.psycopg.org/psycopg3/)

### Database backend

Fly.io Postgres cluster with custom modifications:
[rs2sim-postgres-ha](https://github.com/tuokri/rs2sim-postgres-ha)
