# rs2simulator

# BETA VERSION - WORK IN PROGRESS

Data visualization app for the Rising Storm 2: Vietnam video game.
Features accurate simulation of in-engine physics projectiles in the browser,
weapon statistics comparison tools and data visualization based on automatically
collected data from the game's source files.

## Using the CLI tools

**TODO: REMEMBER TO UPDATE THIS GUIDE ONCE THE CLI IS RE-DESIGNED!**

### Step 1: Parse UnrealScript source directories

```bash
python rs2simulator/main.py -p path/to/script_sources
```

### Step 2: Parse localization data

```bash
python rs2simulator/main.py -l path/to/localization/file.int
```

### Step 3 Run simulations:

```bash
python rs2simulator/main.py --simulate
```

### Step 4: Write the data from previous steps into a PostgreSQL database:

```bash
python rs2simulator/main.py --enter-sim-data
```

## Running the web app

### Run a development server using PowerShell (option 1):

```powershell
.\run_app.ps1
```

### Run Docker development server container (option 2):

```bash
# TODO: instructions!
```

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

### PostgreSQL database backend

Fly.io Postgres cluster with TimescaleDB extension

- [TimescaleDB extension](https://docs.timescale.com/timescaledb/latest/)

[PGBouncer connection pooler](http://www.pgbouncer.org/)

### TODO

- Migrate to UV at some point?

- Replace logbook with loguru?
