import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import joblib
import plotly.express as px
import requests
import sqlite3
from sklearn.preprocessing import LabelEncoder
cricket_model = joblib.load("cricket_model.pkl")
football_model = joblib.load("football_model.pkl")
basketball_model = joblib.load("basketball_model.pkl")
football_encoder = joblib.load("football_encoder.pkl")
basketball_encoder = joblib.load("basketball_encoder.pkl")
cricket_encoder = joblib.load("cricket_encoder.pkl")
conn = sqlite3.connect("sports_injuries.db")
football_data = pd.read_sql_query("SELECT * FROM football_injuries", conn)
basketball_data = pd.read_sql_query("SELECT * FROM basketball_injuries", conn)
conn.close()
cricket_data = pd.read_csv("real_cricket_injuries.csv")
weather_data = pd.read_csv("weather_fully_working_locations.csv")

external_stylesheets = ["https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap"]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
app.title = "Sports Dashboard"

app.layout = html.Div([
    html.H1("Sports Injury & Environment Dashboard", style={"textAlign": "center", "color": "#ffffff"}),
    dcc.Tabs(id="main-tabs", value='predict', children=[
        dcc.Tab(label='Injury Prediction', value='predict'),
        dcc.Tab(label='Weather & Pitch Conditions', value='weather')
    ], style={"fontWeight": "bold"}),
    html.Div(id='tab-content')
], style={"backgroundColor": "#1e1e2f", "fontFamily": "Roboto", "padding": "20px"})

@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value')
)
def render_tab_content(tab):
    if tab == 'predict':
        return html.Div([
            html.Label("Select Sport", style={"color": "#ffffff"}),
            dcc.Dropdown(
                id='sport-dropdown',
                options=[
                    {'label': 'Football', 'value': 'football'},
                    {'label': 'Basketball', 'value': 'basketball'},
                    {'label': 'Cricket', 'value': 'cricket'}
                ],
                placeholder='Choose a sport'
            ),
            html.Br(),
            html.Label("Player Name", style={"color": "#ffffff"}),
            dcc.Input(id='player-name', type='text', placeholder='Enter player name'),
            html.Br(), html.Br(),
            html.Label("Injury Type", style={"color": "#ffffff"}),
            dcc.Dropdown(id='injury-type-dropdown', placeholder="Select injury type", searchable=True, clearable=False),
            html.Br(),
            html.Label("Player Event Count", style={"color": "#ffffff"}),
            dcc.Input(id='player-event-count', type='number', placeholder='Number of prior injuries'),
            html.Br(), html.Br(),
            html.Label("Days Since Last Injury", style={"color": "#ffffff"}),
            dcc.Input(id='days-since-last', type='number', placeholder='Days since last'),
            html.Br(), html.Br(),
            html.Button("Predict Severity", id="predict-button", n_clicks=0,
                        style={"backgroundColor": "#2c7be5", "color": "white", "padding": "10px 25px", "border": "none"}),
            html.Div(id="prediction-output", style={'marginTop': '20px', "color": "#ffffff", "fontSize": "20px"})
        ])
    elif tab == 'weather':
        return html.Div([
            html.Label("Select Sport", style={"color": "#ffffff"}),
            dcc.Dropdown(
                id='weather-sport-filter',
                options=[{"label": s.title(), "value": s} for s in sorted(weather_data["sport"].unique())],
                placeholder="Choose a sport"
            ),
            html.Br(),
            html.Label("Select Venue / Location", style={"color": "#ffffff"}),
            dcc.Dropdown(id='venue-dropdown', placeholder="Select venue"),
            html.Br(),
            dcc.Graph(id="weather-graph"),
            html.Div(id="live-weather", style={"marginTop": "20px"})
        ])

@app.callback(
    Output("injury-type-dropdown", "options"),
    Input("sport-dropdown", "value")
)
def update_injury_types(sport):
    if sport == "football" and not football_data.empty:
        types = football_data["Injury"].dropna().unique()
    elif sport == "basketball" and not basketball_data.empty:
        types = basketball_data["Notes"].dropna().unique()
    elif sport == "cricket" and not cricket_data.empty:
        types = cricket_data["injury"].dropna().unique()
    else:
        return []
    return [{"label": t, "value": t} for t in sorted(types)]

@app.callback(
    Output("prediction-output", "children"),
    Input("predict-button", "n_clicks"),
    State("sport-dropdown", "value"),
    State("injury-type-dropdown", "value"),
    State("player-event-count", "value"),
    State("days-since-last", "value")
)
def predict(n_clicks, sport, injury, count, days):
    if n_clicks > 0:
        if None in (sport, injury, count, days):
            return "❗ Please fill all fields to predict."

        if sport == "football":
            model, encoder = football_model, football_encoder
        elif sport == "basketball":
            model, encoder = basketball_model, basketball_encoder
        elif sport == "cricket":
            model, encoder = cricket_model, cricket_encoder
        else:
            return "⚠️ Unsupported sport."

        try:
            encoded_injury = encoder.transform([injury])[0]
        except:
            return f"⚠️ Injury type '{injury}' not recognized."

        df = pd.DataFrame([{
            "injury_type_encoded": encoded_injury,
            "player_event_count": count,
            "days_since_last": days
        }])

        prediction = model.predict(df)[0]
        confidence = model.predict_proba(df).max()

        label = "🚨 Severe" if prediction == 1 else "✅ Not Severe"
        return f"Prediction: {label} (Confidence: {round(confidence, 2)})"
    return ""

@app.callback(
    Output("venue-dropdown", "options"),
    Input("weather-sport-filter", "value")
)
def update_venues_by_sport(sport):
    if sport:
        venues = weather_data[weather_data["sport"] == sport]["location"].dropna().unique()
        return [{"label": v, "value": v} for v in sorted(venues)]
    return []

@app.callback(
    Output("weather-graph", "figure"),
    Input("venue-dropdown", "value"),
    State("weather-sport-filter", "value")
)
def update_weather_figure(venue, sport):
    if venue and sport:
        api_key = "QE22KGDF9CVS2JHFZWQBWHS44"
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{venue}/next2days?unitGroup=metric&key={api_key}&contentType=json"
        try:
            res = requests.get(url)
            data = res.json()
            df = pd.DataFrame(data["days"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            fig = px.line(df, x="datetime", y=["temp", "humidity", "windspeed"],
                          title=f"48-Hour Forecast for {venue.title()} ({sport.title()})",
                          markers=True, labels={"value": "Metric", "datetime": "Time"})
            return fig
        except Exception as e:
            return px.bar(title=f"Error fetching live data for {venue}: {e}")
    return px.bar(title="⚠️ Please select a valid sport and venue.")

@app.callback(
    Output("live-weather", "children"),
    Input("venue-dropdown", "value"),
    prevent_initial_call=True
)
def fetch_live_weather(location):
    if location:
        api_key = "QE22KGDF9CVS2JHFZWQBWHS44"
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/today?unitGroup=metric&key={api_key}&contentType=json"
        try:
            res = requests.get(url)
            data = res.json()
            today = data["days"][0]
            return html.Div([
                html.H4(f"Live Weather in {location.title()}", style={"color": "#fff"}),
                html.P(f"Temperature: {today.get('temp')} °C", style={"color": "#ccc"}),
                html.P(f"Humidity: {today.get('humidity')}%", style={"color": "#ccc"}),
                html.P(f"Conditions: {today.get('conditions')}", style={"color": "#ccc"})
            ])
        except Exception as e:
            return html.Div(f"⚠️ Error fetching weather: {e}", style={"color": "red"})
    return ""

if __name__ == "__main__":
    app.run(debug=True)
