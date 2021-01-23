
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import pickle
import xgboost as xgb


app = dash.Dash(__name__,external_stylesheets=[dbc.themes.CERULEAN])

server = app.server

# test pickle model
loaded_model = pickle.load(open('xgboost_model.sav', 'rb'))
loaded_ohe = pickle.load(open('ohe.sav', 'rb'))
loaded_scaler = pickle.load(open('scaler.sav', 'rb'))


app.layout = dbc.Form(children=[


    dbc.Row(html.H1(children='Herramienta ATMs',className="p-5")),

    dbc.FormGroup(   
     [
        dbc.Label("Regional",width={"size": 1, "offset": 2}, color="primary",className="alert-link"),
        dbc.Col(
        dcc.Dropdown(
            id='regional-dropdown',
            options=[
                {'label': 'Antioquia', 'value': 'Antioquia'},
                {'label': 'Bogotá y Cundinamarca', 'value': 'Bogotá y Cundinamarca'},
                {'label': 'Caribe', 'value': 'Caribe'},
                {'label': 'Centro Sur', 'value': 'Centro Sur'},
                {'label': 'Eje Cafetero', 'value': 'Eje Cafetero'},
                {'label': 'Santanderes Y Arauca', 'value': 'Santanderes Y Arauca'},
                {'label': 'Valle y Cauca', 'value': 'Valle y Cauca'},
                    ]),width={"size": 5, "offset": 0},align="center"),
        dbc.Label(id='dd-output-container-regional',width={"size": 6, "offset": 2})
    ],row=True),


    dbc.FormGroup(   
     [
        dbc.Label("Ubicación",width={"size": 1, "offset": 2}, color="primary",className="alert-link"),
        dbc.Col(
        dcc.Dropdown(
            id='ubicacion-dropdown',
            options=[
                {'label': 'Rural', 'value': 'Rural'},
                {'label': 'Urbana', 'value': 'Urbana'},
                    ]),width={"size": 5, "offset": 0},align="center"),
        dbc.Label(id='dd-output-container-ubicacion',width={"size": 6, "offset": 2})
    ],row=True),


    dbc.FormGroup(   
     [
        dbc.Label("Segmento",width={"size": 1, "offset": 2}, color="primary",className="alert-link"),
        dbc.Col(
        dcc.Dropdown(
            id='segmento-dropdown',
            options=[
                {'label': 'Calle y remotos', 'value': 'Calle y remotos'},
                {'label': 'Centro Comercial', 'value': 'Centro Comercial'},
                {'label': 'Empresarial Y Corporativo', 'value': 'Empresarial Y Corporativo'},
                {'label': 'Grandes superficies y Almacenes especializados', 'value': 'Grandes superficies y Almacenes especializados'},
                {'label': 'Oficina', 'value': 'Oficina'},
                {'label': 'Trans. Público y Aeropuertos', 'value': 'Trans. Público y Aeropuertos'},
            ]),width={"size": 5, "offset": 0},align="center"),

        dbc.Label(id='dd-output-container-segmento',width={"size": 6, "offset": 2})
    ],row=True),

    dbc.FormGroup(
        [
            dbc.Label("Comercios cercanos",width={"size": 1, "offset": 2}, color="primary",className="alert-link"),
            dbc.Col(
            dbc.Input(
                id="input_comercios_cercanos", 
                type="number", 
                placeholder="Debounce onChange"
                ),width={"size": 5, "offset": 0},align="center"),
            
            dbc.Label(id='output_comercios_cercanos',width={"size": 6, "offset": 2}),            
        ],row=True),

    dbc.FormGroup(
        [
            dbc.Label("ATMs cercanos competencia",width={"size": 1, "offset": 2}, color="primary",className="alert-link"),

            dbc.Col(
            dbc.Input(
                id="atms_cercanos_competencia", 
                type="number", 
                placeholder="Debounce onChange"
                ),width={"size": 5, "offset": 0},align="center"),
            
            dbc.Label(id='output_atms_cercanos_competencia',width={"size": 6, "offset": 2}),                        
        ],row=True),

    dbc.FormGroup(
        [

            dbc.Label("Trafico",width={"size": 1, "offset": 2}, color="primary",className="alert-link"),

            dbc.Col(
            dbc.Input(
                id="input_trafico", 
                type="number", 
                placeholder="Debounce onChange"
                ),width={"size": 5, "offset": 0},align="center"),

            dbc.Label(id='output_trafico',width={"size": 6, "offset": 2}),                                    
        ],row=True),

    dbc.Row([
        dbc.Button('Predecir transacciones',
                  id='btn-nclicks-1', 
                  n_clicks=0, 
                  color="primary", 
                  className="mr-1",
                  size="lg"),
            ],justify="center"),

    dbc.FormGroup(
    dbc.Row(
        [
            dbc.Label(id="container-button-timestamp")
        ],justify="center"))

])

@app.callback(
    dash.dependencies.Output('dd-output-container-regional', 'children'),
    [dash.dependencies.Input('regional-dropdown', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)
 

@app.callback(
    dash.dependencies.Output('dd-output-container-ubicacion', 'children'),
    [dash.dependencies.Input('ubicacion-dropdown', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)


@app.callback(
    dash.dependencies.Output('dd-output-container-segmento', 'children'),
    [dash.dependencies.Input('segmento-dropdown', 'value')])
def update_output(value):
    return 'You have selected "{}"'.format(value)

@app.callback(
    Output("output_comercios_cercanos", "children"),
    Input("input_comercios_cercanos", "value"))
def update_output(input_comercios_cercanos):
    return 'Input comercios cercanos {}'.format(input_comercios_cercanos)

@app.callback(
    Output("output_atms_cercanos_competencia", "children"),
    Input("atms_cercanos_competencia", "value"))
def update_output(atms_cercanos_competencia):
    return 'Input ATMs cercanos  competencia {}'.format(atms_cercanos_competencia)

@app.callback(
    Output("output_trafico", "children"),
    Input("input_trafico", "value"))
def update_output(input_trafico):
    return 'Input trafico {}'.format(input_trafico)


@app.callback(Output('container-button-timestamp', 'children'),
              Input('btn-nclicks-1', 'n_clicks'),
              [dash.dependencies.State('regional-dropdown', 'value')],
              [dash.dependencies.State('ubicacion-dropdown', 'value')],
              [dash.dependencies.State('segmento-dropdown', 'value')],
              [dash.dependencies.State('input_comercios_cercanos', 'value')],
              [dash.dependencies.State('atms_cercanos_competencia', 'value')],
              [dash.dependencies.State('input_trafico', 'value')])
def displayClick(n_clicks,regional_value,ubicacion_value,segmentacion_value,comercios_cercanos_value,atms_cercanos_competencia,trafico_val):

    regional = regional_value
    ubicacion = ubicacion_value
    segmento = segmentacion_value
    comercios_cercanos = comercios_cercanos_value
    atms_competencia = atms_cercanos_competencia
    trafico = trafico_val

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'btn-nclicks-1' in changed_id and all(v is not None for v in [regional, ubicacion, segmento, comercios_cercanos, atms_competencia, trafico]):
       
        # dictionary with list object in values 
        values = { 
            'Regional' : [regional], 
            'Ubicación' : [ubicacion],
            'Segmentación':[segmento],
            'Antiguedad':["Menor o igual a 2 años"],
            'Trafico ':[trafico],
            '#comercios cercanos':[comercios_cercanos],
            'ATMs Cercanos Competencia':[atms_competencia]    
        }

        # creating a Dataframe object  
        df = pd.DataFrame(values) 

        df = df.apply(lambda x: x.astype(str).str.upper())


        #prediction

        # Define which columns should be encoded vs scaled
        columns_to_scale  = ['ATMs Cercanos Competencia', '#comercios cercanos','Trafico ']
        columns_to_encode = ['Regional','Ubicación','Segmentación','Antiguedad']


        cat_ohe_new_data = loaded_ohe.transform(df[columns_to_encode])
        scaled_columns_data  = loaded_scaler.transform(df[columns_to_scale])

        ohe_df_new_data = pd.DataFrame(cat_ohe_new_data, columns = loaded_ohe.get_feature_names(input_features = columns_to_encode))
        scaled_df_new_data = pd.DataFrame(scaled_columns_data, columns = columns_to_scale)

        frames_new_data = [ohe_df_new_data, scaled_df_new_data]
        processed_data_new_data = pd.concat(frames_new_data, axis=1)
        prediction_txs = np.round(loaded_model.predict(processed_data_new_data)[0])

        mensaje = "Transacciones promedio mes esperada: " + str(prediction_txs)

        return html.H1(mensaje)

    else:

        msg = 'por favor llenar todos los campos'
        return html.H6(msg,style={'color':'red'})




if __name__ == '__main__':
    app.run_server(debug=True)
