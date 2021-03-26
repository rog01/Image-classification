#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 21:55:28 2021

@author: roger
"""

import datetime

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

import pickle
import numpy as np
import dash_bootstrap_components as dbc

from keras.models import load_model
import tensorflow as tf

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
bootstrap_theme=[dbc.themes.BOOTSTRAP,'https://use.fontawesome.com/releases/v5.9.0/css/all.css']


app = dash.Dash(__name__, external_stylesheets=bootstrap_theme)

app.layout = html.Div([
    html.H2("Image classifier",style = {'text-align':'center','color':'blue'}),
    html.H4("for animals",style = {'text-align':'center'}),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files to labelize them')
        ]),
        style={
            'width': '60%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px',
            'margin-left' : '250px',
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-image-upload',
              style = {'margin-left' : '200px',
                       'color':'green'
   })
    
])
places_class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']


class_names = ['Bear', 'Cat', 'Chicken', 'Cow', 'Deer', 'Dog', 'Duck', 'Eagle', 'Elephant', 'Human', 'Lion', 'Monkey', 'Mouse', 'Nat', 'Panda', 'Pig', 'Pigeon', 'Rabbit', 'Sheep', 'Tiger', 'Wolf']
Places_Xception_model = Xception_model = load_model('Places_Xception_model.h5', compile=False)

Xception_model = load_model('Xception_model.h5', compile=False)
IMG_SIZE = 150

def image_predict(img):
    img = tf.keras.preprocessing.image.load_img(img, target_size=(IMG_SIZE, IMG_SIZE, 3))

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = Xception_model.predict(img_array)
    y_pred = class_names[np.argmax(predictions)]
    
    # predictions = Places_Xception_model.predict(img_array)
    # y_pred = places_class_names[np.argmax(predictions)]
    return  y_pred   

def parse_contents(contents, filename, date):
    return html.Div([        
        #html.H5(filename),
        #html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Pre(image_predict("test/animaux/google/"+filename),style={'font-size':'20px','color':'green'}),
        #html.Pre(image_predict("test/places/google/"+filename),style={'font-size':'20px','color':'green'}),
        
        #html.Hr(),
        #html.Div('Raw Content'),
        #html.Pre(contents[0:200] + '...', style={
            #'whiteSpace': 'pre-wrap',
            #'wordBreak': 'break-all'
        #})
    ],style={'display': 'inline-block',
                       'margin-right' : '10px',
                       'text-align': 'center'
             })


@app.callback(Output('output-image-upload', 'children'),
              Input('upload-image', 'contents'),
              State('upload-image', 'filename'),
              State('upload-image', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children


if __name__ == '__main__':
    #app.run_server(debug=True)
    app.run_server(debug=True, host='127.0.0.1',port=8050)