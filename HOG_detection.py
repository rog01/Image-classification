import datetime

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import pickle
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import skimage
from skimage.feature import hog
import dash_bootstrap_components as dbc

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
bootstrap_theme=[dbc.themes.BOOTSTRAP,'https://use.fontawesome.com/releases/v5.9.0/css/all.css']


app = dash.Dash(__name__, external_stylesheets=bootstrap_theme)

app.layout = html.Div([
    html.H2("Image classifier",style = {'text-align':'center','color':'blue'}),
    html.H4("for buildings, forest, glacier, mountain, sea, street",style = {'text-align':'center'}),
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

#animal images
#with open('Mlp_clf_image_classifier.pkl','rb') as file:
    #Mlp_image_clf = pickle.load(file)
#palces images    
with open('places_mlp_clf.pkl','rb') as file:
    Mlp_image_clf = pickle.load(file)
 
class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
    Convert an array of RGB images to grayscale
    """ 
    def __init__(self):
        pass 
    def fit(self, X, y=None):
        """returns itself"""
        return self 
    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([skimage.color.rgb2gray(img) for img in X])


class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """ 
    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X, y=None):
 
        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)
 
        try: # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])

# create an instance of each transformer
grayify = RGB2GrayTransformer()
hogify = HogTransformer(
    pixels_per_cell=(8, 8), 
    cells_per_block=(3,3), 
    orientations=8, 
    block_norm='L2-Hys'
)
scalify = StandardScaler()

with open('scalify_fit.pkl','rb') as file:
    scalify_fit = pickle.load(file)

def image_predict(img):
    img = imread(img)
    width = 120
    height = 120
    img = resize(img, (width, height)) 
    X_test_gray = grayify.transform([img])
    X_test_hog = hogify.transform(X_test_gray)
    X_test_prepared = scalify_fit.transform(X_test_hog)
    y_pred = Mlp_image_clf.predict(X_test_prepared) 
    return  y_pred   

def parse_contents(contents, filename, date):
    return html.Div([        
        #html.H5(filename),
        #html.H6(datetime.datetime.fromtimestamp(date)),

        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        html.Img(src=contents),
        html.Pre(image_predict("test/places/google/"+filename),style={'font-size':'20px','color':'green'}),
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