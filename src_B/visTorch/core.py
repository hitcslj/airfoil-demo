# -*- coding: utf-8 -*-
import random
import dash
# import dash_html_components as html
from dash import html
import dash_bootstrap_components as dbc
from visTorch.components import pca


class VisBoard:
    def __init__(self):
        """Creates and initializes a Plotly Dash App to which more visual components could be added"""

        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.app.config.suppress_callback_exceptions = True
        self.navbar = self.get_navbar(self.app)
        self.body_children = []

    @staticmethod
    def get_navbar(app):
        navbar = dbc.NavbarSimple(
            brand="",
            brand_href="#"
        )
        return navbar
    
    def run_server(self, host, port, debug=True):
        """

        :param host: Address to host the app
        :param port: Port for app hosting
        :param debug:
        :return:
        """
        body = dbc.Container(self.body_children)
        self.app.layout = html.Div([self.navbar, body])
        self.app.run_server(host=host, port=port, debug=debug)


    def add_pca(self, model, dataset, latent_options, pre_process=None):
        """ Adds an PCA Interface to the app.
        :param model: Pytorch model with functions : encoder(), decoder()
        :param dataset: Dataset for sampling input
        """
        prefix = str(random.randint(0, 100000000000))
        pca_ins = pca(self.app, model, dataset, latent_options, pre_process, prefix=prefix)
        self.body_children.append(pca_ins)