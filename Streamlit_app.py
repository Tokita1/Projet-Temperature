import streamlit as st
# <-- Insérer code Python contenant les commandes Streamlit dans le fichier streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error

st.title("Température Terrestre")

st.sidebar.title("Sommaire")
pages=["Introduction au projet", "Compréhension et manipulation des données", "DataVisualisation", "Modélisation"]
page=st.sidebar.radio("Aller vers", pages)

##########################################################
#INTRODUCTION AU PROJET
##########################################################

if page == pages[0] :
  st.write("### I. INTRODUCTION AU PROJET")
  texte_introduction_au_projet = """
  Ce projet s’inscrit pleinement dans notre apprentissage du métier de Data Analyst grâce à la manipulation de données “from scratch”, menant à l’analyse et l’interprétation de ces données brutes.
  De plus, ce sujet s’inscrit pleinement dans le contexte professionnel d’une partie de notre groupe, confrontés aux objectifs carbones et environnementaux qui sont essentiels face aux évolutions rapides des réglementations.
  L’objectif de ce projet est de constater le réchauffement climatique et le dérèglement climatique global à l’échelle de la planète sur les derniers siècles et dernières décennies. Ce phénomène sera analysé au niveau mondial et par zone géographique. Nous comparerons avec des phases d’évolution de température antérieure à notre époque.
  Le projet est piloté par Florian Delattre, Philippe Grenesche, Yves Liais et Florian Matrat,
  et est supervisé par Alain Ferlac.
  Aucun des membres du projet n’a d’expérience dans l’analyse du réchauffement climatique.
  """
  st.write(texte_introduction_au_projet)
