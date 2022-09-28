# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 12:13:30 2022

@author: nicolas.julien
"""

import streamlit as st
from streamlit_lottie import st_lottie
import requests
from io import BytesIO
import os
import json
from pathlib import Path
from uuid import uuid4
from stqdm import stqdm
import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_float_dtype
from collections import Counter
import warnings 
import sklearn
from sklearn import preprocessing
import torch
from transformers import LayoutLMv3Processor, AutoModelForTokenClassification
from transformers.utils.logging import set_verbosity_error as set_verbosity_error_transfo
from PIL import Image, ImageDraw, ImageFont, ImageColor
import pytesseract
import pdf2image
from pdf2image import convert_from_bytes
from streamlit_lottie import st_lottie_spinner
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config("ObligAtIon", page_icon = "https://play-lh.googleusercontent.com/atlogEde4hQhEmhZEerOxv16b_2JHUOfvSTQBCOk5bgIPcnqgOWdhZOU5UEbnM3pk30=w240-h480-rw")

st.image("https://www.mazarsrecrute.fr/assets/logos/logoMazars.png", width = 600)

st.header("Scan automatique d'obligations grâce à Layout LMV3")
   
if 'vecteur_utile' not in st.session_state:
    st.session_state.vecteur_utile = []
# permet de ne pas recharger le modèle à chaque fois

if 'compteur_image' not in st.session_state:
    st.session_state.compteur_image = 0

def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

if 'document_upload' not in st.session_state:
    st.session_state.document_upload = True

if st.session_state.document_upload:
    lottie_url_accueil = "https://assets8.lottiefiles.com/packages/lf20_thlxl22p.json"
    lottie_accueil = load_lottie_url(lottie_url_accueil)
    
    col1, mid, col2 = st.columns([10,3,15])
    with col1:
        st_lottie(lottie_accueil, key = 'accueil', width=300)
    with col2:
        st.markdown("Cette application vous propose d'identifier et d'extraire automatiquement dans vos contrats d'obligations .pdf les informations suivantes :")
        st.markdown(" * le Nominal")
        st.markdown(" *  l'Issue Date")
        st.markdown(" * la Première Date de Paiement des Coupons")
        st.markdown(" * la Maturité")
        st.markdown(" * le Taux d'Intérêt des Coupons")
        st.markdown(" * le Facteur de Risque")
        st.markdown(" * l'Option de Call")
     

with st.spinner("Chargement du modèle..."):
 
    path_model = r'NicolasJulienData/ObligAtIon'

    
    #Definition des fonctions utiles------------------------------------------------------------------------------------------------------------------------------------------------
    
    #Création label list
    label_list = ['B-Call Option', 'I-Call Option', 'B-Coupon Period', 'I-Coupon Period', 
                   'B-First Payment Date', 'I-First Payment Date', 'B-Fixed Rate', 'I-Fixed Rate', 
                   'B-Issue Date', 'I-Issue Date', 'B-Maturity', 'I-Maturity', 'B-Nominal', 'I-Nominal', 
                   'B-Risk Factor', 'I-Risk Factor', 'Other']
    
    true_label_list = ['Nominal', 'Issue Date', 'First Payment Date', 'Maturity', 'Coupon Period', 'Fixed Rate', 'Risk Factor', 'Call Option']
    
    color_list = ['magenta', 'magenta', 'yellow','yellow', 'cornflowerblue','cornflowerblue','darkorange','darkorange',
                  'cyan','cyan','navy','navy','red','red','yellowgreen','yellowgreen','grey']
    
    label2color = {k.lower(): v for k,v in zip(label_list, color_list)}
    
    encoder = preprocessing.LabelEncoder()
    encoder.fit(label_list)
    
    id2label = {int(k): v for k,v in zip(encoder.transform(label_list), label_list)}
    label2id = {v: int(k) for k,v in zip(encoder.transform(label_list), label_list)}
    
    set_verbosity_error_transfo()
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    if st.session_state.vecteur_utile ==  []:
        
        st.session_state.vecteur_utile = [LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base"), 
                                          LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", truncation_side = 'right'), 
                                          LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base",truncation_side = 'left'), 
                                          AutoModelForTokenClassification.from_pretrained(path_model)]
        
        
    
    processor_pred = st.session_state.vecteur_utile[0]
    processor_pred_1 = st.session_state.vecteur_utile[1]
    processor_pred_2 = st.session_state.vecteur_utile[2]
    model = st.session_state.vecteur_utile[3]
    
    def unnormalize_box_predictions(bbox, width, height):
         return [
             width * (bbox[0] / 1000),
             height * (bbox[1] / 1000),
             width * (bbox[2] / 1000),
             height * (bbox[3] / 1000),
         ]
    
    def iob_to_label(label):
        if not label:
          return 'other'
        return label
    
    def mode(array):
      b = Counter(array) 
      return(b.most_common(1)[0][0])
    
    def draw_encoding(encoding, image, device = 'cpu'):
    
      width, height = image.size
      draw = ImageDraw.Draw(image)
      font = ImageFont.load_default()
    
      offset_mapping = encoding.pop('offset_mapping') #création des vecteurs words
      encoding.to(device)
      with torch.no_grad():
        outputs = model(**encoding) #création des predictions
    
      predictions = outputs.logits.argmax(-1).squeeze().tolist() #récup des prédictions
      token_boxes = encoding.bbox.squeeze().tolist()
      is_subword = np.array(offset_mapping.squeeze().tolist())[:,0] != 0
      true_predictions = [id2label[pred] for idx, pred in enumerate(predictions) if not is_subword[idx]]
      true_boxes = [unnormalize_box_predictions(box, width, height) for idx, box in enumerate(token_boxes) if not is_subword[idx]]
    
      #bloc drawing
      for prediction, box in stqdm(zip(true_predictions, true_boxes)):
        predicted_label = iob_to_label(prediction).lower()
        if predicted_label != 'other':
          draw.rectangle(box, outline=label2color[predicted_label], width = 3)
          draw.text((box[0]-10, box[1]-30), text=predicted_label, fill=label2color[predicted_label], font=font, stroke_width = 1)
      
      #bloc dataframe
      liste_mots = []
      mot = []
      for indice in range(len(is_subword)):
        token = encoding.input_ids[0][indice]
        token_is_subword = is_subword[indice]
    
        
        if token_is_subword == False:
    
          true_word = processor_pred.tokenizer.decode(mot)
          liste_mots.append(true_word)
          mot = [token]
    
        else:
    
          mot.append(token)
    
      liste_mots = liste_mots[1:]
      limite = len(liste_mots)
      true_predictions = true_predictions[:limite]
      df_predictions = pd.DataFrame({'mots':liste_mots, 'label':true_predictions})
      df_predictions = df_predictions[df_predictions['label'] != 'Other']
    
      return(df_predictions)
    
    def prediction(image, model = model, processor_pred = processor_pred, processor_pred_1 = processor_pred_1, processor_pred_2 = processor_pred_2, device = 'cpu'):
    
      encoding = processor_pred(image, return_offsets_mapping=True, return_tensors="pt") #encodage des données
      length = len(encoding.input_ids[0])
    
      if length > 512:
      
        encoding_1 = processor_pred_1(image, return_offsets_mapping=True, return_tensors="pt", max_length = 512, truncation = True)
    
        encoding_2 = processor_pred_2(image, return_offsets_mapping=True, return_tensors="pt", max_length = min(length-512+1, 512), truncation = True)
        prediction_1 = draw_encoding(encoding_1, image, device)
        prediction_2 = draw_encoding(encoding_2, image, device)
        predict = pd.concat([prediction_1,prediction_2])
    
      else:  
        predict = draw_encoding(encoding, image, device)
      
      return(image, predict)
   
   def search_best_prediction(df):
      ancien_indice = -1
      liste_suspects = []
      phrase = ''
      for idx, mot in zip(df.index, df['mots']):
        if idx != 0:
          if idx == ancien_indice + 1 and mot not in phrase: 
            phrase = phrase + mot
          else:
            if len(phrase)>0:
              liste_suspects.append(phrase[1:])
            phrase = mot 
        ancien_indice = idx

      liste_suspects.append(phrase[1:])
      category_prediction = mode(liste_suspects)

      return(category_prediction, liste_suspects)

   def Inference(path_list, display_images = True, device = 'cpu'):

  index = 0
  array_images = []

  for path in tqdm(path_list):
    pred = prediction(path, device)
    array_images.append(pred[0])
    if index == 0:
      dataframe = pred[1]
    else : 
      dataframe = pd.concat([dataframe, pred[1]])

    index +=1

  if display_images == True:
    for image in array_images:
      display(image)


  best_final_predictions = []
  liste_final_predictions = []

  for Category in true_label_list:

    df_category = dataframe.groupby(dataframe['label']==Category)

    if len(df_category) < 2: #Si pas de prédictions

      best_final_predictions.append('None')
      liste_final_predictions.append('None')

    else:
      df_category = pd.DataFrame(df_category)[1].iloc[1]
      array_pred = search_best_prediction(df_category)
      best_final_predictions.append(array_pred[0])
      liste_final_predictions.append(array_pred[1])

  dic_best_predictions = {category: pred for category,pred in zip(true_label_list, best_final_predictions)}
  dic_liste_predictions = {category: liste for category,liste in zip(true_label_list, liste_final_predictions)}

  return(dic_best_predictions, dic_liste_predictions)

       #------------------------------------------------------------------------------------------------------------------------------------------------


if st.session_state.document_upload :
    
    lottie_url_validation = "https://assets10.lottiefiles.com/packages/lf20_zitlff5a.json"
    lottie_validation = load_lottie_url(lottie_url_validation) 
    
    col_1, milieu, col_2 = st.columns([5,1,15])

    with col_2:
            st_lottie(lottie_validation, loop = False, width = 80)
            html_charge = """
            <span style='color: #26B260'> Modèle chargé !</span> 
            """ 
            st.markdown(html_charge, unsafe_allow_html=True)

st.markdown("") 
st.markdown("")   
st.session_state.document_upload = False
lottie_validation = None

st.sidebar.header("Veuillez déposer le PDF à analyser")

uploaded_file = st.sidebar.file_uploader('Choose your .pdf file', type="pdf")

if uploaded_file is not None:   #and st.button("C'est parti !")

    fichier_pdf = uploaded_file.read() 
    
    if 'predictions_effectuee' not in st.session_state:
        st.session_state.predictions_effectuee = 0

    if st.session_state.predictions_effectuee == 0 or fichier_pdf != st.session_state.ancien_doc:      
    
        lottie_url_process = "https://assets3.lottiefiles.com/packages/lf20_7fwvvesa.json"
        lottie_process = load_lottie_url(lottie_url_process)
        
        col1, mid, col2 = st.columns([10,3,40])
        with col2:
            st.markdown("### *Analyse en cours...*")
            
        col1, mid, col2 = st.columns([1,3,1])      
        with mid: 
            with st_lottie_spinner(lottie_process, key = 'process', width=300, speed=0.5):
                
                st.session_state.predictions_effectuee = Inference(convert_from_bytes(fichier_pdf))
                st.session_state.compteur_image = 0
        
    inf = st.session_state.predictions_effectuee
    keys = inf[0].keys()
    
    k=0
    
    for category in keys:
        predict = inf[0][category]
        html_prediction = f"""
        <style>
        p.a {{
          font: {18}px Arial;
        }}
        </style>
        <p class="a"><b>{category}</b> : <span style='color: #87CEFA'>{predict}</span></p>
        """  
        st.markdown(html_prediction, unsafe_allow_html = True)
        input_button = st.button("Voir les autres propositions", key = k)
        if input_button:
            for element in np.unique(inf[1][category]):
                
                html_element = f"""
                <style>
                p.e {{
                  font: {13}px Arial;
                }}
                </style>
                <p class="e"><span style='color: #D3D3D3'>{element}</span></p>
                """           
                st.markdown(html_element, unsafe_allow_html=True)
        k+=1  
     
    images_traitees = inf[2]
    taille = len(images_traitees)
    
    col1, mid, col2 = st.columns([70,11,9])
    
    with mid:
        if st.button("Previous"):
            st.session_state.compteur_image = max(st.session_state.compteur_image-1, 0)
    with col2:
        if st.button("Next"):
            st.session_state.compteur_image = min(st.session_state.compteur_image+1, taille-1)
    with col1:
      st.write("Page ", st.session_state.compteur_image+1, "/", taille)
    
     
    st.image(images_traitees[st.session_state.compteur_image])
    
    if 'ancien_doc' not in st.session_state:
        st.session_state.ancien_doc = 0
    
    st.session_state.ancien_doc = fichier_pdf
    
