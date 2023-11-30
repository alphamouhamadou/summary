import streamlit as st
from bs4 import BeautifulSoup as bs
from requests import get
import re
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer

# Charger les données depuis Wikipedia
def load_data(language):
    st.title("Text Summarizer")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;"> Yves & Mouhamadou </h2>
    </div>
    """
    url = 'https://en.wikipedia.org/wiki/Iranian_cuisine'
    if language == 'French':
        url = 'https://fr.wikipedia.org/wiki/Cuisine_iranienne'
    
    resp = get(url)
    article_soup = bs(resp.text)
    paragraphs = article_soup.find_all('p')
    article_text = ""
    for p in paragraphs:
        article_text += p.text
    return article_text

# Nettoyer les données
def clean_data(article_text):
    article_text = re.sub(r'[[\w]*]', ' ', article_text)
    article_text = re.sub(r'\xa0|\u200c', ' ', article_text)
    article_text = re.sub(r'/s+', ' ', article_text)
    article_text = re.sub(r'^\s|\s$', '', article_text)
    return article_text

# Résumé avec NLTK
def nltk_summary(article_text, num_sentences):
    sentence_list = nltk.sent_tokenize(article_text)
    
    # Logique pour générer le résumé
    summary_sentences = sentence_list[:num_sentences]

    return summary_sentences

# Résumé avec Sumy TextRank
def sumy_textrank_summary(article_text, num_sentences):
    parser = PlaintextParser.from_string(article_text, Tokenizer('english'))
    summarizer_textrank = TextRankSummarizer()
    summary = summarizer_textrank(parser.document, num_sentences)
    return summary

# Résumé avec Sumy LexRank
def sumy_lexrank_summary(article_text, num_sentences):
    parser = PlaintextParser.from_string(article_text, Tokenizer('english'))
    summarizer_lexrank = LexRankSummarizer()
    summary = summarizer_lexrank(parser.document, num_sentences)
    return summary

# Résumé avec Sumy LsaSummarizer
def sumy_lsa_summary(article_text, num_sentences):
    parser = PlaintextParser.from_string(article_text, Tokenizer('english'))
    summarizer_lsa = LsaSummarizer()
    summary = summarizer_lsa(parser.document, num_sentences)
    return summary

# Main Streamlit App
def main():
    st.title("Text Summarization App")
    
    # Choix de la langue
    language = st.radio("Choisissez la langue de votre texte:", ('English', 'French'))

    # Zone de texte pour la saisie de l'utilisateur
    user_input = st.text_area("Saisissez votre texte ici:")

    # Choix du nombre de phrases pour le résumé
    num_sentences = st.selectbox("Choisissez le nombre de phrases pour le résumé:", list(range(1, 11)), index=4)

    # Boutons pour les modèles
    if st.button("Résumé avec NLTK") and user_input:
        cleaned_text = clean_data(user_input)
        summary = nltk_summary(cleaned_text, num_sentences)
        st.write("Résumé avec NLTK:", " ".join(summary))

    if st.button("Résumé avec Sumy TextRank") and user_input:
        cleaned_text = clean_data(user_input)
        summary = sumy_textrank_summary(cleaned_text, num_sentences)
        st.write("Résumé avec Sumy TextRank:", " ".join(map(str, summary)))

    if st.button("Résumé avec Sumy LexRank") and user_input:
        cleaned_text = clean_data(user_input)
        summary = sumy_lexrank_summary(cleaned_text, num_sentences)
        st.write("Résumé avec Sumy LexRank:", " ".join(map(str, summary)))

    if st.button("Résumé avec Sumy LsaSummarizer") and user_input:
        cleaned_text = clean_data(user_input)
        summary = sumy_lsa_summary(cleaned_text, num_sentences)
        st.write("Résumé avec Sumy LsaSummarizer:", " ".join(map(str, summary)))

if __name__ == "__main__":
    main()
