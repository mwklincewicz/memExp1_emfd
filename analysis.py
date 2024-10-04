import numpy as np
import spacy
print("spaCy version:", spacy.__version__)
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
import pandas as pd
print("pandas version:", pd.__version__)
import seaborn as sns
from matplotlib import pyplot as plt
from emfdscore.scoring import score_docs

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 400)

#exp1_df = pd.read_excel("./test.xlsx")

exp1_df = pd.read_excel("./rawdata_exp1.xlsx")
#exp2_df = pd.read_excel("./rawdata_exp2.xlsx")

#exp1_df = pd.concat([exp1_df, exp2_df] )

my_translator = GoogleTranslator(source='auto', target='en')
my_nl_translator = GoogleTranslator(source='auto', target='nl')

sentimentAnalysisVader = SentimentIntensityAnalyzer()
for index, text in exp1_df.iterrows():
    exp1_df.loc[index, 'abortion_3words_en'] = my_translator.translate(exp1_df['abortion_3words'][index])
    exp1_df.loc[index, 'LGBTQI_3words_en'] = my_translator.translate(exp1_df['LGBTQI_3words'][index])
    exp1_df.loc[index, 'terrorism_3words_en'] = my_translator.translate(exp1_df['terrorism_3words'][index])
    exp1_df.loc[index, 'blackPete_3words_en'] = my_translator.translate(exp1_df['blackPete_3words'][index])
    exp1_df.loc[index, 'climateChange_3words_en'] = my_translator.translate(exp1_df['climateChange_3words'][index])
    exp1_df.loc[index, 'developmentAid_3words_en'] = my_translator.translate(exp1_df['developmentAid_3words'][index])
    exp1_df.loc[index, 'EU_3words_en'] = my_translator.translate(exp1_df['EU_3words'][index])
    exp1_df.loc[index, 'Tax_3words_en'] = my_translator.translate(exp1_df['Tax_3words'][index])
    exp1_df.loc[index, 'healthcare_3words_en'] = my_translator.translate(exp1_df['healthcare_3words'][index])

    exp1_df.loc[index, 'abortion_3words'] = my_nl_translator.translate(exp1_df['abortion_3words'][index])
    exp1_df.loc[index, 'LGBTQI_3words'] = my_nl_translator.translate(exp1_df['LGBTQI_3words'][index])
    exp1_df.loc[index, 'terrorism_3words'] = my_nl_translator.translate(exp1_df['terrorism_3words'][index])
    exp1_df.loc[index, 'blackPete_3words'] = my_nl_translator.translate(exp1_df['blackPete_3words'][index])
    exp1_df.loc[index, 'climateChange_3words'] = my_nl_translator.translate(exp1_df['climateChange_3words'][index])
    exp1_df.loc[index, 'developmentAid_3words'] = my_nl_translator.translate(exp1_df['developmentAid_3words'][index])
    exp1_df.loc[index, 'EU_3words'] = my_nl_translator.translate(exp1_df['EU_3words'][index])
    exp1_df.loc[index, 'Tax_3words'] = my_nl_translator.translate(exp1_df['Tax_3words'][index])
    exp1_df.loc[index, 'healthcare_3words'] = my_nl_translator.translate(exp1_df['healthcare_3words'][index])

    exp1_df.loc[index, 'allText'] = exp1_df.loc[index, 'abortion_3words_en'] + " " + \
                                    exp1_df.loc[index, 'LGBTQI_3words_en']+ " " + \
                                    exp1_df.loc[index, 'terrorism_3words_en']+ " " + \
                                    exp1_df.loc[index, 'blackPete_3words_en']+ " " + \
                                    exp1_df.loc[index, 'climateChange_3words_en']+ " " + \
                                    exp1_df.loc[index, 'developmentAid_3words_en']+ " " + \
                                    exp1_df.loc[index, 'EU_3words_en'] + " " + \
                                    exp1_df.loc[index, 'Tax_3words_en'] + " " + \
                                    exp1_df.loc[index, 'healthcare_3words_en'] + " "


    if (exp1_df.notnull().loc[index, 'recentEvent']):
        # translate to both English and Dutch (if original in English)
        exp1_df.loc[index, 'recentEvent_en'] = my_translator.translate(exp1_df['recentEvent'][index])
        exp1_df.loc[index, 'recentEvent'] = my_nl_translator.translate(exp1_df['recentEvent'][index])
        # get sentiment for both English and Dutch into new columns
        sentiment_dict = sentimentAnalysisVader.polarity_scores(exp1_df['recentEvent_en'][index])
        exp1_df.loc[index, 'recentEvent_en_NEG'] = sentiment_dict['neg'] * 100
        exp1_df.loc[index, 'recentEvent_en_POS'] = sentiment_dict['pos'] * 100
        exp1_df.loc[index, 'recentEvent_en_NEU'] = sentiment_dict['neu'] * 100
        exp1_df.loc[index, 'recentEvent_en_COM'] = sentiment_dict['compound'] * 100

        exp1_df.loc[index, 'allText'] = exp1_df.loc[index, 'allText'] + " " + exp1_df.loc[index, 'recentEvent_en']


    if ( exp1_df.notnull().loc[index, 'abortion'] ):
        #translate to both English and Dutch (if original in English)
        exp1_df.loc[index, 'abortion_en'] = my_translator.translate(exp1_df['abortion'][index])
        exp1_df.loc[index, 'abortion'] = my_nl_translator.translate(exp1_df['abortion'][index])
        #get sentiment for both English and Dutch into new columns
        sentiment_dict = sentimentAnalysisVader.polarity_scores(exp1_df['abortion_en'][index])
        exp1_df.loc[index, 'abortion_en_NEG'] = sentiment_dict['neg'] * 100
        exp1_df.loc[index, 'abortion_en_POS'] = sentiment_dict['pos'] * 100
        exp1_df.loc[index, 'abortion_en_NEU'] = sentiment_dict['neu'] * 100
        exp1_df.loc[index, 'abortion_en_COM'] = sentiment_dict['compound'] * 100
        sentiment_dict = sentimentAnalysisVader.polarity_scores(exp1_df['abortion_3words_en'][index])
        exp1_df.loc[index, 'abortion_3words_NEG'] = sentiment_dict['neg'] * 100
        exp1_df.loc[index, 'abortion_3words_POS'] = sentiment_dict['pos'] * 100
        exp1_df.loc[index, 'abortion_3words_NEU'] = sentiment_dict['neu'] * 100
        exp1_df.loc[index, 'abortion_3words_COM'] = sentiment_dict['compound'] * 100

        exp1_df.loc[index, 'allText'] = exp1_df.loc[index, 'allText'] + " " + exp1_df.loc[index, 'abortion_en']

    if ( exp1_df.notnull().loc[index, 'LGBTQI'] ):
        # translate to both English and Dutch (if original in English)
        exp1_df.loc[index, 'LGBTQI_en'] = my_translator.translate(exp1_df['LGBTQI'][index])
        exp1_df.loc[index, 'LGBTQI'] = my_nl_translator.translate(exp1_df['LGBTQI'][index])
        # get sentiment for both English and Dutch into new columns
        sentiment_dict = sentimentAnalysisVader.polarity_scores(exp1_df['LGBTQI_en'][index])
        exp1_df.loc[index, 'LGBTQI_en_NEG'] = sentiment_dict['neg'] * 100
        exp1_df.loc[index, 'LGBTQI_en_POS'] = sentiment_dict['pos'] * 100
        exp1_df.loc[index, 'LGBTQI_en_NEU'] = sentiment_dict['neu'] * 100
        exp1_df.loc[index, 'LGBTQI_en_COM'] = sentiment_dict['compound'] * 100
        sentiment_dict = sentimentAnalysisVader.polarity_scores(exp1_df['LGBTQI_3words_en'][index])
        exp1_df.loc[index, 'LGBTQI_3words_NEG'] = sentiment_dict['neg'] * 100
        exp1_df.loc[index, 'LGBTQI_3words_POS'] = sentiment_dict['pos'] * 100
        exp1_df.loc[index, 'LGBTQI_3words_NEU'] = sentiment_dict['neu'] * 100
        exp1_df.loc[index, 'LGBTQI_3words_COM'] = sentiment_dict['compound'] * 100

        exp1_df.loc[index, 'allText'] = exp1_df.loc[index, 'allText'] + " " + exp1_df.loc[index, 'LGBTQI_en']

    if ( exp1_df.notnull().loc[index, 'blackPete'] ):
        # translate to both English and Dutch (if original in English)
        exp1_df.loc[index, 'blackPete_en'] = my_translator.translate(exp1_df['blackPete'][index])
        exp1_df.loc[index, 'blackPete'] = my_nl_translator.translate(exp1_df['blackPete'][index])
        # get sentiment for both English and Dutch into new columns
        sentiment_dict = sentimentAnalysisVader.polarity_scores(exp1_df['blackPete_en'][index])
        exp1_df.loc[index, 'blackPete_en_NEG'] = sentiment_dict['neg'] * 100
        exp1_df.loc[index, 'blackPete_en_POS'] = sentiment_dict['pos'] * 100
        exp1_df.loc[index, 'blackPete_en_NEU'] = sentiment_dict['neu'] * 100
        exp1_df.loc[index, 'blackPete_en_COM'] = sentiment_dict['compound'] * 100
        sentiment_dict = sentimentAnalysisVader.polarity_scores(exp1_df['blackPete_3words_en'][index])
        exp1_df.loc[index, 'blackPete_3words_NEG'] = sentiment_dict['neg'] * 100
        exp1_df.loc[index, 'blackPete_3words_POS'] = sentiment_dict['pos'] * 100
        exp1_df.loc[index, 'blackPete_3words_NEU'] = sentiment_dict['neu'] * 100
        exp1_df.loc[index, 'blackPete_3words_COM'] = sentiment_dict['compound'] * 100

        exp1_df.loc[index, 'allText'] = exp1_df.loc[index, 'allText'] + " " + exp1_df.loc[index, 'blackPete_en']

    if ( exp1_df.notnull().loc[index, 'terrorism '] ):
        # translate to both English and Dutch (if original in English)
        exp1_df.loc[index, 'terrorism_en'] = my_translator.translate(exp1_df['terrorism '][index])
        exp1_df.loc[index, 'terrorism '] = my_nl_translator.translate(exp1_df['terrorism '][index])
        # get sentiment for both English and Dutch into new columns
        sentiment_dict = sentimentAnalysisVader.polarity_scores(exp1_df['terrorism_en'][index])
        exp1_df.loc[index, 'terrorism_en_NEG'] = sentiment_dict['neg'] * 100
        exp1_df.loc[index, 'terrorism_en_POS'] = sentiment_dict['pos'] * 100
        exp1_df.loc[index, 'terrorism_en_NEU'] = sentiment_dict['neu'] * 100
        exp1_df.loc[index, 'terrorism_en_COM'] = sentiment_dict['compound'] * 100
        sentiment_dict = sentimentAnalysisVader.polarity_scores(exp1_df['terrorism_3words_en'][index])
        exp1_df.loc[index, 'terrorism_3words_NEG'] = sentiment_dict['neg'] * 100
        exp1_df.loc[index, 'terrorism_3words_POS'] = sentiment_dict['pos'] * 100
        exp1_df.loc[index, 'terrorism_3words_NEU'] = sentiment_dict['neu'] * 100
        exp1_df.loc[index, 'terrorism_3words_COM'] = sentiment_dict['compound'] * 100

        exp1_df.loc[index, 'allText'] = exp1_df.loc[index, 'allText'] + " " + exp1_df.loc[index, 'terrorism_en']

    if ( exp1_df.notnull().loc[index, 'climateChange'] ):
        # translate to both English and Dutch (if original in English)
        exp1_df.loc[index, 'climateChange_en'] = my_translator.translate(exp1_df['climateChange'][index])
        exp1_df.loc[index, 'climateChange'] = my_translator.translate(exp1_df['climateChange'][index])
        # get sentiment for both English and Dutch into new columns
        sentiment_dict = sentimentAnalysisVader.polarity_scores(exp1_df['climateChange_en'][index])
        exp1_df.loc[index, 'climateChange_en_NEG'] = sentiment_dict['neg'] * 100
        exp1_df.loc[index, 'climateChange_en_POS'] = sentiment_dict['pos'] * 100
        exp1_df.loc[index, 'climateChange_en_NEU'] = sentiment_dict['neu'] * 100
        exp1_df.loc[index, 'climateChange_en_COM'] = sentiment_dict['compound'] * 100
        sentiment_dict = sentimentAnalysisVader.polarity_scores(exp1_df['climateChange_3words_en'][index])
        exp1_df.loc[index, 'climateChange_3words_NEG'] = sentiment_dict['neg'] * 100
        exp1_df.loc[index, 'climateChange_3words_POS'] = sentiment_dict['pos'] * 100
        exp1_df.loc[index, 'climateChange_3words_NEU'] = sentiment_dict['neu'] * 100
        exp1_df.loc[index, 'climateChange_3words_COM'] = sentiment_dict['compound'] * 100

        exp1_df.loc[index, 'allText'] = exp1_df.loc[index, 'allText'] + " " + exp1_df.loc[index, 'climateChange_en']

    if ( exp1_df.notnull().loc[index, 'developmentAid'] ):
        # translate to both English and Dutch (if original in English)
        exp1_df.loc[index, 'developmentAid_en'] = my_translator.translate(exp1_df['developmentAid'][index])
        exp1_df.loc[index, 'developmentAid'] = my_translator.translate(exp1_df['developmentAid'][index])
        # get sentiment for both English and Dutch into new columns
        sentiment_dict = sentimentAnalysisVader.polarity_scores(exp1_df['developmentAid_en'][index])
        exp1_df.loc[index, 'developmentAid_en_NEG'] = sentiment_dict['neg'] * 100
        exp1_df.loc[index, 'developmentAid_en_POS'] = sentiment_dict['pos'] * 100
        exp1_df.loc[index, 'developmentAid_en_NEU'] = sentiment_dict['neu'] * 100
        exp1_df.loc[index, 'developmentAid_en_COM'] = sentiment_dict['compound'] * 100
        sentiment_dict = sentimentAnalysisVader.polarity_scores(exp1_df['developmentAid_3words_en'][index])
        exp1_df.loc[index, 'developmentAid_3words_NEG'] = sentiment_dict['neg'] * 100
        exp1_df.loc[index, 'developmentAid_3words_POS'] = sentiment_dict['pos'] * 100
        exp1_df.loc[index, 'developmentAid_3words_NEU'] = sentiment_dict['neu'] * 100
        exp1_df.loc[index, 'developmentAid_3words_COM'] = sentiment_dict['compound'] * 100

        exp1_df.loc[index, 'allText'] = exp1_df.loc[index, 'allText'] + " " + exp1_df.loc[index, 'developmentAid_en']

    if ( exp1_df.notnull().loc[index, 'EU '] ):
        # translate to both English and Dutch (if original in English)
        exp1_df.loc[index, 'EU_en'] = my_translator.translate(exp1_df['EU '][index])
        exp1_df.loc[index, 'EU '] = my_translator.translate(exp1_df['EU '][index])
        # get sentiment for both English and Dutch into new columns
        sentiment_dict = sentimentAnalysisVader.polarity_scores(exp1_df['EU_en'][index])
        exp1_df.loc[index, 'EU_en_NEG'] = sentiment_dict['neg'] * 100
        exp1_df.loc[index, 'EU_en_POS'] = sentiment_dict['pos'] * 100
        exp1_df.loc[index, 'EU_en_NEU'] = sentiment_dict['neu'] * 100
        exp1_df.loc[index, 'EU_en_COM'] = sentiment_dict['compound'] * 100
        sentiment_dict = sentimentAnalysisVader.polarity_scores(exp1_df['EU_3words_en'][index])
        exp1_df.loc[index, 'EU_3words_NEG'] = sentiment_dict['neg'] * 100
        exp1_df.loc[index, 'EU_3words_POS'] = sentiment_dict['pos'] * 100
        exp1_df.loc[index, 'EU_3words_NEU'] = sentiment_dict['neu'] * 100
        exp1_df.loc[index, 'EU_3words_COM'] = sentiment_dict['compound'] * 100

        exp1_df.loc[index, 'allText'] = exp1_df.loc[index, 'allText'] + " " + exp1_df.loc[index, 'EU_en']

    if ( exp1_df.notnull().loc[index, 'Tax '] ):
        # translate to both English and Dutch (if original in English)
        exp1_df.loc[index, 'Tax_en'] = my_translator.translate(exp1_df['Tax '][index])
        exp1_df.loc[index, 'Tax '] = my_translator.translate(exp1_df['Tax '][index])
        # get sentiment for both English and Dutch into new columns
        sentiment_dict = sentimentAnalysisVader.polarity_scores(exp1_df['Tax_en'][index])
        exp1_df.loc[index, 'Tax_en_NEG'] = sentiment_dict['neg'] * 100
        exp1_df.loc[index, 'Tax_en_POS'] = sentiment_dict['pos'] * 100
        exp1_df.loc[index, 'Tax_en_NEU'] = sentiment_dict['neu'] * 100
        exp1_df.loc[index, 'Tax_en_COM'] = sentiment_dict['compound'] * 100
        sentiment_dict = sentimentAnalysisVader.polarity_scores(exp1_df['Tax_3words_en'][index])
        exp1_df.loc[index, 'Tax_3words_NEG'] = sentiment_dict['neg'] * 100
        exp1_df.loc[index, 'Tax_3words_POS'] = sentiment_dict['pos'] * 100
        exp1_df.loc[index, 'Tax_3words_NEU'] = sentiment_dict['neu'] * 100
        exp1_df.loc[index, 'Tax_3words_COM'] = sentiment_dict['compound'] * 100

        exp1_df.loc[index, 'allText'] = exp1_df.loc[index, 'allText'] + " " + exp1_df.loc[index, 'Tax_en']

    if ( exp1_df.notnull().loc[index, 'healthcare'] ):
        # translate to both English and Dutch (if original in English)
        exp1_df.loc[index, 'healthcare_en'] = my_translator.translate(exp1_df['healthcare'][index])
        exp1_df.loc[index, 'healthcare'] = my_translator.translate(exp1_df['healthcare'][index])
        # get sentiment for both English and Dutch into new columns
        sentiment_dict = sentimentAnalysisVader.polarity_scores(exp1_df['healthcare_en'][index])
        exp1_df.loc[index, 'healthcare_en_NEG'] = sentiment_dict['neg'] * 100
        exp1_df.loc[index, 'healthcare_en_POS'] = sentiment_dict['pos'] * 100
        exp1_df.loc[index, 'healthcare_en_NEU'] = sentiment_dict['neu'] * 100
        exp1_df.loc[index, 'healthcare_en_COM'] = sentiment_dict['compound'] * 100
        sentiment_dict = sentimentAnalysisVader.polarity_scores(exp1_df['healthcare_3words_en'][index])
        exp1_df.loc[index, 'healthcare_3words_NEG'] = sentiment_dict['neg'] * 100
        exp1_df.loc[index, 'healthcare_3words_POS'] = sentiment_dict['pos'] * 100
        exp1_df.loc[index, 'healthcare_3words_NEU'] = sentiment_dict['neu'] * 100
        exp1_df.loc[index, 'healthcare_3words_COM'] = sentiment_dict['compound'] * 100

        exp1_df.loc[index, 'allText'] = exp1_df.loc[index, 'allText'] + " " + exp1_df.loc[index, 'healthcare_en']

    tempDf = pd.DataFrame( [exp1_df.loc[index, 'allText']] )
    tempDf.to_csv('emfdTemp.csv', index=False, header=False)
    tempDf = pd.read_csv('emfdTemp.csv', header=None)
    length = len( tempDf )
    eMFD_df = score_docs(tempDf, 'emfd', 'single', 'bow', 'vice-virtue', length)
    exp1_df.loc[index, 'care.virtue'] = eMFD_df['care.virtue'].values[0]
    exp1_df.loc[index, 'fairness.virtue'] = eMFD_df['fairness.virtue'].values[0]
    exp1_df.loc[index, 'loyalty.virtue'] = eMFD_df['loyalty.virtue'].values[0]
    exp1_df.loc[index, 'authority.virtue'] = eMFD_df['authority.virtue'].values[0]
    exp1_df.loc[index, 'sanctity.virtue'] = eMFD_df['sanctity.virtue'].values[0]
    exp1_df.loc[index, 'care.vice'] = eMFD_df['care.vice'].values[0]
    exp1_df.loc[index, 'fairness.vice'] = eMFD_df['fairness.vice'].values[0]
    exp1_df.loc[index, 'loyalty.vice'] = eMFD_df['loyalty.vice'].values[0]
    exp1_df.loc[index, 'authority.vice'] = eMFD_df['authority.vice'].values[0]
    exp1_df.loc[index, 'sanctity.vice'] = eMFD_df['sanctity.vice'].values[0]
    exp1_df.loc[index, 'moral_nonmoral_ratio'] = eMFD_df['moral_nonmoral_ratio'].values[0]
    exp1_df.loc[index, 'f_var'] = eMFD_df['f_var'].values[0]

exp1_df.to_excel( "exp1_sentimentAndMACd.xlsx")
#save to new excel
