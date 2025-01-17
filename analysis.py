import spacy
#has to be spacy 3.4
#typing-extensions has to be 4.4
print("spaCy version:", spacy.__version__)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
import pandas as pd
#has to be pandas 1.5.3
#also install scikit-learn 1.3
#also install openpyxl
print("pandas version:", pd.__version__)
from emfdscore.scoring import score_docs as emfd_score_docs
from emacscore.scoring import score_docs as emac_score_docs

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 400)

#exp1_df = pd.read_excel("./test.xlsx")

expA_df = pd.read_excel("./rawdata_exp1.xlsx")
expB_df = pd.read_excel("./rawdata_exp2.xlsx")

exp1_df = pd.merge(expA_df, expB_df, how="outer")

my_translator = GoogleTranslator(source='auto', target='en')
my_nl_translator = GoogleTranslator(source='auto', target='nl')

sentimentAnalysisVader = SentimentIntensityAnalyzer()
for index, text in exp1_df.iterrows():
    print(index)
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

    if (exp1_df.notnull().loc[index, 'Q22']):
        # translate to both English and Dutch (if original in English)
        exp1_df.loc[index, 'recentEvent_en'] = my_translator.translate(exp1_df['Q22'][index])
        exp1_df.loc[index, 'recentEvent'] = my_nl_translator.translate(exp1_df['Q22'][index])
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
# Parse with MFT dictionary and MAC dictionary
    tempDf = pd.DataFrame( [exp1_df.loc[index, 'allText']] )
    tempDf.to_csv('emfdTemp.csv', index=False, header=False)
    tempDf = pd.read_csv('emfdTemp.csv', header=None)
    length = len( tempDf )
    eMFD_df = emfd_score_docs(tempDf, 'emfd', 'single', 'bow', 'vice-virtue', length)
    eMAC_df = emac_score_docs(tempDf, 'emac', 'single', 'bow', 'vice-virtue', length)
    eMFD_df_all = emfd_score_docs(tempDf, 'emfd', 'all', 'bow', 'vice-virtue', length)
    eMAC_df_all = emac_score_docs(tempDf, 'emac', 'all', 'bow', 'vice-virtue', length)
# MFT virtue-vice bow per word
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
# MAC virtue-vice bow per word
    exp1_df.loc[index, 'macFairness.virtue'] = eMAC_df['fairness.virtue'].values[0]
    exp1_df.loc[index, 'macGroup.virtue'] = eMAC_df['group.virtue'].values[0]
    exp1_df.loc[index, 'macDeference.virtue'] = eMAC_df['deference.virtue'].values[0]
    exp1_df.loc[index, 'macHeroism.virtue'] = eMAC_df['heroism.virtue'].values[0]
    exp1_df.loc[index, 'macReciprocity.virtue'] = eMAC_df['reciprocity.virtue'].values[0]
    exp1_df.loc[index, 'macFamily.virtue'] = eMAC_df['family.virtue'].values[0]
    exp1_df.loc[index, 'macProperty.virtue'] = eMAC_df['property.virtue'].values[0]
    exp1_df.loc[index, 'macFairness.vice'] = eMAC_df['fairness.vice'].values[0]
    exp1_df.loc[index, 'macGroup.vice'] = eMAC_df['group.vice'].values[0]
    exp1_df.loc[index, 'macDeference.vice'] = eMAC_df['deference.vice'].values[0]
    exp1_df.loc[index, 'macHeroism.vice'] = eMAC_df['heroism.vice'].values[0]
    exp1_df.loc[index, 'macReciprocity.vice'] = eMAC_df['reciprocity.vice'].values[0]
    exp1_df.loc[index, 'macFamily.vice'] = eMAC_df['family.vice'].values[0]
    exp1_df.loc[index, 'macProperty.vice'] = eMAC_df['property.vice'].values[0]
    exp1_df.loc[index, 'macF_var'] = eMAC_df['f_var'].values[0]
# MFT virtue-vice bow all
    exp1_df.loc[index, 'care.virtueAll'] = eMFD_df_all['care.virtue'].values[0]
    exp1_df.loc[index, 'fairness.virtueAll'] = eMFD_df_all['fairness.virtue'].values[0]
    exp1_df.loc[index, 'loyalty.virtueAll'] = eMFD_df_all['loyalty.virtue'].values[0]
    exp1_df.loc[index, 'authority.virtueAll'] = eMFD_df_all['authority.virtue'].values[0]
    exp1_df.loc[index, 'sanctity.virtueAll'] = eMFD_df_all['sanctity.virtue'].values[0]
    exp1_df.loc[index, 'care.viceAll'] = eMFD_df_all['care.vice'].values[0]
    exp1_df.loc[index, 'fairness.viceAll'] = eMFD_df_all['fairness.vice'].values[0]
    exp1_df.loc[index, 'loyalty.viceAll'] = eMFD_df_all['loyalty.vice'].values[0]
    exp1_df.loc[index, 'authority.viceAll'] = eMFD_df_all['authority.vice'].values[0]
    exp1_df.loc[index, 'sanctity.viceAll'] = eMFD_df_all['sanctity.vice'].values[0]
    exp1_df.loc[index, 'moral_nonmoral_ratioAll'] = eMFD_df_all['moral_nonmoral_ratio'].values[0]
    exp1_df.loc[index, 'f_varAll'] = eMFD_df_all['f_var'].values[0]
# MAC virtue-vice bow all
    exp1_df.loc[index, 'macFairness.virtueAll'] = eMAC_df_all['fairness.virtue'].values[0]
    exp1_df.loc[index, 'macGroup.virtueAll'] = eMAC_df_all['group.virtue'].values[0]
    exp1_df.loc[index, 'macDeference.virtueAll'] = eMAC_df_all['deference.virtue'].values[0]
    exp1_df.loc[index, 'macHeroism.virtueAll'] = eMAC_df_all['heroism.virtue'].values[0]
    exp1_df.loc[index, 'macReciprocity.virtueAll'] = eMAC_df_all['reciprocity.virtue'].values[0]
    exp1_df.loc[index, 'macFamily.virtueAll'] = eMAC_df_all['family.virtue'].values[0]
    exp1_df.loc[index, 'macProperty.virtueAll'] = eMAC_df_all['property.virtue'].values[0]
    exp1_df.loc[index, 'macFairness.viceAll'] = eMAC_df_all['fairness.vice'].values[0]
    exp1_df.loc[index, 'macGroup.viceAll'] = eMAC_df_all['group.vice'].values[0]
    exp1_df.loc[index, 'macDeference.viceAll'] = eMAC_df_all['deference.vice'].values[0]
    exp1_df.loc[index, 'macHeroism.viceAll'] = eMAC_df_all['heroism.vice'].values[0]
    exp1_df.loc[index, 'macReciprocity.viceAll'] = eMAC_df_all['reciprocity.vice'].values[0]
    exp1_df.loc[index, 'macFamily.viceAll'] = eMAC_df_all['family.vice'].values[0]
    exp1_df.loc[index, 'macProperty.viceAll'] = eMAC_df_all['property.vice'].values[0]
    exp1_df.loc[index, 'macF_varAll'] = eMAC_df_all['f_var'].values[0]

exp1_df.to_excel( "exp1_sentimentMFTAndMACd.xlsx")
#save to new excel
