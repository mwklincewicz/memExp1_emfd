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

exp1_df = pd.read_excel("./8_8_articles.xlsx")

my_translator = GoogleTranslator(source='auto', target='en')
my_nl_translator = GoogleTranslator(source='auto', target='nl')

sentimentAnalysisVader = SentimentIntensityAnalyzer()

for index, text in exp1_df.iterrows():
    print(index)
    exp1_df.loc[index, 'allText'] = my_translator.translate(exp1_df['Dutch_text'][index])

    if (exp1_df.notnull().loc[index, 'allText']):
        # get sentiment for both English and Dutch into new columns
        sentiment_dict = sentimentAnalysisVader.polarity_scores(exp1_df['allText'][index])
        exp1_df.loc[index, 'allText_en_NEG'] = sentiment_dict['neg'] * 100
        exp1_df.loc[index, 'allText_en_POS'] = sentiment_dict['pos'] * 100
        exp1_df.loc[index, 'allText_en_NEU'] = sentiment_dict['neu'] * 100
        exp1_df.loc[index, 'allText_en_COM'] = sentiment_dict['compound'] * 100

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

exp1_df.to_excel( "8_8_articles_sentimentMFTAndMACd.xlsx")
#save to new excel
