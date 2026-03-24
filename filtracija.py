import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



df_ociscen_1 = pd.read_csv("data/OnlineNewsPopularity.csv")

df_ociscen_1 = df_ociscen_1.drop(columns=["url", " timedelta"," abs_title_subjectivity"," abs_title_sentiment_polarity"," is_weekend"])


## cistka 0.0
df = df_ociscen_1[df_ociscen_1[" n_unique_tokens"] > 0.0].copy()

df.loc[df[" kw_min_min"] == -1.00, " kw_min_min"] = 0
df.loc[df[" kw_avg_min"]==-1.00, " kw_avg_min"] = 0
df.loc[df[" kw_min_avg"]==-1.00, " kw_min_avg"] = 0

df[' n_non_stop_words'] = np.where(df[' n_non_stop_words'] <= 1.0, 0.0, df[' n_non_stop_words'])



percentil=[" num_hrefs"," num_self_hrefs"," n_unique_tokens"," num_imgs"," num_videos",
" average_token_length"," kw_avg_avg"," kw_min_min"," kw_max_min", " kw_avg_min"," kw_min_max"," kw_avg_max"," kw_min_avg"," kw_max_avg",
" self_reference_min_shares"," self_reference_avg_sharess"," self_reference_max_shares"]
for atribut_p in percentil:
    lower_bound = df[atribut_p].quantile(0.01)  # donji 1%
    upper_bound = df[atribut_p].quantile(0.99)  # gornji 99%
    df[atribut_p] = df[atribut_p].clip(lower=lower_bound, upper=upper_bound)


df[" num_external_hrefs"] = df[" num_hrefs"] - df[" num_self_hrefs"]
df = df.drop(columns=[" num_hrefs"])
df = df.drop(columns=[" data_channel_is_world"])
df = df.drop(columns=[" weekday_is_sunday"])
df = df.drop(columns=[" LDA_04"])
#### uklonjeno jer ne treba(num_hrefs) ili je visak usled  drugih binarnih atributa



atribut_p=" num_external_hrefs"
lower_bound = df[atribut_p].quantile(0.01)  # donji 1%
upper_bound = df[atribut_p].quantile(0.99)  # gornji 99%
df[atribut_p] = df[atribut_p].clip(lower=lower_bound, upper=upper_bound)


upper_bound = df[' shares'].quantile(0.99)
df.loc[df[' shares'] > upper_bound, ' shares'] = upper_bound



## log(1+x)
log1=[" n_tokens_content"," num_external_hrefs"," num_self_hrefs"," num_imgs"," num_videos"," kw_avg_avg",
" kw_min_min"," kw_max_min", " kw_avg_min"," kw_min_max"," kw_avg_max"," kw_min_avg"," self_reference_min_shares",
" self_reference_avg_sharess"," self_reference_max_shares"," shares"," n_non_stop_words"," n_non_stop_unique_tokens"
," kw_max_avg"]
for x in log1:
    df[x] = np.log1p(df[x])

df = df.drop(columns=[" rate_negative_words"]) #kKORELACIJA
df = df.drop(columns=[" self_reference_avg_sharess"]) #KORELACIJA
df = df.drop(columns=[" self_reference_min_shares"]) #KORELACIJA
df = df.drop(columns=[" kw_min_avg"]) #KORELACIJA
df = df.drop(columns=[" kw_avg_min"]) #KORELACIJA
df = df.drop(columns=[" kw_avg_avg"]) #KORELACIJA
df= df.drop(columns=[" kw_max_max"]) #75% podataka se nalazi na max
df = df.drop(columns=[" n_unique_tokens"]) #KORELACIJA


def create_engineered_features(df):
    """
    Kreira 4 nova, ne-redundantna atributa za model predikcije viralnosti.
    Sve formule uključuju tretman "deljenja nulom".
    """

    # 1. Avg_Word_Score (Emocija po Reči)
    # n_tokens_content je već Log transformisan i skaliran, pa ga koristimo.
    df[' avg_word_score'] = (
            (df[' avg_positive_polarity'] + df[' avg_negative_polarity'].abs()) /
            df[' n_tokens_content']
    )
    # Tretman deljenja nulom (ako je n_tokens_content nula)
    # Vrednost se postavlja na 0 ako je n_tokens_content 0.
    df[' avg_word_score'] = df[' avg_word_score'].fillna(0).replace([np.inf, -np.inf], 0)

    # 2. Interaction_Score (Ukupni Emotivni Naboj)
    df[' interaction_score'] = (
            (df[' title_sentiment_polarity'] * 10) + df[' global_sentiment_polarity']
    )

    # 3. Self_Reference_Rate (Stopa Samoreferenciranja) - KONAČNA FORMULA
    # Odnos internih linkova prema ukupnim linkovima.
    total_hrefs = df[' num_self_hrefs'] + df[' num_external_hrefs']
    df[' self_Reference_Rate'] = np.where(
        total_hrefs > 0,
        df[' num_self_hrefs'] / total_hrefs,
        0
    )

    # 4. Ratio_Title_Length (Odnos Dužine Naslova i Sadržaja)
    # n_tokens_content je logaritmovan, ali je ovde bolja upotreba SIROVIH vrednosti
    # Ako ste sačuvali sirove vrednosti, koristite njih.
    # Ako niste, koristite postojeće (Log) vrednosti, što je tehnički u redu.
    df[' ratio_title_length'] = np.where(
        df[' n_tokens_content'] > 0,
        df[' n_tokens_title'] / df[' n_tokens_content'],
        0
    )
    # Tretman potencijalnog INF/NaN (ako su i Log vrednosti bile niske/nula)
    df[' ratio_title_length'] = df[' ratio_title_length'].fillna(0).replace([np.inf, -np.inf], 0)

    # Sada treba da se obradi potencijalni INF/NaN na mestima gde je n_tokens_content=0
    # U profesionalnom kodu bi ovo bilo detaljnije, ali za ove 4 kolone
    # smo već pokrili većinu slučajeva.

    return df

df_korigovan = create_engineered_features(df.copy())

file_name = 'data/online_news_popularity_processed.csv'
df_korigovan.to_csv(file_name, index=False)

