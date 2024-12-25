import os
import time
from datetime import timedelta, datetime

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.stemmers import Stemmer
from collections import defaultdict
from sumy.utils import get_stop_words

import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scaler = MinMaxScaler()


formats = [
    '%B %d, %Y — %I:%M %p',  # "September 12, 2023 — 06:15 pm"
    '%b %d, %Y %I:%M%p',  # "Nov 14, 2023 7:35AM"
    '%d-%b-%y',  # "6-Jan-22"
    '%Y-%m-%d',  # "2021-4-5"
    '%Y/%m/%d',  # "2021/4/5"
    '%b %d, %Y'  # "DEC 7, 2023"
]

def convert_to_utc_datetime(time_str, formats=formats):
    if "EDT" in time_str:
        time_str_cleaned = time_str.replace(' EDT','')
        offset = timedelta(hours=-4)
    elif "EST" in time_str:
        time_str_cleaned = time_str.replace(' EST','')
        offset =timedelta(hours=-5)
    else:
        time_str_cleaned = time_str
        offset = timedelta(hours=0)


    for fmt in formats:
        try:
            dt = datetime.strptime(time_str_cleaned, fmt)
            # if the date contains only dates without time
            if fmt == '%d-%b-%y':
                offset = timedelta(hours=0)

            dt_utc = dt + offset

            return dt_utc.strftime('%Y-%m-%d %H:%M:%S UTC')

        except ValueError:
            continue

    # if there is no corresponding date format, return an error
    return "Invalid date format"

def df_date_convert(folder_path, saving_path):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    for csv_file in csv_files:
        print('Starting: ' + csv_file)
        file_path = os.path.join(folder_path,csv_file)

        df_temp = pd.read_csv(file_path, on_bad_lines="warn")
        df_temp.columns = df_temp.columns.str.lower()

        if "datetime" in df_temp.columns:
            df_temp.rename({'datetime':'date'}, inplace=True)

        df_temp['date'] = df_temp['date'].apply(convert_to_utc_datetime)

        df_temp['date'] = pd.to_datetime(df_temp['date'], utc=True)
        df_temp = df_temp.sort_values(by='date', ascending=False)

        symbol = csv_file.split(".")[0].lower()

        # df_temp.to_csv(os.path.join(saving_path, csv_file), index=False)
        # print('DONE: ', csv_file)
        return df_temp, symbol


# Boosts the importance of sentences containing specific keywords.
def increase_weight_for_key_words(sentences, key_words):
    sentence_weights = defaultdict(float)

    for sentence in sentences:
        for word in key_words:
            if word.lower() in str(sentence).lower():
                sentence_weights[sentence] += 1
    return sentence_weights

# Creates summaries by combining an initial LSA-based summary with keyword-weighted sentences.
def new_sum(text, key_words, num_sentences):

    stemmer = Stemmer("english")
    summarizer = LsaSummarizer(stemmer)
    tokenizer = Tokenizer("english")
    summarizer.stop_words = get_stop_words("english")

    parser = PlaintextParser.from_string(text, tokenizer)
    initial_summary = summarizer(parser.document, num_sentences)

    # Increase weight
    sentence_weights = increase_weight_for_key_words(parser.document.sentences, key_words)

    # Combine weights from initial summary with additional weights
    for sentence in initial_summary:
        sentence_weights[sentence] += 1  # Initial summary sentences get additional weight

    # Select top sentences as final summary
    final_summary = sorted(sentence_weights, key=sentence_weights.get, reverse=True)[:num_sentences]

    # Output final summary
    final_summary_text = " ".join(str(sentence) for sentence in final_summary)

    return final_summary_text

# Processes multiple CSV files, generates summaries for text columns, cleans the data, and saves the results.
def from_csv_summarize(folder_path, saving_path, df, symbol):

    # csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
        a = time.time()

    # for csv_file in csv_files:
    #     print(csv_file)
    #     file_path = os.path.join(folder_path, csv_file)

    #     try:
    #         df = pd.read_csv(file_path, encoding="utf-8")
    #     except UnicodeDecodeError:
    #         df = pd.read_csv(file_path, encoding="Windows-1252")
    #     symbol = csv_file.split(".")[0].lower()

        # AAPL - dataset is used
        df.columns = df.columns.str.lower()
        key_words_value = {symbol}
        num_sentences_value = 3

        df['new_text'] = df['text'].apply(new_sum, key_words=key_words_value, num_sentences=num_sentences_value)

        # Drop the old text column
        df = df.drop(columns=['text'])

        # Keep only those rows that are not empty
        df = df[df['mark'] == 1]
        print(time.time()-a, "s") # Timing of the whole proccess

        # df.to_csv(os.path.join(saving_path, symbol.lower()+".csv"), index=False)
        return df

def predict_sentiment(text, model, tokenizer, device):
    model.eval()  # Put the model in evaluation mode
    with torch.no_grad():
        # Tokenize and prepare the input
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Get model predictions
        outputs = model(input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=-1).cpu().item()

        # Map prediction back to label
        label_map = {0: "neutral", 1: "positive", 2: "negative"}
        return label_map[predictions]


def from_csv_sentiment(folder_path, saving_path, df, model, tokenizer, device):

    # csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    # a = time.time()

    # for csv_file in csv_files:

    #     print(csv_file)
    #     file_path = os.path.join(folder_path, csv_file)

        # try:
        #     df = pd.read_csv(file_path, encoding="utf-8")
        # except UnicodeDecodeError:
        #     df = pd.read_csv(file_path, encoding="Windows-1252")
        # symbol = csv_file.split(".")[0].lower()

        # AAPL - dataset is used
        df.columns = df.columns.str.lower()

        df['sentiment'] = df['new_text'].apply(lambda x: predict_sentiment(x, model, tokenizer, device))

        # Define a label mapping dictionary
        labels = {'neutral': 0, 'positive': 1, 'negative': 2}

        # Convert labels to integers using the mapping
        df['sentiment'] = df['sentiment'].map(labels)

        # df.to_csv(os.path.join(saving_path, symbol.lower()+".csv"), index=False)

        return df

# Localization of time
def convert_to_utc(df, date_column):
    print('Entered to convert_to_utc function')
    """
    Changing into UTC format
    """
    df[date_column] = pd.to_datetime(df[date_column])

    if df[date_column].dt.tz is None:  # Checking whether it is UTC formatted date
        df[date_column] = df[date_column].dt.tz_localize('UTC')
    return df


# The exponential decay method is preferable as the older data (news) quickly become irrelevant
def fill_missing_dates_with_exponential_decay(df, date_column, sentiment_column, decay_rate=0.05):
    print('Entered to fill_missing_dates_with_exponential_decay function')

    # Ensure that the datetime column has a correct format
    df[date_column] = pd.to_datetime(df[date_column])

    # Creating complete date range
    # Adds missing dates within the range for time-series continuity
    date_range = pd.date_range(start=df[date_column].min(), end=df[date_column].max())

    # Crating a df containing all dates
    # Combines the existing data with the full date range
    full_df = pd.DataFrame(date_range, columns=[date_column])
    full_df = pd.merge(full_df, df, on=date_column, how='left')

    # Creating a "news_flag" column
    # Flags and helps to identify the actual news data
    full_df['news_flag'] = full_df[sentiment_column].notna().astype(int)

    # Fill missing sentiment values using an exponential decay rule
    # Estimates missing sentiment values based on exponential decay over time
    last_valid_sentiment = None
    last_valid_date = None
    for i, row in full_df.iterrows():
        if pd.isna(row[sentiment_column]):
            if last_valid_sentiment is not None:
                days_since_last_valid = (row[date_column] - last_valid_date).days
                decayed_sentiment = 0
                decayed_sentiment = 3 + (last_valid_sentiment - 3) * np.exp(-decay_rate * days_since_last_valid)
                full_df.at[i, sentiment_column] = decayed_sentiment
                full_df.at[i, 'news_flag'] = 0
        else:
            last_valid_sentiment = row[sentiment_column]
            last_valid_date = row[date_column]
    return full_df


def integrate_data(stock_price_df, news_df):

    # Create a copy of the original DataFrame
    # Prevents modifying the original dataset directly
    # stock_price_df_copy = stock_price_df.copy()
    # news_df_copy = news_df.copy()

    # Convert date formats and sort the data
    # Ensures consistency and proper alignment of data

    stock_price_df = convert_to_utc(stock_price_df, 'date')
    news_df = convert_to_utc(news_df, 'date')

    stock_price_df['date'] = pd.to_datetime(stock_price_df['date'])
    news_df['date'] = pd.to_datetime(news_df['date'])

    # To align datetime to the start of the day
    stock_price_df['date'] = pd.to_datetime(stock_price_df['date']).dt.normalize()
    news_df['date'] = pd.to_datetime(news_df['date']).dt.normalize()

    stock_price_df.set_index('date', inplace=True)
    news_df.set_index('date', inplace=True)

    # Sorting the dates
    stock_price_df.sort_index(inplace=True)
    news_df.sort_index(inplace=True)

    average_sentiment = news_df.groupby('date')['sentiment'].mean().reset_index()

    # Fill missing dates using an exponential decay rule
    # Ensures sentiment continuity with decay for missing days
    average_sentiment_filled = fill_missing_dates_with_exponential_decay(average_sentiment, 'date', 'sentiment')

    # Merge the data
    # Combines stock prices with news sentiment data for modeling
    merged_df = pd.merge(stock_price_df, average_sentiment_filled, on='date', how='left')

    # Replace NaN values with 3
    # Sets a neutral sentiment default where no data exists
    merged_df['sentiment'].fillna(0, inplace=True)

    # Drop rows with missing News_flag
    # Cleans up rows without proper sentiment values
    df_cleaned = merged_df.dropna(subset=['news_flag'])

    # Filter out rows where sentiment is 0
    # Removes invalid sentiment data for analysis
    df_cleaned = df_cleaned[df_cleaned['sentiment'] != 0]
    # Values are scaled to fall approximately in the range [0, 1]
    df_cleaned['scaled_sentiment'] = df_cleaned['sentiment'].apply(lambda x: (x - 0.9999) / 4)


    df_cleaned.columns.str.lower()
    print(len(df_cleaned['close']))
    if len(df_cleaned['close']) < 333:
        print("Lower than 333")
        return 0, df_cleaned
    
    print(df_cleaned)


    return 1, df_cleaned


def start_inte(df_news, df_price, symbol, saving_path):

        _, merged_data = integrate_data(df_price, df_news)


        symbol+= ".csv"
        merged_data.to_csv(os.path.join(saving_path, symbol), index=False)


def scale_data(data) -> pd.DataFrame:

    data[['close', 'volume']] = scaler.fit_transform(data[['close', 'volume']])
    return data

#prepare data
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length, 0]  # Close price as target
        xs.append(x)
        ys.append(y)
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)

