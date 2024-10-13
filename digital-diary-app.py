import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sqlalchemy import create_engine, Column, Integer, String, Date, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from datetime import datetime
import plotly.express as px
import base64
from PIL import Image
import io

# Set page config for better mobile experience
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for better mobile experience
st.markdown("""
<style>
    .reportview-container .main .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stTextInput>div>div>input,
    .stSelectbox>div>div>select {
        min-height: 44px;
    }
    @media (max-width: 768px) {
        .reportview-container .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ... [rest of the imports and database setup remain the same]

def show_dashboard():
    try:
        session = Session()
        entries = session.query(Entry).order_by(Entry.date.desc()).limit(100).all()
        
        if not entries:
            st.info("No entries found. Start by adding some entries!")
            return
        
        df = pd.DataFrame([(entry.date, entry.mood, entry.experience) for entry in entries], 
                          columns=['date', 'mood', 'experience'])
        
        # Mood trend
        st.write("Mood Trend (Last 100 Entries)")
        fig_mood = px.bar(df['mood'].value_counts().reset_index(), x='index', y='mood', labels={'index': 'Mood', 'mood': 'Count'})
        fig_mood.update_layout(height=400)
        st.plotly_chart(fig_mood, use_container_width=True)
        
        # Entries per day
        st.write("Entries per Day (Last 100 Entries)")
        entries_per_day = df.groupby('date').size().reset_index(name='count')
        fig_entries = px.line(entries_per_day, x='date', y='count', labels={'date': 'Date', 'count': 'Entries'})
        fig_entries.update_layout(height=400)
        st.plotly_chart(fig_entries, use_container_width=True)
        
        # Common phrases
        st.write("Common Phrases (Last 100 Entries)")
        all_text = ' '.join(df['experience'])
        tokens = word_tokenize(all_text.lower())
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        freq_dist = nltk.FreqDist(filtered_tokens)
        common_words = pd.DataFrame(freq_dist.most_common(10), columns=['word', 'count'])
        fig_words = px.bar(common_words, x='word', y='count', labels={'word': 'Word', 'count': 'Count'})
        fig_words.update_layout(height=400)
        st.plotly_chart(fig_words, use_container_width=True)
    
    except OperationalError:
        st.error("Unable to access the database. Please make sure it's initialized.")
    except SQLAlchemyError as e:
        st.error(f"A database error occurred: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
    finally:
        session.close()

def show_memory_pad():
    try:
        session = Session()
        entries_per_page = 5
        total_entries = session.query(Entry).count()
        total_pages = (total_entries - 1) // entries_per_page + 1
        
        col1, col2, col3 = st.columns([1,3,1])
        with col2:
            page = st.slider("Page", min_value=1, max_value=total_pages, value=1)
        
        entries = session.query(Entry).order_by(Entry.date.desc()).offset((page-1)*entries_per_page).limit(entries_per_page).all()
        
        if not entries:
            st.info("No memories found. Start by adding some entries!")
            return
        
        for entry in entries:
            with st.expander(f"Entry from {entry.date}"):
                st.write(f"Mood: {entry.mood}")
                st.write(f"Experience: {entry.experience}")
                st.write(f"Mode: {entry.mode}")
                if entry.image:
                    try:
                        image = Image.open(io.BytesIO(base64.b64decode(entry.image)))
                        st.image(image, use_column_width=True)
                    except Exception as e:
                        st.error(f"Error loading image: {str(e)}")
    
    except OperationalError:
        st.error("Unable to access the database. Please make sure it's initialized.")
    except SQLAlchemyError as e:
        st.error(f"A database error occurred: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
    finally:
        session.close()

def show_mood_memory():
    try:
        session = Session()
        entries = session.query(Entry).all()
        
        if not entries:
            st.info("No entries found. Start by adding some entries!")
            return
        
        moods = list(set([entry.mood for entry in entries]))
        selected_mood = st.selectbox("Select Mood", moods)
        
        entries_per_page = 5
        total_entries = session.query(Entry).filter(Entry.mood == selected_mood).count()
        total_pages = (total_entries - 1) // entries_per_page + 1
        
        col1, col2, col3 = st.columns([1,3,1])
        with col2:
            page = st.slider("Page", min_value=1, max_value=total_pages, value=1)
        
        filtered_entries = session.query(Entry).filter(Entry.mood == selected_mood).order_by(Entry.date.desc()).offset((page-1)*entries_per_page).limit(entries_per_page).all()
        
        if not filtered_entries:
            st.info(f"No entries found for the mood: {selected_mood}")
            return
        
        for entry in filtered_entries:
            with st.expander(f"Entry from {entry.date}"):
                st.write(f"Experience: {entry.experience}")
                st.write(f"Mode: {entry.mode}")
                if entry.image:
                    try:
                        image = Image.open(io.BytesIO(base64.b64decode(entry.image)))
                        st.image(image, use_column_width=True)
                    except Exception as e:
                        st.error(f"Error loading image: {str(e)}")
    
    except OperationalError:
        st.error("Unable to access the database. Please make sure it's initialized.")
    except SQLAlchemyError as e:
        st.error(f"A database error occurred: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
    finally:
        session.close()

def show_prediction():
    try:
        session = Session()
        entries = session.query(Entry).all()
        
        if not entries:
            st.info("Not enough data for prediction. Start by adding some entries!")
            return
        
        df = pd.DataFrame([(entry.mode, entry.experience) for entry in entries], 
                          columns=['mode', 'experience'])
        
        sia = SentimentIntensityAnalyzer()
        df['sentiment'] = df['experience'].apply(lambda x: sia.polarity_scores(x)['compound'])
        df['sentiment_category'] = df['sentiment'].apply(lambda x: 'Positive' if x > 0.05 else ('Negative' if x < -0.05 else 'Neutral'))
        
        mode_sentiment = df.groupby('mode')['sentiment_category'].value_counts(normalize=True).unstack().reset_index()
        mode_sentiment_long = pd.melt(mode_sentiment, id_vars=['mode'], var_name='sentiment', value_name='percentage')
        
        st.write("Prediction of Experience Quality by Mode")
        fig = px.bar(mode_sentiment_long, x='mode', y='percentage', color='sentiment', barmode='group')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    except OperationalError:
        st.error("Unable to access the database. Please make sure it's initialized.")
    except SQLAlchemyError as e:
        st.error(f"A database error occurred: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
    finally:
        session.close()

def add_new_entry():
    try:
        session = Session()
        
        date = st.date_input("Date")
        experience = st.text_area("Experience", height=150)
        mood = st.selectbox("Mood", ["Happy", "Sad", "Excited", "Angry", "Neutral"])
        mode = st.selectbox("Mode of Experience", ["Work", "Personal", "Travel", "Social", "Other"])
        image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        
        if st.button("Add Entry"):
            if not experience:
                st.error("Please enter your experience.")
                return
            
            new_entry = Entry(date=date, experience=experience, mood=mood, mode=mode)
            if image:
                try:
                    img = Image.open(image)
                    img.thumbnail((800, 800))  # Resize image to max 800x800
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    new_entry.image = img_str
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    return
            
            session.add(new_entry)
            session.commit()
            st.success("Entry Added Successfully")
    
    except OperationalError:
        st.error("Unable to access the database. Please make sure it's initialized.")
    except SQLAlchemyError as e:
        st.error(f"A database error occurred: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
    finally:
        session.close()

# ... [main function remains the same]
