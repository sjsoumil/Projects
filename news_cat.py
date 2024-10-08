import asyncio
import aiohttp
import feedparser
import logging
import datetime
import hashlib
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, UniqueConstraint, Index
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base  

from celery import Celery
from celery.schedules import crontab
import spacy

# Database Configuration
DATABASE_URL = "postgresql+psycopg2://sj_user:sj_sj@localhost/news_db"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
Base = declarative_base()

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Celery configuration
app = Celery('news_categorization_app', broker='redis://localhost:6379/0')
app.conf.beat_schedule = {
    'parse_rss_every_hour': {
        'task': 'news_categorization_app.parse_rss',
        'schedule': crontab(minute=0, hour='*')  # Every hour
    },
    'categorize_articles_every_hour': {
        'task': 'news_categorization_app.categorize_articles',
        'schedule': crontab(minute=0, hour='*')  # Every hour
    }
}

# Load spaCy model for NLP
nlp = spacy.load("en_core_web_sm")

# Define categories and keywords for classification
CATEGORIES = {
    "Terrorism / protest / political unrest / riot": ["protest", "riot", "unrest", "terrorism"],
    "Positive/Uplifting": ["positive", "uplifting", "achievement", "innovation"],
    "Natural Disasters": ["earthquake", "flood", "hurricane", "disaster", "wildfire"]
}

# RSS Feeds to parse
FEEDS = [
    "http://rss.cnn.com/rss/cnn_topstories.rss",
    "http://qz.com/feed",
    "http://feeds.foxnews.com/foxnews/politics",
    "http://feeds.reuters.com/reuters/businessNews",
    "http://feeds.feedburner.com/NewshourWorld",
    "https://feeds.bbci.co.uk/news/world/asia/india/rss.xml"
]

# Define News Article model for database
class NewsArticle(Base):
    __tablename__ = 'news_articles'
    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    publication_date = Column(DateTime, default=datetime.datetime.utcnow)
    source_url = Column(String, unique=True)
    content_hash = Column(String, unique=True)
    category = Column(String, default='Uncategorized')
    UniqueConstraint('title', 'source_url', name='unique_article')
    Index('content_hash_idx', 'content_hash')  # Index for content hash

# Create the database table
try:
    Base.metadata.create_all(engine)
except Exception as e:
    logging.error(f"Error creating database tables: {e}")

# Function to classify articles based on keywords
def classify_article(content):
    doc = nlp(content)
    for category, keywords in CATEGORIES.items():
        if any(keyword in doc.text.lower() for keyword in keywords):
            return category
    return "Others"

# Generate a hash for article content to detect duplicates
def get_content_hash(content):
    return hashlib.md5(content.encode()).hexdigest()

# Celery task to parse RSS feeds and store articles in the database
@app.task
async def parse_rss():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_feed(session, url) for url in FEEDS]
        results = await asyncio.gather(*tasks)

    db_session = Session()
    articles = []
    for feed_data in results:
        if not feed_data:  # Skip if feed data is empty
            logging.warning("Received empty feed data, skipping.")
            continue
        feed = feedparser.parse(feed_data)
        logging.info(f"Parsing feed: {feed_data}")  # Added logging
        for entry in feed.entries:
            title = entry.title if 'title' in entry else None
            # Check for both description and summary attributes
            content = getattr(entry, 'description', None) or getattr(entry, 'summary', None)
            pub_date = datetime.datetime(*entry.published_parsed[:6]) if 'published_parsed' in entry else datetime.datetime.utcnow()
            source_url = entry.link if 'link' in entry else None
            
            if title is None or content is None or source_url is None:
                logging.warning("Missing title, content, or source URL. Skipping entry.")
                continue

            content_hash = get_content_hash(content)

            # Avoid duplicate entries by checking content hash
            if db_session.query(NewsArticle).filter_by(content_hash=content_hash).first():
                logging.info(f"Duplicate article found: {title}, skipping.")
                continue
            
            # Add new article to the list
            article = NewsArticle(
                title=title,
                content=content,
                publication_date=pub_date,
                source_url=source_url,
                content_hash=content_hash
            )
            articles.append(article)

    db_session.bulk_save_objects(articles)
    db_session.commit()
    db_session.close()
    logging.info(f"Parsed and saved {len(articles)} articles from RSS feeds.")

# Async helper function to fetch feed data
async def fetch_feed(session, url):
    try:
        async with session.get(url) as response:
            response.raise_for_status()  # Raise an error for bad responses
            return await response.text()
    except Exception as e:
        logging.error(f"Error fetching {url}: {e}")
        return ""

# Celery task to categorize uncategorized articles in the database
@app.task
def categorize_articles():
    db_session = Session()
    articles = db_session.query(NewsArticle).filter_by(category='Uncategorized').all()
    
    logging.info(f"Categorizing {len(articles)} uncategorized articles.")  # Added logging
    
    for article in articles:
        category = classify_article(article.content)
        article.category = category
        db_session.commit()
        logging.info(f"Article '{article.title}' categorized as {category}")

    db_session.close()

# Utility function to export data to CSV
def export_data_to_csv():
    import pandas as pd
    import os

    # Specify the path where you want to save the CSV file
    csv_file_path = os.path.join("C:\\Users\\91825\\Desktop\\Ten", 'news_articles.csv')

    db_session = Session()
    articles = db_session.query(NewsArticle).all()
    
    data = [{
        'title': article.title,
        'content': article.content,
        'publication_date': article.publication_date,
        'source_url': article.source_url,
        'category': article.category
    } for article in articles]

    df = pd.DataFrame(data)
    df.to_csv(csv_file_path, index=False)  # Save the CSV to the specified path
    db_session.close()
    logging.info(f"Data exported to {csv_file_path}.")

# Entry point for testing purposes
if __name__ == "__main__":
    logging.info("Starting the RSS parsing and categorization process...")
    asyncio.run(parse_rss())  # Run the parse_rss task for testing
    categorize_articles()  # Run the categorize_articles task for testing
    export_data_to_csv()  # Export data to CSV
