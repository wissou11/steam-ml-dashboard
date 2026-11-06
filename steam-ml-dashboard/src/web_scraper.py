# web_scraper.py

import requests
from bs4 import BeautifulSoup
import json
import time
import csv
from datetime import datetime
import random
from typing import List, Dict, Optional, Union, Callable
import logging
from functools import wraps
from abc import ABC, abstractmethod
import pandas as pd

# ==================== EXCEPTIONS ====================

class SteamScraperError(Exception):
    """Base exception for Steam scraper errors."""
    pass

class APIRequestError(SteamScraperError):
    """Raised when API requests fail."""
    pass

class RateLimitError(SteamScraperError):
    """Raised when hitting rate limits."""
    pass

class InvalidGameDataError(SteamScraperError):
    """Raised when game data is invalid or missing."""
    pass

class NetworkError(SteamScraperError):
    """Raised for network-related errors."""
    pass

# ==================== DECORATORS ====================

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry failed operations."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (requests.RequestException, APIRequestError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                        logging.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__}")
                    continue
            raise last_exception
        return wrapper
    return decorator

def delay_execution(min_delay: float = 1.0, max_delay: float = 3.0):
    """Decorator to add random delay between calls."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            time.sleep(random.uniform(min_delay, max_delay))
            return func(*args, **kwargs)
        return wrapper
    return decorator

def validate_response(func: Callable):
    """Decorator to validate API responses."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        response = func(*args, **kwargs)
        if response.status_code == 429:
            raise RateLimitError(f"Rate limited: HTTP {response.status_code}")
        if response.status_code >= 500:
            raise APIRequestError(f"Server error: HTTP {response.status_code}")
        response.raise_for_status()
        return response
    return wrapper

# ==================== INTERFACES ====================

class ISteamDataProvider(ABC):
    """Interface for Steam data providers."""
    
    @abstractmethod
    def get_game_details(self, app_id: int) -> Optional[Dict]:
        pass
    
    @abstractmethod
    def get_game_reviews(self, app_id: int, num_reviews: int = 100) -> Optional[Dict]:
        pass

class IDataSaver(ABC):
    """Interface for data saving strategies."""
    
    @abstractmethod
    def save(self, data: Dict, app_id: int) -> bool:
        pass

# ==================== DATA MODELS ====================

class GameDetails:
    """Data model for game details."""
    
    def __init__(self, data: Dict):
        self.app_id = data.get('app_id')
        self.name = data.get('name')
        self.type = data.get('type')
        self.is_free = data.get('is_free', False)
        self.price_initial = data.get('price_initial', 0)
        self.price_final = data.get('price_final', 0)
        self.discount_percent = data.get('discount_percent', 0)
        self.release_date = data.get('release_date')
        self.coming_soon = data.get('coming_soon', False)
        self.developers = data.get('developers', '')
        self.detailed_description_length = data.get('detailed_description_length', 0)
        self.about_game_length = data.get('about_game_length', 0)
        self.short_description = data.get('short_description', '')
        self.windows = data.get('windows', False)
        self.mac = data.get('mac', False)
        self.linux = data.get('linux', False)
        self.categories = data.get('categories', '')
        self.category_count = data.get('category_count', 0)
        self.genres = data.get('genres', '')
        self.genre_count = data.get('genre_count', 0)
        self.metacritic_score = data.get('metacritic_score')
        self.recommendations = data.get('recommendations')
        self.dlc_count = data.get('dlc_count', 0)
        self.achievements_total = data.get('achievements_total', 0)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

class ReviewSummary:
    """Data model for review summary."""
    
    def __init__(self, data: Dict):
        self.app_id = data.get('app_id')
        self.review_score = data.get('review_score')
        self.review_score_desc = data.get('review_score_desc')
        self.total_positive = data.get('total_positive')
        self.total_negative = data.get('total_negative')
        self.total_reviews = data.get('total_reviews')
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

# ==================== DATA SAVERS ====================

class CSVSaver(IDataSaver):
    """Save data to CSV file."""
    
    def __init__(self, filename: str, checkpoint_file: str = 'checkpoint.txt'):
        self.filename = filename
        self.checkpoint_file = checkpoint_file
        self.scraped_ids = self._load_checkpoint()
    
    def _load_checkpoint(self) -> set:
        """Load already scraped IDs from checkpoint file."""
        try:
            with open(self.checkpoint_file, 'r') as f:
                return {int(line.strip()) for line in f}
        except FileNotFoundError:
            return set()
    
    def save(self, data: Dict, app_id: int) -> bool:
        """Save data and update checkpoint."""
        try:
            # Write to CSV
            with open(self.filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not self.scraped_ids:
                    writer.writeheader()
                writer.writerow(data)
            
            # Update checkpoint
            with open(self.checkpoint_file, 'a') as cf:
                cf.write(f"{app_id}\n")
            
            self.scraped_ids.add(app_id)
            return True
            
        except Exception as e:
            logging.error(f"Error saving data for app {app_id}: {e}")
            return False

class DataFrameSaver(IDataSaver):
    """Save data to pandas DataFrame (for in-memory processing)."""
    
    def __init__(self):
        self.data = []
        self.scraped_ids = set()
    
    def save(self, data: Dict, app_id: int) -> bool:
        self.data.append(data)
        self.scraped_ids.add(app_id)
        return True
    
    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data)

# ==================== MAIN SCRAPER ====================

class SteamMLDataScraper(ISteamDataProvider):
    """Main scraper class with OOP design and error handling."""
    
    def __init__(self, delay_range: tuple = (1, 3), max_retries: int = 3):
        self.base_url = "https://store.steampowered.com"
        self.api_url = "https://store.steampowered.com/api"
        self.delay_range = delay_range
        self.max_retries = max_retries
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'application/json'
        }
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
    
    def _safe_join(self, items: List, delimiter: str = ',') -> str:
        """Safely join list items."""
        if not items:
            return ''
        return delimiter.join(str(item) for item in items)
    
    @delay_execution(min_delay=1.0, max_delay=3.0)
    def _delay(self):
        """Add random delay between requests."""
        pass
    
    @validate_response
    @retry_on_failure(max_retries=3)
    def _make_request(self, url: str) -> requests.Response:
        """Make HTTP request with validation and retry logic."""
        return self.session.get(url, headers=self.headers)
    
    def get_all_app_ids(self) -> List[int]:
        """Get all Steam app IDs."""
        url = f"{self.api_url}/ISteamApps/GetAppList/v2/"
        
        try:
            response = self._make_request(url)
            data = response.json()
            
            app_list = data['applist']['apps']
            app_ids = [app['appid'] for app in app_list if app.get('name')]
            self.logger.info(f"Retrieved {len(app_ids)} app IDs")
            return app_ids
            
        except (APIRequestError, RateLimitError, KeyError) as e:
            self.logger.error(f"Error getting app list: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return []
    
    @retry_on_failure(max_retries=3)
    def get_game_details(self, app_id: int) -> Optional[GameDetails]:
        """Get detailed information for a specific game."""
        url = f"{self.api_url}/appdetails?appids={app_id}&cc=us&l=en"
        
        try:
            response = self._make_request(url)
            data = response.json()
            
            if not data.get(str(app_id), {}).get('success'):
                return None
            
            game = data[str(app_id)]['data']
            
            game_data = {
                'app_id': app_id,
                'name': game.get('name'),
                'type': game.get('type'),
                'is_free': game.get('is_free', False),
                'price_initial': game.get('price_overview', {}).get('initial', 0) / 100 
                               if game.get('price_overview') else 0,
                'price_final': game.get('price_overview', {}).get('final', 0) / 100 
                             if game.get('price_overview') else 0,
                'discount_percent': game.get('price_overview', {}).get('discount_percent', 0),
                'release_date': game.get('release_date', {}).get('date'),
                'coming_soon': game.get('release_date', {}).get('coming_soon', False),
                'developers': self._safe_join(game.get('developers', [])),
                'detailed_description_length': len(game.get('detailed_description', '')),
                'about_game_length': len(game.get('about_the_game', '')),
                'short_description': game.get('short_description', ''),
                'windows': game.get('platforms', {}).get('windows', False),
                'mac': game.get('platforms', {}).get('mac', False),
                'linux': game.get('platforms', {}).get('linux', False),
                'categories': self._safe_join([cat.get('description', '') 
                                             for cat in game.get('categories', [])]),
                'category_count': len(game.get('categories', [])),
                'genres': self._safe_join([g.get('description', '') 
                                         for g in game.get('genres', [])]),
                'genre_count': len(game.get('genres', [])),
                'metacritic_score': game.get('metacritic', {}).get('score'),
                'recommendations': game.get('recommendations', {}).get('total'),
                'dlc_count': len(game.get('dlc', [])),
                'achievements_total': game.get('achievements', {}).get('total', 0),
            }
            
            return GameDetails(game_data)
            
        except (APIRequestError, RateLimitError) as e:
            self.logger.error(f"API error for app {app_id}: {e}")
            raise
        except KeyError as e:
            self.logger.error(f"Missing data key for app {app_id}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error for app {app_id}: {e}")
            return None
    
    @retry_on_failure(max_retries=3)
    def get_game_reviews(self, app_id: int, num_reviews: int = 100) -> Optional[ReviewSummary]:
        """Get review summary for a game."""
        url = f"{self.base_url}/appreviews/{app_id}?json=1&num_per_page={min(num_reviews, 100)}&language=all"
        
        try:
            response = self._make_request(url)
            data = response.json()
            
            if data.get('success') != 1:
                return None
            
            reviews_data = {
                'app_id': app_id,
                'review_score': data.get('query_summary', {}).get('review_score'),
                'review_score_desc': data.get('query_summary', {}).get('review_score_desc'),
                'total_positive': data.get('query_summary', {}).get('total_positive'),
                'total_negative': data.get('query_summary', {}).get('total_negative'),
                'total_reviews': data.get('query_summary', {}).get('total_reviews'),
            }
            
            return ReviewSummary(reviews_data)
            
        except (APIRequestError, RateLimitError) as e:
            self.logger.error(f"API error for app {app_id} reviews: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error getting reviews for app {app_id}: {e}")
            return None
    
    @retry_on_failure(max_retries=2)
    def get_game_review_texts(self, app_id: int, max_reviews: int = 50) -> List[Dict]:
        """Get detailed review texts for a game."""
        reviews = []
        cursor = "*"
        collected = 0

        try:
            while collected < max_reviews:
                url = (f"{self.base_url}/appreviews/{app_id}?json=1"
                       f"&filter=all&language=english&day_range=3650"
                       f"&review_type=all&purchase_type=all"
                       f"&num_per_page=100&cursor={cursor}")

                response = self._make_request(url)
                data = response.json()

                if "reviews" not in data:
                    break

                for r in data["reviews"]:
                    if collected >= max_reviews:
                        break

                    reviews.append({
                        "review_text": r.get("review", "").strip(),
                        "voted_up": r.get("voted_up"),
                        "votes_helpful": r.get("votes_helpful", 0),
                        "votes_funny": r.get("votes_funny", 0),
                        "timestamp_created": r.get("timestamp_created"),
                    })

                    collected += 1

                cursor = data.get("cursor")
                if not cursor:
                    break

                self._delay()

            return reviews

        except (APIRequestError, RateLimitError) as e:
            self.logger.error(f"API error for app {app_id} review texts: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error scraping review texts for app {app_id}: {e}")
            return []
    
    @delay_execution(min_delay=1.0, max_delay=2.0)
    def scrape_top_games_by_tags(self, tags: List[str], games_per_tag: int = 50) -> List[int]:
        """Scrape top games by tags."""
        all_app_ids = []
        
        for tag in tags:
            url = f"{self.base_url}/search/?tags={tag}&category1=998"
            
            try:
                response = self._make_request(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                games = soup.find_all('a', class_='search_result_row', limit=games_per_tag)
                
                for game in games:
                    app_id = game.get('data-ds-appid')
                    if app_id:
                        all_app_ids.append(int(app_id))
                
            except Exception as e:
                self.logger.error(f"Error scraping tag {tag}: {e}")
                continue
        
        return list(set(all_app_ids))
    
    def scrape_dataset(self, 
                      app_ids: List[int], 
                      saver: IDataSaver,
                      batch_size: int = 10,
                      max_games: int = None) -> None:
        """Scrape dataset using the specified data saver."""
        
        if max_games:
            app_ids = app_ids[:max_games]
        
        games_processed = 0
        total_games = len(app_ids)
        
        for idx, app_id in enumerate(app_ids, 1):
            try:
                self.logger.info(f"Processing game {idx}/{total_games}: {app_id}")
                
                game_details = self.get_game_details(app_id)
                
                if not game_details or game_details.type != 'game':
                    self.logger.info(f"Skipping non-game or invalid app: {app_id}")
                    continue
                
                self._delay()
                
                review_summary = self.get_game_reviews(app_id)
                review_texts = self.get_game_review_texts(app_id, max_reviews=50)
                
                # Combine all data
                combined_data = game_details.to_dict()
                if review_summary:
                    combined_data.update(review_summary.to_dict())
                combined_data["review_texts"] = json.dumps(review_texts, ensure_ascii=False)
                
                # Save data
                success = saver.save(combined_data, app_id)
                
                if success:
                    games_processed += 1
                    self.logger.info(f"Successfully saved data for app {app_id}")
                
                if games_processed >= batch_size and idx < total_games:
                    self.logger.info(f"Batch completed. Taking a short break...")
                    time.sleep(5)
                    games_processed = 0
                
            except (RateLimitError, APIRequestError) as e:
                self.logger.error(f"Critical error for app {app_id}: {e}. Waiting...")
                time.sleep(30)
                continue
            except Exception as e:
                self.logger.error(f"Error processing app {app_id}: {e}")
                continue
        
        self.logger.info("Scraping completed!")