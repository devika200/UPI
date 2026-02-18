"""Database connection and utilities"""
import logging
from urllib.parse import urlparse, quote_plus, urlunparse
from pymongo import MongoClient
import certifi

logger = logging.getLogger(__name__)


class Database:
    """MongoDB database manager"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.users_collection = None
        self.transactions_collection = None
    
    def init_app(self, app):
        """Initialize database with Flask app"""
        mongodb_uri = self._safe_mongodb_uri(app.config['MONGODB_URI'])
        
        try:
            self.client = MongoClient(
                mongodb_uri,
                serverSelectionTimeoutMS=app.config['MONGODB_TIMEOUT'],
                tls=True,
                tlsAllowInvalidCertificates=False,
                tlsAllowInvalidHostnames=False,
                tlsCAFile=certifi.where()
            )
            self.client.server_info()  # Test connection
            
            self.db = self.client[app.config['MONGODB_DB_NAME']]
            self.users_collection = self.db['users']
            self.transactions_collection = self.db['transactions']
            
            logger.info("MongoDB connected successfully")
            return True
            
        except Exception as e:
            logger.warning(f"MongoDB connection failed: {e}. Running without database.")
            self.client = None
            self.db = None
            self.users_collection = None
            self.transactions_collection = None
            return False
    
    @staticmethod
    def _safe_mongodb_uri(uri):
        """Escape username/password in URI per RFC 3986"""
        if not uri or "@" not in uri or "://" not in uri:
            return uri
        
        parsed = urlparse(uri)
        if parsed.username is not None or parsed.password is not None:
            netloc = parsed.hostname or ""
            if parsed.port is not None:
                netloc = f"{netloc}:{parsed.port}"
            
            user = quote_plus(parsed.username) if parsed.username is not None else ""
            passwd = quote_plus(parsed.password) if parsed.password is not None else ""
            
            if user or passwd:
                netloc = f"{user}:{passwd}@{netloc}" if passwd else f"{user}@{netloc}"
            
            parsed = parsed._replace(netloc=netloc)
            return urlunparse(parsed)
        
        return uri
    
    def is_connected(self):
        """Check if database is connected"""
        return self.client is not None


# Global database instance
db = Database()
