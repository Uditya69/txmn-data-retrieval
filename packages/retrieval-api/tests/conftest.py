import os

os.environ.setdefault(
    "MONGO_URI",
    "mongodb://localhost:27017/?serverSelectionTimeoutMS=2000&connectTimeoutMS=2000",
)
os.environ.setdefault("MONGO_DB", "test-auth-db")
os.environ.setdefault("JWT_SECRET", "test-jwt-secret")
os.environ.setdefault("JWT_EXPIRY_MINUTES", "60")
