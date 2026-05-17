# superset_config.py

# This tells Superset exactly where the Postgres DB is
SQLALCHEMY_DATABASE_URI = 'postgresql://admin:mlops_pass@postgres:5432/superset_db'

# A secret key for session cookies
SECRET_KEY = 'something_very_secret_and_random_wikan'

# This allows you to see more detailed errors in the UI
DEBUG = True