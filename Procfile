release: python manage.py migrate
web: daphne Datathon.asgi:application --port $PORT --bind 0.0.0.0 -v2
celery: celery -A Datathon.celery worker --pool=solo -l info
celerybeat: celery -A Datathon beat -l info