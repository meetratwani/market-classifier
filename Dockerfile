FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install -r requirements.txt

RUN mkdir -p /var/log/supervisor /var/run

COPY supervisord.conf /etc/supervisor/supervisord.conf

EXPOSE 5000
EXPOSE 8080

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]