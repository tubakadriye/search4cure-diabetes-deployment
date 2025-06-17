FROM python:3.10.14-bullseye

WORKDIR /app

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    locales \
    locales-all && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy all files into the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Streamlit runs on port 8501 by default
EXPOSE 8501

# Healthcheck to confirm container is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

#docker run -p 8501:8501 search4cure-diabetes
# docker login
#docker tag search4cure-diabetes tubakadriye/search4cure-diabetes:latest
#docker push tubakadriye/search4cure-diabetes:latest