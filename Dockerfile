# base image
FROM python:3.11-slim

#set working directory
WORKDIR /app

# copy requirements first (caching)
COPY requirements.txt .

#install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy all projects files
COPY . .

# expose port
EXPOSE 8000

# run the api
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]