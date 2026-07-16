FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /code

# 3. Copy only the requirements first (optimizes Docker build caching)
COPY ./requirements.txt /code/requirements.txt

# 4. Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 5. Copy the 'models' and 'src' folders into the container
COPY ./models /code/models
COPY ./src /code/src

# 6. Expose the port FastAPI runs on
EXPOSE 8000

# 7. Command to run the application using Uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]