FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir shiny pandas scikit-learn==1.6.1 matplotlib shap joblib jinja2

EXPOSE 7860

CMD ["shiny", "run", "--host", "0.0.0.0", "--port", "7860", "app/app.py"]