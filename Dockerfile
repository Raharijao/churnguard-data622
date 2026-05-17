FROM python:3.10-slim

WORKDIR /code

COPY . .

RUN pip install --no-cache-dir \
    shiny \
    pandas \
    numpy \
    matplotlib \
    scikit-learn==1.6.1 \
    joblib \
    shap \
    xgboost \
    jinja2

EXPOSE 7860

ENV PYTHONPATH=/code

CMD ["shiny", "run", "--host", "0.0.0.0", "--port", "7860", "app/app.py"]
