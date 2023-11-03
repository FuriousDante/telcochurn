FROM python:3.11.1

WORKDIR /telcodir
COPY . /telcodir
EXPOSE 8501
RUN pip install -r requirements.txt
CMD streamlit run server.py
