FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY gateway/requirements.txt gateway/requirements.lock /app/
RUN pip install --no-cache-dir -r requirements.lock \
    && pip check \
    && groupadd --gid 10001 gateway \
    && useradd --uid 10001 --gid gateway --no-create-home gateway
COPY gateway/server.py /app/server.py
USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--no-proxy-headers"]
