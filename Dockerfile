FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[live]"

RUN useradd --create-home engine && mkdir /app/data && chown engine /app/data
USER engine
VOLUME /app/data

ENTRYPOINT ["pmengine"]
CMD ["run", "--config", "/app/config.yaml"]
