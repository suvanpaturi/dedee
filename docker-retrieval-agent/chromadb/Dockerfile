FROM python:3.9-slim

WORKDIR /chroma

RUN pip install chromadb

COPY scripts/ /app/scripts/

# Ensure the startup script is executable
RUN chmod +x /app/scripts/startup.sh

EXPOSE 8000

CMD ["/app/scripts/startup.sh"]
