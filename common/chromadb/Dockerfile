FROM python:3.11-slim-bookworm

WORKDIR /chroma

RUN pip install chromadb

COPY scripts/ /app/scripts/

# Fix line endings and ensure executable permissions
RUN apt-get update && apt-get install -y dos2unix \
    && dos2unix /app/scripts/startup.sh \
    && chmod +x /app/scripts/startup.sh

EXPOSE 8000

# Ensure the startup script is executable
RUN chmod +x /app/scripts/startup.sh

# Use bash to execute the script instead of direct execution
CMD ["/bin/bash", "/app/scripts/startup.sh"]