FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY deploy/cloud-center/mcp/cloud_center_mcp ./cloud_center_mcp

EXPOSE 8101

CMD ["python", "-m", "cloud_center_mcp.server"]
