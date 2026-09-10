FROM python:3.11-slim
WORKDIR /latexocr
COPY pix2tex /latexocr/pix2tex/
COPY pyproject.toml setup.py README.md LICENSE /latexocr/
RUN useradd --create-home --uid 10001 pix2tex \
    && chown -R pix2tex:pix2tex /home/pix2tex
ENV PIX2TEX_CACHE_DIR=/home/pix2tex/.cache/pix2tex
RUN pip install --no-cache-dir '.[api]'

USER pix2tex
RUN python -m pix2tex.model.checkpoints.get_latest_checkpoint

EXPOSE 8502

ENTRYPOINT ["uvicorn", "pix2tex.api.app:app", "--host", "0.0.0.0", "--port", "8502"]
