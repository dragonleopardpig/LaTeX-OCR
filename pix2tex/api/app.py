import asyncio
import os
import secrets
from contextlib import asynccontextmanager
from http import HTTPStatus
from io import BytesIO
from typing import Annotated

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse

from pix2tex.cli import LatexOCR

model = None
inference_slot = asyncio.Semaphore(1)
MAX_UPLOAD_BYTES = int(os.environ.get('PIX2TEX_MAX_UPLOAD_BYTES', 10 * 1024 * 1024))
MAX_IMAGE_PIXELS = int(os.environ.get('PIX2TEX_MAX_IMAGE_PIXELS', 20_000_000))
ALLOWED_FORMATS = {'PNG', 'JPEG', 'WEBP', 'BMP', 'TIFF'}


class RequestBodyTooLarge(Exception):
    pass


class ContentSizeLimitMiddleware:
    """Reject oversized bodies before multipart parsing can spool them to disk."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get('headers', ()))
        content_length = headers.get(b'content-length')
        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    await self._reject(scope, receive, send)
                    return
            except ValueError:
                await JSONResponse(
                    {'detail': 'Invalid Content-Length header'},
                    status_code=HTTPStatus.BAD_REQUEST,
                )(scope, receive, send)
                return

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message['type'] == 'http.request':
                received += len(message.get('body', b''))
                if received > self.max_bytes:
                    raise RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except RequestBodyTooLarge:
            await self._reject(scope, receive, send)

    async def _reject(self, scope, receive, send):
        await JSONResponse(
            {'detail': f'Request exceeds the {self.max_bytes}-byte body limit'},
            status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        )(scope, receive, send)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model
    if model is None:
        model = LatexOCR()
    yield


app = FastAPI(title='pix2tex API', lifespan=lifespan)
app.add_middleware(ContentSizeLimitMiddleware, max_bytes=MAX_UPLOAD_BYTES + 1024 * 1024)


def require_api_key(x_api_key: str | None = None) -> None:
    """Require X-API-Key only when PIX2TEX_API_KEY is configured."""
    expected = os.environ.get('PIX2TEX_API_KEY')
    if expected and (x_api_key is None or not secrets.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=HTTPStatus.UNAUTHORIZED, detail='Invalid API key')


async def read_upload(file: UploadFile) -> bytes:
    content_type = (file.content_type or '').lower()
    if content_type and not content_type.startswith('image/'):
        raise HTTPException(
            status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            detail='Only image uploads are accepted',
        )
    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
            detail=f'Image exceeds the {MAX_UPLOAD_BYTES}-byte upload limit',
        )
    return content


def decode_image(content: bytes) -> Image.Image:
    if not content:
        raise HTTPException(status_code=HTTPStatus.UNPROCESSABLE_ENTITY, detail='Empty upload')
    try:
        with Image.open(BytesIO(content)) as source:
            if source.format not in ALLOWED_FORMATS:
                raise HTTPException(
                    status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    detail=f'Unsupported image format: {source.format}',
                )
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    detail=f'Image exceeds the {MAX_IMAGE_PIXELS}-pixel limit',
                )
            source.load()
            return source.convert('RGB')
    except HTTPException:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as error:
        raise HTTPException(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            detail='The upload is not a valid image',
        ) from error


def run_prediction(image: Image.Image, resize: bool) -> str:
    if model is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail='Model is not ready',
        )
    return model(image, resize=resize, copy_to_clipboard=False)


async def predict_upload(file: UploadFile, resize: bool = True) -> str:
    content = await read_upload(file)
    image = decode_image(content)
    async with inference_slot:
        return await run_in_threadpool(run_prediction, image, resize)


@app.get('/')
def root():
    '''Health check.'''
    response = {
        'message': HTTPStatus.OK.phrase,
        'status-code': HTTPStatus.OK,
        'data': {},
    }
    return response


@app.post('/predict/')
async def predict(
    file: Annotated[UploadFile, File()],
    _authorized: Annotated[str | None, Header(alias='X-API-Key')] = None,
) -> str:
    """Predict the Latex code from an image file.

    Args:
        file (UploadFile, optional): Image to predict. Defaults to File(...).

    Returns:
        str: Latex prediction
    """
    require_api_key(_authorized)
    return await predict_upload(file)


@app.post('/bytes/', deprecated=True)
async def predict_from_bytes(
    file: Annotated[UploadFile, File()],
    _authorized: Annotated[str | None, Header(alias='X-API-Key')] = None,
) -> str:
    """Predict the Latex code from a byte array

    Args:
        file (bytes, optional): Image as byte array. Defaults to File(...).

    Returns:
        str: Latex prediction
    """
    require_api_key(_authorized)
    return await predict_upload(file, resize=False)
