import json

from django.conf import settings
from django.contrib.auth import get_user_model
from django.http import StreamingHttpResponse
from rest_framework import mixins, status, viewsets
from rest_framework.authtoken.models import Token
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from docqa.models import Document
from docqa.serializers import (
    AskSerializer,
    DocumentSerializer,
    DocumentUploadSerializer,
    RegisterSerializer,
)
from docqa.services import llm, retrieval
from docqa.services.ingest import IngestError, extract_text, ingest_document


class RegisterView(APIView):
    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request):
        ser = RegisterSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        user = get_user_model().objects.create_user(**ser.validated_data)
        token, _ = Token.objects.get_or_create(user=user)
        return Response({"token": token.key}, status=status.HTTP_201_CREATED)


class DocumentViewSet(
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    mixins.DestroyModelMixin,
    viewsets.GenericViewSet,
):
    serializer_class = DocumentSerializer
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def get_queryset(self):
        return Document.objects.filter(owner=self.request.user)

    def create(self, request):
        ser = DocumentUploadSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        upload = ser.validated_data.get("file")
        try:
            if upload:
                if upload.size > settings.MAX_UPLOAD_BYTES:
                    return Response(
                        {"detail": "File too large."},
                        status=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    )
                text = extract_text(upload.name, upload.read())
                title = ser.validated_data.get("title") or upload.name
            else:
                text = ser.validated_data["text"]
                title = ser.validated_data["title"]
            doc = ingest_document(request.user, title, text)
        except IngestError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(DocumentSerializer(doc).data, status=status.HTTP_201_CREATED)


def _sources(hits):
    return [
        {
            "n": i,
            "document_id": h.chunk.document_id,
            "document_title": h.chunk.document.title,
            "chunk_index": h.chunk.index,
            "score": round(h.score, 4),
            "text": h.chunk.text,
        }
        for i, h in enumerate(hits, start=1)
    ]


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


class AskStreamView(APIView):
    """Server-sent events: `sources`, then `token` pieces, then `done` (or `error`)."""

    def post(self, request):
        ser = AskSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        data = ser.validated_data
        doc_id = data.get("document_id")
        if doc_id is not None and not Document.objects.filter(
            pk=doc_id, owner=request.user
        ).exists():
            return Response({"detail": "Document not found."}, status=status.HTTP_404_NOT_FOUND)

        hits = retrieval.search(request.user, data["question"], data["top_k"], doc_id)
        sources = _sources(hits)  # evaluated now: the generator below must not touch the database
        passages = [h.chunk.text for h in hits]

        def events():
            yield _sse("sources", {"sources": sources})
            try:
                for piece in llm.stream_answer(data["question"], passages):
                    yield _sse("token", {"t": piece})
            except llm.LLMError as exc:
                yield _sse("error", {"detail": f"The language model request failed: {exc}"})
                return
            yield _sse("done", {})

        response = StreamingHttpResponse(events(), content_type="text/event-stream")
        response["Cache-Control"] = "no-cache"
        response["X-Accel-Buffering"] = "no"  # tell nginx not to buffer the stream
        return response


class AskView(APIView):
    def post(self, request):
        ser = AskSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        data = ser.validated_data

        doc_id = data.get("document_id")
        if doc_id is not None and not Document.objects.filter(
            pk=doc_id, owner=request.user
        ).exists():
            return Response({"detail": "Document not found."}, status=status.HTTP_404_NOT_FOUND)

        hits = retrieval.search(request.user, data["question"], data["top_k"], doc_id)
        try:
            text = llm.answer(data["question"], [h.chunk.text for h in hits])
        except llm.LLMError as exc:
            return Response(
                {"detail": f"The language model request failed: {exc}"},
                status=status.HTTP_502_BAD_GATEWAY,
            )
        return Response({"answer": text, "sources": _sources(hits)})
