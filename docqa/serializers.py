from django.contrib.auth import get_user_model
from django.contrib.auth.password_validation import validate_password
from rest_framework import serializers

from docqa.models import Document


class RegisterSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=150)
    password = serializers.CharField(write_only=True)

    def validate_username(self, value):
        if get_user_model().objects.filter(username__iexact=value).exists():
            raise serializers.ValidationError("Username already taken.")
        return value

    def validate_password(self, value):
        validate_password(value)
        return value


class DocumentSerializer(serializers.ModelSerializer):
    chunk_count = serializers.SerializerMethodField()

    def get_chunk_count(self, doc):
        total = getattr(doc, "chunk_total", None)  # annotated by the list view; avoids one query per document
        return doc.chunks.count() if total is None else total

    class Meta:
        model = Document
        fields = ["id", "title", "created_at", "chunk_count"]


class DocumentUploadSerializer(serializers.Serializer):
    title = serializers.CharField(max_length=255, required=False)
    file = serializers.FileField(required=False)
    text = serializers.CharField(required=False)

    def validate(self, attrs):
        if bool(attrs.get("file")) == bool(attrs.get("text")):
            raise serializers.ValidationError("Provide exactly one of 'file' or 'text'.")
        if "file" not in attrs and "title" not in attrs:
            raise serializers.ValidationError({"title": "Required when sending raw text."})
        return attrs


class HistoryItemSerializer(serializers.Serializer):
    role = serializers.ChoiceField(choices=["user", "assistant"])
    content = serializers.CharField(max_length=4000, allow_blank=True)


class AskSerializer(serializers.Serializer):
    question = serializers.CharField(max_length=2000)
    document_id = serializers.IntegerField(required=False)
    top_k = serializers.IntegerField(required=False, min_value=1, max_value=10, default=4)
    # documents: only from your files (default for API callers); auto: files when relevant, otherwise
    # general knowledge; general: ignore the files and answer like a normal assistant
    mode = serializers.ChoiceField(choices=["auto", "documents", "general"], default="documents")
    history = HistoryItemSerializer(many=True, required=False, max_length=30, default=list)
