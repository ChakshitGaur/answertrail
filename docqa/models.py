import json

from django.conf import settings
from django.db import models


class EmbeddingField(models.Field):
    """A vector: `vector(N)` on PostgreSQL (pgvector), JSON text elsewhere (SQLite)."""

    def db_type(self, connection):
        if connection.vendor == "postgresql":
            return f"vector({settings.EMBEDDING_DIM})"
        return "text"

    def get_prep_value(self, value):
        if value is None:
            return None
        # '[0.1,0.2]' is valid pgvector input and valid JSON
        return json.dumps([float(x) for x in value])

    def from_db_value(self, value, expression, connection):
        if value is None:
            return None
        return json.loads(value) if isinstance(value, str) else list(value)


class Document(models.Model):
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="documents"
    )
    title = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return self.title


class Chunk(models.Model):
    document = models.ForeignKey(
        Document, on_delete=models.CASCADE, related_name="chunks"
    )
    index = models.PositiveIntegerField()
    text = models.TextField()
    embedding = EmbeddingField()

    class Meta:
        ordering = ["document_id", "index"]
        constraints = [
            models.UniqueConstraint(
                fields=["document", "index"], name="unique_chunk_index_per_document"
            )
        ]
