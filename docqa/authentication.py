from django.conf import settings
from django.utils import timezone
from rest_framework import exceptions
from rest_framework.authentication import TokenAuthentication


class ExpiringTokenAuthentication(TokenAuthentication):
    """Tokens stop working after TOKEN_TTL_DAYS (0 = never); the user signs in again for a fresh one."""

    def authenticate_credentials(self, key):
        user, token = super().authenticate_credentials(key)
        ttl = settings.TOKEN_TTL_DAYS
        if ttl and token.created + timezone.timedelta(days=ttl) < timezone.now():
            token.delete()
            raise exceptions.AuthenticationFailed("Token expired. Sign in again.")
        return user, token
