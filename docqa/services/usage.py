from django.conf import settings
from django.db import IntegrityError, transaction
from django.db.models import F
from django.utils import timezone

from docqa.models import DailyUsage

LIMIT_MESSAGE = "Daily question limit reached ({limit} questions that use the language model). It resets at midnight UTC."


def try_spend(user) -> bool:
    """Count one model question for today; False (and nothing counted) if the daily limit is already used up."""
    limit = settings.DAILY_MODEL_QUESTIONS
    if not limit:
        return True
    today = timezone.now().date()
    try:  # create today's row once; a concurrent request that loses the race just uses the existing row
        with transaction.atomic():
            DailyUsage.objects.get_or_create(user=user, day=today)
    except IntegrityError:
        pass
    # one conditional UPDATE, so two requests at the limit cannot both get in
    return DailyUsage.objects.filter(user=user, day=today, questions__lt=limit).update(questions=F("questions") + 1) == 1
