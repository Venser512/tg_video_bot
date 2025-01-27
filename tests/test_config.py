
import pytest
from config import bot_token, bot_user_id, OPENROUTER_API_KEY, PROJECT_PATH, conf_checkpoint_path

def test_config():        
    assert len(bot_token) > 10
    assert len(bot_user_id) > 5
    assert bot_user_id in bot_token
    assert len(OPENROUTER_API_KEY) > 20
    assert len(conf_checkpoint_path) > 10

