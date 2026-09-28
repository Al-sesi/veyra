#!/usr/bin/env python3
"""
Warmup script for AI models.

This script can be called by monitoring services or deployment scripts
to ensure AI models are loaded and ready, reducing cold start latency.

Usage:
    python scripts/warmup_models.py

Or via HTTP:
    curl https://your-service.onrender.com/warmup
"""

import logging
import sys
import time
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("veyra.warmup")


def warmup_models():
    """Warmup all AI models (English, Hausa, Igbo, Yoruba)."""
    try:
        from app.asr import load_essential_models, get_model_status

        logger.info("Starting AI model warmup (all languages)...")
        start_time = time.time()

        # Load all models with reduced retries for faster warmup
        load_status = load_essential_models(max_retries=2, retry_delay=1.0)

        elapsed = time.time() - start_time
        model_status = get_model_status()

        logger.info(f"Warmup completed in {elapsed:.1f}s")
        logger.info(f"Load status: {load_status}")
        logger.info(f"Model status: {model_status}")

        if load_status["failed"]:
            logger.warning(f"Some models failed to load: {load_status['failed']}")
            return False
        else:
            logger.info("All models loaded successfully")
            return True

    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        return False


if __name__ == "__main__":
    success = warmup_models()
    sys.exit(0 if success else 1)
