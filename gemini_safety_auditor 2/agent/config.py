"""
Config loader for the Gemini Safety Auditor.

Never hardcode your API key. Set it as an environment variable before running:

    export GEMINI_API_KEY="your-key-here"

Or create a .env file in the project root:

    GEMINI_API_KEY=your-key-here

and load it with python-dotenv (see requirements.txt).
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed -- fine, we'll just read from the real
    # environment instead. Install with: pip install python-dotenv
    pass


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Model IDs -- pin these explicitly rather than relying on aliases, since
# Google periodically retires/redirects preview names. Google's model
# lineup moves fast (see https://ai.google.dev/gemini-api/docs/changelog);
# if a default here ever 404s, check that page before assuming your code
# is broken.
#
# NOTE (updated for this project's model refresh): the original defaults
# here -- gemini-2.5-flash and text-embedding-004 -- are on Google's
# deprecation list (2.5-flash is slated to shut down Oct 2026;
# text-embedding-004 has *already* been shut down as of this update). Both
# defaults below have been bumped forward accordingly.
GENERATION_MODEL = os.environ.get("GEMINI_GENERATION_MODEL", "gemini-3.6-flash")
EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

# Second tier, used only by agent/cross_tier.py (Phase 5) to compare a
# cheaper/faster model against a more capable one on the same prompts.
# GENERATION_MODEL above is treated as the "fast" tier for that comparison;
# this is the "capable" tier. Still a Preview model ID as of this writing
# (Google's own naming, not a typo) -- swap it for whatever's GA when you
# run this.
GENERATION_MODEL_CAPABLE = os.environ.get("GEMINI_GENERATION_MODEL_CAPABLE", "gemini-3.1-pro-preview")

API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def require_api_key(override: str | None = None) -> str:
    """`override` lets a caller (e.g. the Streamlit app, where each visitor
    may supply their own key at runtime) supply a key explicitly instead of
    relying on the process-wide environment variable -- important since a
    shared Streamlit process serves multiple sessions concurrently."""
    key = override or GEMINI_API_KEY
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Get one at https://aistudio.google.com "
            "(click 'Get API key'), then set it as an environment variable "
            "or in a .env file before running this project."
        )
    return key
