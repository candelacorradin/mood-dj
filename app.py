import logging
import os
import random
from functools import lru_cache

import spotipy
from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("spotipy.client").setLevel(logging.WARNING)

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

auth_manager = SpotifyClientCredentials(
    client_id=client_id,
    client_secret=client_secret,
)
sp = spotipy.Spotify(
    auth_manager=auth_manager,
    requests_timeout=10,
    retries=2,
)

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    timeout=10.0,
)

REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8000/callback")
USER_CACHE = ".spotify_user_cache"

# Spotify API limits
SEARCH_LIMIT = 10  # max permitido por /v1/search


# ---------------------------------------------------------------------------
# User auth helpers
# ---------------------------------------------------------------------------

def _user_oauth() -> SpotifyOAuth:
    return SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=REDIRECT_URI,
        scope="user-top-read user-read-recently-played",
        cache_path=USER_CACHE,
        open_browser=False,
    )


def get_user_sp() -> spotipy.Spotify | None:
    oauth = _user_oauth()
    token = oauth.get_cached_token()
    if not token:
        return None
    return spotipy.Spotify(auth_manager=oauth, requests_timeout=10, retries=2)


def get_user_taste_summary() -> str | None:
    """Genres only — artist names are NOT fed into Claude's prompt because the
    model tends to overfit and just return that artist's name as the query."""
    user_sp = get_user_sp()
    if not user_sp:
        return None
    try:
        artists = user_sp.current_user_top_artists(limit=10, time_range="medium_term")["items"]
        genres: list[str] = []
        for a in artists:
            genres.extend(a.get("genres", [])[:3])
        seen: set[str] = set()
        unique = [g for g in genres if not (g in seen or seen.add(g))][:6]  # type: ignore[func-returns-value]
        return ", ".join(unique) if unique else None
    except Exception as exc:
        logger.warning(f"Could not fetch user taste: {exc}")
        return None


def get_user_seed_tracks(limit: int = 8) -> list[dict]:
    """Returns full Spotify track objects from the user's recent listening,
    used as real expansion seeds in the radio pool."""
    user_sp = get_user_sp()
    if not user_sp:
        return []
    try:
        # Short-term reflects current rotation; medium-term gives broader taste.
        short = user_sp.current_user_top_tracks(limit=limit, time_range="short_term")["items"]
        medium = user_sp.current_user_top_tracks(limit=limit, time_range="medium_term")["items"]
        out: list[dict] = []
        seen: set[str] = set()
        for t in short + medium:
            tid = t.get("id")
            if tid and tid not in seen:
                seen.add(tid)
                out.append(t)
        random.shuffle(out)
        return out[:limit]
    except Exception as exc:
        logger.warning(f"Could not fetch user seed tracks: {exc}")
        return []


def _normalize_mood(mood: str) -> str:
    return mood.lower().strip()


QUERY_COUNT = 4


@lru_cache(maxsize=500)
def interpret_mood_queries(mood: str, taste: str | None = None) -> tuple[str, ...]:
    """Return a tuple of focused Spotify search queries for the mood.

    Each query must capture the *same* mood — they vary only along genre /
    instrumentation / era to broaden catalogue coverage without diluting fit.
    Cached per (mood, taste) pair. Always call with a normalized mood.
    """
    taste_line = (
        f"\nThe listener enjoys these genres: {taste}. Lean toward those styles."
        if taste else ""
    )
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[
            {
                "role": "user",
                "content": f"""Mood: {mood}

Generate {QUERY_COUNT} Spotify search queries (2-4 words each) that all capture THIS EXACT mood.{taste_line}

Every query must clearly evoke the same feeling. Vary only by genre, instrumentation, or era — never let the mood slip.

Rules:
- Use only genre, style, energy, instrument, or feeling words.
- Each query must include at least one word that anchors the mood (not just an adjacent vibe).
- Never include artist names, song titles, or person names.
- Avoid generic filler like "music", "song", "playlist", "vibe", "mood".

Example for "melancholic rainy day":
melancholy indie folk
sad acoustic ballads
rainy day piano
slow heartbreak shoegaze

Return exactly {QUERY_COUNT} queries, one per line. No numbering, quotes, or explanations.""",
            }
        ],
    )
    raw = response.content[0].text.strip()
    queries = []
    for line in raw.split("\n"):
        q = line.strip().strip('"\'').strip("-•*0123456789. ").strip()
        if q and len(q) <= 60:
            queries.append(q)
    if not queries:
        queries = [mood]
    return tuple(queries[:QUERY_COUNT])


def filter_pool_by_mood(pool: list[dict], mood: str) -> list[dict]:
    """Pass the assembled pool through Claude and drop tracks that clearly
    don't fit the mood. Conservative — only removes obvious mismatches."""
    if len(pool) < 20:
        return pool

    listing = "\n".join(
        f"{i + 1}. {(t.get('name') or '').strip()} — {(t.get('artists') or [{}])[0].get('name', '').strip()}"
        for i, t in enumerate(pool)
    )

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": f"""Mood: "{mood}"

Below are {len(pool)} tracks pulled by keyword search for that mood. Many will fit. Some won't.

Flag tracks to REMOVE if they are:
- Obvious mood/genre mismatches (e.g. an upbeat track in a melancholy pool)
- Backing tracks, karaoke versions, instructional guitar tabs, "type beats"
- Sleep/ambient noise, white noise, rain sounds, meditation tracks (unless the mood explicitly calls for ambient)
- Loops, jingles, soundboards, sound-effect compilations
- Anything where the title/artist combo screams "not a real song for this mood"

Keep real songs that plausibly fit, even if not perfect. When borderline, lean toward removal of low-quality / non-song results.

Tracks:
{listing}

Return comma-separated numbers only (e.g. "3,7,12,18"). If everything fits, return "none".""",
                }
            ],
        )
        text = response.content[0].text.strip().lower()
        if not text or "none" in text:
            return pool

        bad: set[int] = set()
        for token in text.replace(" ", "").split(","):
            if token.isdigit():
                idx = int(token) - 1
                if 0 <= idx < len(pool):
                    bad.add(idx)

        # Safety valve: if Claude wants to nuke more than half the pool, ignore.
        if len(bad) > len(pool) // 2:
            logger.warning(
                f"Mood filter flagged {len(bad)}/{len(pool)} tracks; ignoring as over-aggressive"
            )
            return pool

        logger.info(f"Mood filter removed {len(bad)}/{len(pool)} tracks")
        return [t for i, t in enumerate(pool) if i not in bad]
    except Exception as exc:
        logger.warning(f"Mood filter failed: {exc}")
        return pool


def track_payload(track: dict) -> dict:
    return {
        "track_id": track["id"],
        "song": track["name"],
        "artist": track["artists"][0]["name"],
        "url": track["external_urls"]["spotify"],
        "duration_ms": track.get("duration_ms", 0),
        "preview_url": track.get("preview_url"),
    }


def _dedupe_tracks(tracks: list[dict], seen: set[str]) -> list[dict]:
    unique = []
    for track in tracks:
        tid = track.get("id")
        if tid and tid not in seen:
            seen.add(tid)
            unique.append(track)
    return unique


def _cap_per_artist(tracks: list[dict], max_per_artist: int = 3) -> list[dict]:
    """Stops any single artist's album/discography from flooding the pool."""
    counts: dict[str, int] = {}
    kept: list[dict] = []
    for t in tracks:
        artists = t.get("artists") or []
        if not artists:
            continue
        key = artists[0].get("id") or artists[0].get("name", "")
        if counts.get(key, 0) >= max_per_artist:
            continue
        counts[key] = counts.get(key, 0) + 1
        kept.append(t)
    return kept


def search_tracks_paginated(
    query: str,
    seen: set[str],
    *,
    max_pages: int = 5,
    start_offset: int = 0,
) -> list[dict]:
    pool: list[dict] = []
    for page in range(max_pages):
        offset = start_offset + page * SEARCH_LIMIT
        try:
            results = sp.search(q=query, type="track", limit=SEARCH_LIMIT, offset=offset)
        except SpotifyException as e:
            logger.warning(f"Spotify search failed (query='{query}', offset={offset}): {e}")
            break
        items = results["tracks"]["items"]
        if not items:
            break
        pool.extend(_dedupe_tracks(items, seen))
    return pool


def expand_from_albums(tracks: list[dict], seen: set[str], max_albums: int = 3) -> list[dict]:
    extra: list[dict] = []
    for track in tracks[:max_albums]:
        album_id = track.get("album", {}).get("id")
        if not album_id:
            continue
        try:
            album_tracks = sp.album_tracks(album_id, limit=50)["items"]
            extra.extend(_dedupe_tracks(album_tracks, seen))
        except SpotifyException as e:
            logger.warning(f"Failed to expand from album {album_id}: {e}")
            continue
    return extra


def expand_from_artists(tracks: list[dict], seen: set[str], max_artists: int = 3) -> list[dict]:
    """Expand pool with more tracks from the same artists.

    Uses search with artist ID filter instead of /artists/{id}/top-tracks,
    which is deprecated for apps using Client Credentials auth.
    """
    extra: list[dict] = []
    searched: set[str] = set()
    for track in tracks[:max_artists]:
        artists = track.get("artists") or []
        if not artists:
            continue
        artist_name = artists[0].get("name")
        if not artist_name or artist_name in searched:
            continue
        searched.add(artist_name)
        try:
            results = sp.search(
                q=f'artist:"{artist_name}"',
                type="track",
                limit=SEARCH_LIMIT,
            )
            extra.extend(_dedupe_tracks(results["tracks"]["items"], seen))
        except SpotifyException as e:
            logger.warning(f"Failed to expand from artist {artist_name}: {e}")
            continue
    return extra


def build_radio_pool(
    search_queries: list[str],
    mood: str | None = None,
    exclude: set[str] | None = None,
    seed_track_id: str | None = None,
    personal_seeds: list[dict] | None = None,
) -> list[dict]:
    exclude = exclude or set()
    seen = set(exclude)
    pool: list[dict] = []

    # Multi-query fan-out: each query fishes from a different region of
    # Spotify's catalogue. Random page-offset per query escapes the
    # top-of-relevance trap that flattened previous pools.
    exclude_offset = (len(exclude) // SEARCH_LIMIT) * SEARCH_LIMIT
    for query in search_queries:
        jitter = random.randint(0, 3) * SEARCH_LIMIT
        pool.extend(search_tracks_paginated(
            query, seen,
            start_offset=exclude_offset + jitter,
            max_pages=3,
        ))

    seed_tracks: list[dict] = []
    if seed_track_id:
        try:
            seed_tracks = [sp.track(seed_track_id)]
        except SpotifyException as e:
            logger.warning(f"Failed to fetch seed track {seed_track_id}: {e}")

    expand_sources = seed_tracks or pool[:5]

    # If the user is logged in, use their top tracks only as expansion seeds —
    # we pull adjacent music from artists they love, but the pool stays
    # anchored to the mood queries.
    if personal_seeds:
        expand_sources = list(personal_seeds[:3]) + list(expand_sources)

    pool.extend(expand_from_artists(expand_sources, seen))
    pool.extend(expand_from_albums(expand_sources, seen))

    pool = _cap_per_artist(pool, max_per_artist=3)

    if mood:
        pool = filter_pool_by_mood(pool, mood)

    random.shuffle(pool)
    return pool


def resolve_search_queries(mood: str, taste: str | None = None) -> list[str]:
    """Return the list of search queries for this mood, dropping any that
    Spotify can't satisfy. Falls back to the literal mood if all fail."""
    normalized = _normalize_mood(mood)
    candidates = list(interpret_mood_queries(normalized, taste))

    good: list[str] = []
    for q in candidates:
        try:
            probe = sp.search(q=q, type="track", limit=1)
            if probe["tracks"]["items"]:
                good.append(q)
        except SpotifyException as e:
            logger.warning(f"Probe search failed for query '{q}': {e}")

    if not good:
        logger.info("No interpreted queries returned results; falling back to literal mood")
        return [normalized]
    return good


def _join_queries(queries: list[str]) -> str:
    return "|".join(queries)


def _split_queries(value: str) -> list[str]:
    return [q.strip() for q in value.split("|") if q.strip()]


@app.get("/")
def index():
    return FileResponse("index.html")


@app.get("/generate")
def generate(mood: str = Query(..., min_length=1, max_length=200)):
    taste = get_user_taste_summary()
    queries = resolve_search_queries(mood, taste)
    personal_seeds = get_user_seed_tracks() if taste else []
    pool = build_radio_pool(queries, mood=mood, personal_seeds=personal_seeds)

    if not pool:
        raise HTTPException(status_code=404, detail="No tracks found for this mood")

    tracks = [track_payload(t) for t in pool]
    current = tracks[0]

    return {
        "mood": mood,
        "search_query": _join_queries(queries),
        "mode": "radio",
        **current,
        "tracks": tracks,
        "queue": tracks[1:],
    }


@app.get("/radio/next")
def radio_next(
    mood: str = Query(..., min_length=1, max_length=200),
    exclude: str = Query("", max_length=5000),
    search_query: str | None = Query(None, max_length=800),
):
    exclude_list = [tid.strip() for tid in exclude.split(",") if tid.strip()]
    exclude_set = set(exclude_list)

    if search_query:
        queries = _split_queries(search_query)
    else:
        taste = get_user_taste_summary()
        queries = resolve_search_queries(mood, taste)

    seed_track_id = exclude_list[-1] if exclude_list else None
    personal_seeds = get_user_seed_tracks() if get_user_sp() else []

    pool = build_radio_pool(queries, mood=mood, exclude=exclude_set, seed_track_id=seed_track_id, personal_seeds=personal_seeds)

    if not pool:
        raise HTTPException(status_code=404, detail="No more tracks for this mood")

    tracks = [track_payload(t) for t in pool]
    current = tracks[0]

    return {
        "mood": mood,
        "search_query": _join_queries(queries),
        "mode": "radio",
        **current,
        "tracks": tracks,
        "queue": tracks[1:],
    }


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@app.get("/login")
def login():
    return RedirectResponse(_user_oauth().get_authorize_url())


@app.get("/callback")
def callback(code: str = Query(None), error: str = Query(None)):
    if error or not code:
        return RedirectResponse("/?auth_error=1")
    oauth = _user_oauth()
    oauth.get_access_token(code, as_dict=False)  # writes to USER_CACHE automatically
    return RedirectResponse("/?logged_in=1")


@app.get("/me")
def me():
    user_sp = get_user_sp()
    if not user_sp:
        return {"logged_in": False}
    try:
        profile = user_sp.me()
        return {
            "logged_in": True,
            "name": profile.get("display_name") or profile["id"],
        }
    except Exception:
        return {"logged_in": False}


@app.get("/logout")
def logout():
    if os.path.exists(USER_CACHE):
        os.remove(USER_CACHE)
    return RedirectResponse("/")


