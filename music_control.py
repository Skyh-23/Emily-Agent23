"""
Music Control Module — Emily Voice Assistant
=============================================
Controls music playback via:
  1) System Media Keys (works with any player — Spotify, YouTube, VLC, etc.)
  2) Spotify API (optional, for search/playlists — requires credentials)

Usage:
    from music_control import music_controller
    music_controller.play()
    music_controller.pause()
    music_controller.next_track()
    music_controller.get_status()
"""

import logging
import time
import subprocess
import ctypes
from ctypes import wintypes

logger = logging.getLogger(__name__)


# ==============================================================================
# Windows Virtual Key Codes for Media Keys
# ==============================================================================
VK_MEDIA_PLAY_PAUSE = 0xB3
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_STOP = 0xB2
VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF

# Key event flags
KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002


def _press_media_key(vk_code: int):
    """Press a media key using Windows API."""
    try:
        # keybd_event is simpler and works for media keys
        ctypes.windll.user32.keybd_event(vk_code, 0, KEYEVENTF_EXTENDEDKEY, 0)
        time.sleep(0.05)
        ctypes.windll.user32.keybd_event(vk_code, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)
        return True
    except Exception as e:
        logger.error("Failed to press media key 0x%X: %s", vk_code, e)
        return False


# ==============================================================================
# System Media Controller — Works with ANY media player
# ==============================================================================
class SystemMediaController:
    """
    Controls media playback using Windows media keys.
    Works with: Spotify, YouTube (browser), VLC, Windows Media Player, etc.
    """

    def __init__(self):
        self._is_playing = False  # Track state (best effort)
        logger.info("✅ SystemMediaController initialized (Windows media keys)")

    def play_pause(self) -> str:
        """Toggle play/pause."""
        if _press_media_key(VK_MEDIA_PLAY_PAUSE):
            self._is_playing = not self._is_playing
            action = "Playing" if self._is_playing else "Paused"
            logger.info("Media: %s", action)
            return f"{action} music, boss!"
        return "Sorry boss, couldn't control media playback."

    def play(self) -> str:
        """Start/resume playback."""
        if not self._is_playing:
            return self.play_pause()
        return "Music is already playing, boss!"

    def pause(self) -> str:
        """Pause playback."""
        if self._is_playing:
            return self.play_pause()
        return "Music is already paused, boss!"

    def stop(self) -> str:
        """Stop playback."""
        if _press_media_key(VK_MEDIA_STOP):
            self._is_playing = False
            logger.info("Media: Stopped")
            return "Stopped the music, boss!"
        return "Sorry boss, couldn't stop the music."

    def next_track(self) -> str:
        """Skip to next track."""
        if _press_media_key(VK_MEDIA_NEXT_TRACK):
            logger.info("Media: Next track")
            return "Skipping to next track, boss!"
        return "Sorry boss, couldn't skip track."

    def previous_track(self) -> str:
        """Go to previous track."""
        if _press_media_key(VK_MEDIA_PREV_TRACK):
            logger.info("Media: Previous track")
            return "Going back to previous track, boss!"
        return "Sorry boss, couldn't go back."

    def volume_up(self) -> str:
        """Increase volume."""
        for _ in range(5):  # Press 5 times for noticeable change
            _press_media_key(VK_VOLUME_UP)
            time.sleep(0.05)
        logger.info("Media: Volume up")
        return "Volume up, boss!"

    def volume_down(self) -> str:
        """Decrease volume."""
        for _ in range(5):
            _press_media_key(VK_VOLUME_DOWN)
            time.sleep(0.05)
        logger.info("Media: Volume down")
        return "Volume down, boss!"

    def mute(self) -> str:
        """Toggle mute."""
        if _press_media_key(VK_VOLUME_MUTE):
            logger.info("Media: Mute toggled")
            return "Muted, boss!"
        return "Sorry boss, couldn't mute."

    def get_status(self) -> str:
        """Get current playback status (best effort)."""
        status = "playing" if self._is_playing else "paused"
        return f"Music is currently {status}, boss. I'm using system media keys, so I can't see the track name."


# ==============================================================================
# Spotify Controller — For search/playlists (optional, requires API setup)
# ==============================================================================
class SpotifyController:
    """
    Controls Spotify via Web API using spotipy.
    Requires: SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI in config.
    """

    def __init__(self):
        self.sp = None
        self.available = False
        self._init_spotify()

    def _init_spotify(self):
        """Initialize Spotify client if credentials are available."""
        try:
            from config import (
                SPOTIFY_CLIENT_ID,
                SPOTIFY_CLIENT_SECRET,
                SPOTIFY_REDIRECT_URI,
            )

            if not SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_ID == "your_client_id_here":
                logger.info("Spotify credentials not configured — Spotify features disabled")
                return

            import spotipy
            from spotipy.oauth2 import SpotifyOAuth

            scope = "user-read-playback-state user-modify-playback-state user-read-currently-playing"

            self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET,
                redirect_uri=SPOTIFY_REDIRECT_URI,
                scope=scope,
                cache_path=".spotify_cache",
            ))

            # Test connection
            self.sp.current_user()
            self.available = True
            logger.info("✅ SpotifyController initialized")
            print("✅  Spotify API connected!")

        except ImportError:
            logger.info("spotipy not installed — run: pip install spotipy")
        except Exception as e:
            logger.warning("Spotify init failed: %s", e)

    def play(self, query: str = None) -> str:
        """Play music. If query provided, search and play."""
        if not self.available:
            return None  # Fall back to system media

        try:
            if query:
                # Search for track
                results = self.sp.search(q=query, type="track", limit=1)
                if results["tracks"]["items"]:
                    track = results["tracks"]["items"][0]
                    track_uri = track["uri"]
                    track_name = track["name"]
                    artist = track["artists"][0]["name"]

                    self.sp.start_playback(uris=[track_uri])
                    logger.info("Spotify: Playing %s by %s", track_name, artist)
                    return f"Playing {track_name} by {artist}, boss!"
                else:
                    return f"Sorry boss, couldn't find '{query}' on Spotify."
            else:
                self.sp.start_playback()
                return "Resuming Spotify playback, boss!"
        except Exception as e:
            logger.error("Spotify play error: %s", e)
            return None

    def pause(self) -> str:
        """Pause Spotify playback."""
        if not self.available:
            return None
        try:
            self.sp.pause_playback()
            return "Paused Spotify, boss!"
        except Exception as e:
            logger.error("Spotify pause error: %s", e)
            return None

    def next_track(self) -> str:
        """Skip to next track on Spotify."""
        if not self.available:
            return None
        try:
            self.sp.next_track()
            time.sleep(0.5)
            return self.get_current_track()
        except Exception as e:
            logger.error("Spotify next error: %s", e)
            return None

    def previous_track(self) -> str:
        """Go to previous track on Spotify."""
        if not self.available:
            return None
        try:
            self.sp.previous_track()
            time.sleep(0.5)
            return self.get_current_track()
        except Exception as e:
            logger.error("Spotify previous error: %s", e)
            return None

    def get_current_track(self) -> str:
        """Get currently playing track info."""
        if not self.available:
            return None
        try:
            current = self.sp.current_playback()
            if current and current.get("item"):
                track = current["item"]
                name = track["name"]
                artist = track["artists"][0]["name"]
                is_playing = current["is_playing"]
                status = "playing" if is_playing else "paused"
                return f"Currently {status}: {name} by {artist}"
            return "Nothing is playing on Spotify right now, boss."
        except Exception as e:
            logger.error("Spotify current track error: %s", e)
            return None

    def set_volume(self, percent: int) -> str:
        """Set Spotify volume (0-100)."""
        if not self.available:
            return None
        try:
            self.sp.volume(percent)
            return f"Spotify volume set to {percent}%, boss!"
        except Exception as e:
            logger.error("Spotify volume error: %s", e)
            return None


# ==============================================================================
# Music Controller Facade — Unified interface with fallbacks
# ==============================================================================
class MusicController:
    """
    Unified music controller that tries Spotify first, then falls back to system media keys.
    """

    def __init__(self):
        self.system = SystemMediaController()
        self.spotify = SpotifyController()
        logger.info("MusicController initialized")

    def play(self, query: str = None) -> str:
        """Play music. If query provided, try Spotify search first."""
        if query and self.spotify.available:
            result = self.spotify.play(query)
            if result:
                return result

        # Fall back to system media
        return self.system.play()

    def pause(self) -> str:
        """Pause music."""
        if self.spotify.available:
            result = self.spotify.pause()
            if result:
                return result
        return self.system.pause()

    def play_pause(self) -> str:
        """Toggle play/pause."""
        return self.system.play_pause()

    def stop(self) -> str:
        """Stop music."""
        return self.system.stop()

    def next_track(self) -> str:
        """Skip to next track."""
        if self.spotify.available:
            result = self.spotify.next_track()
            if result:
                return result
        return self.system.next_track()

    def previous_track(self) -> str:
        """Go to previous track."""
        if self.spotify.available:
            result = self.spotify.previous_track()
            if result:
                return result
        return self.system.previous_track()

    def volume_up(self) -> str:
        """Increase volume."""
        return self.system.volume_up()

    def volume_down(self) -> str:
        """Decrease volume."""
        return self.system.volume_down()

    def mute(self) -> str:
        """Toggle mute."""
        return self.system.mute()

    def get_status(self) -> str:
        """Get current playback status."""
        if self.spotify.available:
            result = self.spotify.get_current_track()
            if result:
                return result
        return self.system.get_status()

    def is_spotify_available(self) -> bool:
        """Check if Spotify is configured and connected."""
        return self.spotify.available


# ==============================================================================
# Singleton instance
# ==============================================================================
_music_controller = None


def get_music_controller() -> MusicController:
    """Get or create the shared MusicController instance."""
    global _music_controller
    if _music_controller is None:
        _music_controller = MusicController()
    return _music_controller


# Convenience alias
music_controller = None  # Will be initialized on first import


def _init_on_import():
    global music_controller
    music_controller = get_music_controller()


# Don't auto-init on import — lazy load to avoid slowing startup
# Call get_music_controller() when needed


# ==============================================================================
# Quick test
# ==============================================================================
if __name__ == "__main__":
    print("Testing Music Control Module...")
    print("-" * 40)

    ctrl = get_music_controller()

    print(f"Spotify available: {ctrl.is_spotify_available()}")
    print()

    # Test commands
    commands = [
        ("Play/Pause", ctrl.play_pause),
        ("Status", ctrl.get_status),
    ]

    for name, func in commands:
        print(f"Testing: {name}")
        result = func()
        print(f"  Result: {result}")
        print()
        time.sleep(1)

    print("Done! Try running with music playing to test next/previous.")
