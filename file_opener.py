"""
File Opener Module — Emily Voice Assistant
==========================================
Indexes and opens files/folders from user directories via fuzzy search.
Supports: audio, video, image, document files + named folders.

Usage:
    from file_opener import get_file_opener
    opener = get_file_opener()
    await opener.open("my song", file_type="audio")
    await opener.open_folder("music")
"""

import os
import asyncio
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ==============================================================================
# Configuration
# ==============================================================================
SCAN_FOLDERS = [
    r"C:\Users\Lenovo\Documents",
    r"C:\Users\Lenovo\Desktop",
    r"C:\Users\Lenovo\Downloads",
    r"C:\Users\Lenovo\Documents\OneDrive\Videos",
    r"C:\Users\Lenovo\Desktop\Music",
    r"C:\Users\Lenovo\Desktop\Movies",
    r"C:\Users\Lenovo\Documents\OneDrive\Pictures",
]

# Named folder aliases for quick access
FOLDER_ALIASES = {
    # Music folders
    "music": r"C:\Users\Lenovo\Desktop\Music",
    "music folder": r"C:\Users\Lenovo\Desktop\Music",
    "songs": r"C:\Users\Lenovo\Desktop\Music",
    "gaane": r"C:\Users\Lenovo\Desktop\Music",
    
    # Video folders
    "movies": r"C:\Users\Lenovo\Desktop\Movies",
    "movies folder": r"C:\Users\Lenovo\Desktop\Movies",
    "films": r"C:\Users\Lenovo\Desktop\Movies",
    "videos": r"C:\Users\Lenovo\Documents\OneDrive\Videos",
    "videos folder": r"C:\Users\Lenovo\Documents\OneDrive\Videos",
    
    # Pictures
    "pictures": r"C:\Users\Lenovo\Documents\OneDrive\Pictures",
    "pictures folder": r"C:\Users\Lenovo\Documents\OneDrive\Pictures",
    "photos": r"C:\Users\Lenovo\Documents\OneDrive\Pictures",
    "images": r"C:\Users\Lenovo\Documents\OneDrive\Pictures",
    
    # Common folders
    "documents": r"C:\Users\Lenovo\Documents",
    "documents folder": r"C:\Users\Lenovo\Documents",
    "docs": r"C:\Users\Lenovo\Documents",
    "desktop": r"C:\Users\Lenovo\Desktop",
    "downloads": r"C:\Users\Lenovo\Downloads",
    "download": r"C:\Users\Lenovo\Downloads",
    "download folder": r"C:\Users\Lenovo\Downloads",
}

FILE_TYPES = {
    "audio": [".mp3", ".wav", ".flac", ".aac", ".m4a", ".ogg"],
    "video": [".mp4", ".mkv", ".avi", ".mov", ".wmv"],
    "image": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"],
    "document": [".pdf", ".docx", ".txt", ".xlsx", ".pptx"],
}

# Keywords that indicate folder intent (not file)
FOLDER_KEYWORDS = ["folder", "directory", "dir", "location", "path"]

ALL_EXTENSIONS = set()
for exts in FILE_TYPES.values():
    ALL_EXTENSIONS.update(exts)

FUZZY_THRESHOLD = 70

# ==============================================================================
# Folder Browser State (Interactive Selection System)
# ==============================================================================
class FolderBrowserState:
    """
    Maintains state for interactive folder browsing.
    When user opens a folder, contents are listed with numbers.
    User can then say "play 1" or "open 2" to select an item.
    """
    
    def __init__(self):
        self.active = False
        self.current_folder: Optional[str] = None
        self.current_items: list[dict] = []  # [{name, path, is_folder, type, index}]
        self.folder_name: str = ""
    
    def clear(self):
        """Reset browser state."""
        self.active = False
        self.current_folder = None
        self.current_items = []
        self.folder_name = ""
    
    def set_folder(self, folder_path: str, items: list[dict], folder_name: str = ""):
        """Set current folder and its contents."""
        self.active = True
        self.current_folder = folder_path
        self.current_items = items
        self.folder_name = folder_name or os.path.basename(folder_path)
    
    def get_item(self, index: int) -> Optional[dict]:
        """Get item by 1-based index."""
        if 1 <= index <= len(self.current_items):
            return self.current_items[index - 1]
        return None
    
    def is_active(self) -> bool:
        return self.active and len(self.current_items) > 0


# Global browser state (singleton)
_browser_state = FolderBrowserState()

# ==============================================================================
# Optional Dependencies
# ==============================================================================
FUZZYWUZZY_AVAILABLE = False
fuzz = None
fuzz_process = None

try:
    from fuzzywuzzy import fuzz as _fuzz
    from fuzzywuzzy import process as _fuzz_process
    fuzz = _fuzz
    fuzz_process = _fuzz_process
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    logger.warning("fuzzywuzzy not available — install with: pip install fuzzywuzzy python-Levenshtein")

PYGETWINDOW_AVAILABLE = False
gw = None

try:
    import pygetwindow as _gw
    gw = _gw
    PYGETWINDOW_AVAILABLE = True
except ImportError:
    logger.warning("pygetwindow not available — window focus disabled")


# ==============================================================================
# File Entry Data Class
# ==============================================================================
class FileEntry:
    __slots__ = ("name", "path", "extension", "modified", "file_type")

    def __init__(self, name: str, path: str, extension: str, modified: datetime, file_type: str):
        self.name = name
        self.path = path
        self.extension = extension
        self.modified = modified
        self.file_type = file_type

    def __repr__(self):
        return f"FileEntry({self.name}, {self.file_type})"


# ==============================================================================
# FileOpener Class
# ==============================================================================
class FileOpener:
    def __init__(self):
        self._index: list[FileEntry] = []
        self._indexed = False
        self._stats = {"total": 0, "audio": 0, "video": 0, "image": 0, "document": 0}

    def _get_file_type(self, extension: str) -> Optional[str]:
        ext_lower = extension.lower()
        for ftype, extensions in FILE_TYPES.items():
            if ext_lower in extensions:
                return ftype
        return None

    def _scan_folder(self, folder: str) -> int:
        if not os.path.exists(folder):
            logger.warning("Folder does not exist, skipping: %s", folder)
            return 0

        if not os.path.isdir(folder):
            logger.warning("Path is not a directory, skipping: %s", folder)
            return 0

        count = 0
        try:
            for root, _, files in os.walk(folder):
                for filename in files:
                    try:
                        filepath = os.path.join(root, filename)
                        _, ext = os.path.splitext(filename)
                        ext_lower = ext.lower()

                        if ext_lower not in ALL_EXTENSIONS:
                            continue

                        file_type = self._get_file_type(ext_lower)
                        if not file_type:
                            continue

                        try:
                            mtime = os.path.getmtime(filepath)
                            modified = datetime.fromtimestamp(mtime)
                        except (OSError, ValueError):
                            modified = datetime.now()

                        entry = FileEntry(
                            name=filename,
                            path=filepath,
                            extension=ext_lower,
                            modified=modified,
                            file_type=file_type,
                        )
                        self._index.append(entry)
                        self._stats[file_type] += 1
                        count += 1

                    except PermissionError:
                        logger.warning("Permission denied: %s", os.path.join(root, filename))
                    except Exception as e:
                        logger.warning("Error indexing file %s: %s", filename, e)

        except PermissionError:
            logger.warning("Permission denied for folder: %s", folder)
        except Exception as e:
            logger.warning("Error scanning folder %s: %s", folder, e)

        return count

    async def refresh_index(self) -> dict:
        self._index.clear()
        self._stats = {"total": 0, "audio": 0, "video": 0, "image": 0, "document": 0}

        loop = asyncio.get_event_loop()

        for folder in SCAN_FOLDERS:
            count = await loop.run_in_executor(None, self._scan_folder, folder)
            logger.info("Scanned %s: %d files indexed", folder, count)

        self._stats["total"] = len(self._index)
        self._indexed = True

        logger.info(
            "File index complete: %d total files (audio=%d, video=%d, image=%d, document=%d)",
            self._stats["total"],
            self._stats["audio"],
            self._stats["video"],
            self._stats["image"],
            self._stats["document"],
        )

        return self._stats.copy()

    async def _ensure_indexed(self):
        if not self._indexed:
            await self.refresh_index()

    async def search(self, query: str, file_type: Optional[str] = None) -> list[tuple[FileEntry, int]]:
        await self._ensure_indexed()

        if not query or not query.strip():
            return []

        query_lower = query.lower().strip()

        candidates = self._index
        if file_type and file_type in FILE_TYPES:
            candidates = [f for f in self._index if f.file_type == file_type]

        if not candidates:
            logger.info("Search '%s' (type=%s): no candidates", query, file_type)
            return []

        if not FUZZYWUZZY_AVAILABLE:
            results = []
            for entry in candidates:
                name_lower = entry.name.lower()
                name_no_ext = os.path.splitext(name_lower)[0]
                if query_lower in name_lower or query_lower in name_no_ext:
                    results.append((entry, 80))
            results.sort(key=lambda x: x[1], reverse=True)
            logger.info("Search '%s': %d matches (fallback mode)", query, len(results))
            return results[:3]

        choices = [(entry, os.path.splitext(entry.name)[0]) for entry in candidates]
        choice_names = [c[1] for c in choices]

        matches = fuzz_process.extract(query_lower, choice_names, scorer=fuzz.ratio, limit=10)

        results = []
        for match_name, score in matches:
            if score >= FUZZY_THRESHOLD:
                for entry, name in choices:
                    if name == match_name:
                        results.append((entry, score))
                        break

        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:3]

        if top_results:
            logger.info(
                "Search '%s' (type=%s): top match '%s' (score=%d), %d total matches",
                query,
                file_type,
                top_results[0][0].name,
                top_results[0][1],
                len(results),
            )
        else:
            logger.info("Search '%s' (type=%s): no matches above threshold", query, file_type)

        return top_results

    async def open(self, query: str, file_type: Optional[str] = None) -> str:
        results = await self.search(query, file_type)

        if not results:
            msg = f"Sorry boss, couldn't find any file matching '{query}'"
            if file_type:
                msg += f" in {file_type} files"
            logger.info("Open failed: %s", msg)
            return msg

        best_match, score = results[0]
        filepath = best_match.path

        if not os.path.exists(filepath):
            logger.warning("File no longer exists: %s", filepath)
            return f"Sorry boss, the file '{best_match.name}' no longer exists."

        try:
            os.startfile(filepath)
            logger.info("Opened file: %s (score=%d)", filepath, score)

            if PYGETWINDOW_AVAILABLE:
                await asyncio.sleep(1.0)
                try:
                    windows = gw.getWindowsWithTitle(os.path.splitext(best_match.name)[0])
                    if windows:
                        windows[0].activate()
                        logger.info("Focused window for: %s", best_match.name)
                except Exception as e:
                    logger.warning("Could not focus window: %s", e)

            return f"Opening {best_match.name}, boss!"

        except PermissionError:
            logger.warning("Permission denied opening: %s", filepath)
            return f"Sorry boss, I don't have permission to open '{best_match.name}'."
        except Exception as e:
            logger.error("Error opening file %s: %s", filepath, e)
            return f"Sorry boss, couldn't open the file: {e}"

    async def get_index_stats(self) -> dict:
        await self._ensure_indexed()
        return self._stats.copy()

    # ==========================================================================
    # Folder Operations
    # ==========================================================================
    
    def _is_folder_request(self, query: str) -> bool:
        """Check if the query is asking to open a folder (not a file)."""
        query_lower = query.lower()
        # Check for folder keywords
        if any(kw in query_lower for kw in FOLDER_KEYWORDS):
            return True
        # Check if it matches a folder alias directly
        if query_lower in FOLDER_ALIASES:
            return True
        return False

    def _get_folder_path(self, query: str) -> Optional[str]:
        """Get folder path from query using aliases or fuzzy match."""
        query_lower = query.lower().strip()
        
        # Remove common prefixes
        for prefix in ["open ", "open the ", "show ", "go to "]:
            if query_lower.startswith(prefix):
                query_lower = query_lower[len(prefix):]
        
        # Direct alias match
        if query_lower in FOLDER_ALIASES:
            return FOLDER_ALIASES[query_lower]
        
        # Try without "folder" suffix
        query_no_folder = query_lower.replace(" folder", "").replace(" directory", "").strip()
        if query_no_folder in FOLDER_ALIASES:
            return FOLDER_ALIASES[query_no_folder]
        
        # Fuzzy match against aliases
        if FUZZYWUZZY_AVAILABLE:
            alias_names = list(FOLDER_ALIASES.keys())
            matches = fuzz_process.extract(query_lower, alias_names, scorer=fuzz.ratio, limit=1)
            if matches and matches[0][1] >= 70:
                return FOLDER_ALIASES[matches[0][0]]
            
            # Also try without "folder" keyword
            matches = fuzz_process.extract(query_no_folder, alias_names, scorer=fuzz.ratio, limit=1)
            if matches and matches[0][1] >= 70:
                return FOLDER_ALIASES[matches[0][0]]
        
        return None

    async def open_folder(self, query: str) -> str:
        """Open a folder by name/alias (simple open in Explorer)."""
        folder_path = self._get_folder_path(query)
        
        if not folder_path:
            logger.info("Folder not found for query: %s", query)
            return None  # Return None to indicate fallback to other handlers
        
        if not os.path.exists(folder_path):
            logger.warning("Folder does not exist: %s", folder_path)
            return f"Sorry boss, the folder doesn't exist: {folder_path}"
        
        try:
            os.startfile(folder_path)
            folder_name = os.path.basename(folder_path) or folder_path
            logger.info("Opened folder: %s", folder_path)
            return f"Opening {folder_name} folder, boss!"
        except Exception as e:
            logger.error("Error opening folder %s: %s", folder_path, e)
            return f"Sorry boss, couldn't open the folder: {e}"

    async def browse_folder(self, query: str) -> tuple[str, list[dict]]:
        """
        Browse folder contents with numbered list (Interactive Mode).
        Returns: (message_for_tts, list_of_items)
        """
        folder_path = self._get_folder_path(query)
        
        if not folder_path:
            return "Sorry boss, I don't know that folder.", []
        
        if not os.path.exists(folder_path):
            return f"Sorry boss, the folder doesn't exist.", []
        
        folder_name = os.path.basename(folder_path) or "folder"
        
        try:
            items = []
            index = 1
            
            # Get all items in folder (files + subfolders)
            for entry in os.scandir(folder_path):
                try:
                    item = {
                        "index": index,
                        "name": entry.name,
                        "path": entry.path,
                        "is_folder": entry.is_dir(),
                        "type": None,
                        "extension": "",
                    }
                    
                    if entry.is_file():
                        _, ext = os.path.splitext(entry.name)
                        ext_lower = ext.lower()
                        item["extension"] = ext_lower
                        item["type"] = self._get_file_type(ext_lower)
                        
                        # Only include supported file types
                        if item["type"] is None and ext_lower not in ALL_EXTENSIONS:
                            continue
                    
                    items.append(item)
                    index += 1
                    
                except PermissionError:
                    continue
            
            if not items:
                _browser_state.clear()
                return f"The {folder_name} folder is empty, boss.", []
            
            # Sort: folders first, then files alphabetically
            items.sort(key=lambda x: (not x["is_folder"], x["name"].lower()))
            
            # Re-index after sorting
            for i, item in enumerate(items):
                item["index"] = i + 1
            
            # Update browser state
            _browser_state.set_folder(folder_path, items, folder_name)
            
            # Build display message
            print(f"\n📂 {folder_name.upper()} FOLDER CONTENTS:")
            print("─" * 50)
            
            for item in items[:20]:  # Show max 20 items
                icon = "📁" if item["is_folder"] else self._get_file_icon(item["type"])
                name_display = item["name"]
                if len(name_display) > 40:
                    name_display = name_display[:37] + "..."
                print(f"  {item['index']:2}. {icon} {name_display}")
            
            if len(items) > 20:
                print(f"  ... and {len(items) - 20} more items")
            
            print("─" * 50)
            print("🎤 Say 'play 1' or 'open 2' to select an item")
            print("─" * 50 + "\n")
            
            # TTS response
            file_count = sum(1 for i in items if not i["is_folder"])
            folder_count = sum(1 for i in items if i["is_folder"])
            
            tts_msg = f"Found {len(items)} items in {folder_name}. "
            if folder_count > 0:
                tts_msg += f"{folder_count} folders and "
            tts_msg += f"{file_count} files. "
            tts_msg += "Say the number to open or play. For example, play 1 or open 2."
            
            logger.info("Browsing folder: %s (%d items)", folder_path, len(items))
            return tts_msg, items
            
        except PermissionError:
            return f"Sorry boss, I don't have permission to access {folder_name}.", []
        except Exception as e:
            logger.error("Error browsing folder %s: %s", folder_path, e)
            return f"Sorry boss, couldn't browse the folder: {e}", []

    def _get_file_icon(self, file_type: Optional[str]) -> str:
        """Get emoji icon for file type."""
        icons = {
            "audio": "🎵",
            "video": "🎬",
            "image": "🖼️",
            "document": "📄",
        }
        return icons.get(file_type, "📄")

    async def select_item(self, number: int) -> str:
        """
        Select an item by number from the current browser state.
        Opens files or navigates into folders.
        """
        if not _browser_state.is_active():
            return "No folder is currently open, boss. Say 'open music folder' first."
        
        item = _browser_state.get_item(number)
        
        if not item:
            return f"Invalid number boss. Choose between 1 and {len(_browser_state.current_items)}."
        
        item_path = item["path"]
        item_name = item["name"]
        
        if not os.path.exists(item_path):
            return f"Sorry boss, {item_name} no longer exists."
        
        try:
            if item["is_folder"]:
                # Navigate into subfolder — browse it
                result, _ = await self.browse_folder(item_path)
                return result
            else:
                # Open the file
                os.startfile(item_path)
                logger.info("Opened from browser: %s", item_path)
                
                # Determine action word based on type
                action = "Playing" if item["type"] in ("audio", "video") else "Opening"
                
                # Clear browser state after opening a file
                _browser_state.clear()
                
                return f"{action} {item_name}, boss!"
                
        except Exception as e:
            logger.error("Error opening item %s: %s", item_path, e)
            return f"Sorry boss, couldn't open {item_name}: {e}"

    async def go_back(self) -> str:
        """Go back to parent folder in browser."""
        if not _browser_state.is_active():
            return "No folder is currently open, boss."
        
        current = _browser_state.current_folder
        parent = os.path.dirname(current)
        
        # Check if parent is in allowed folders
        allowed = False
        for scan_folder in SCAN_FOLDERS:
            if parent.startswith(scan_folder) or scan_folder.startswith(parent):
                allowed = True
                break
        
        if not allowed or parent == current:
            _browser_state.clear()
            return "Already at the top level, boss. Closing browser."
        
        # Browse parent folder
        _browser_state.current_folder = parent
        result, _ = await self.browse_folder(parent)
        return result

    def close_browser(self) -> str:
        """Close the folder browser."""
        if _browser_state.is_active():
            folder_name = _browser_state.folder_name
            _browser_state.clear()
            return f"Closed {folder_name} browser, boss."
        return "No folder browser is open, boss."

    def is_browser_active(self) -> bool:
        """Check if folder browser is active."""
        return _browser_state.is_active()

    def get_browser_state(self) -> dict:
        """Get current browser state info."""
        return {
            "active": _browser_state.active,
            "folder": _browser_state.folder_name,
            "item_count": len(_browser_state.current_items),
        }

    async def smart_open(self, query: str, file_type: Optional[str] = None) -> Optional[str]:
        """
        Smart open: detects if user wants folder or file.
        Returns None if no match found (so caller can fallback to other handlers).
        """
        # First check if it's a folder request
        if self._is_folder_request(query):
            result = await self.open_folder(query)
            if result:
                return result
        
        # Try to find a file
        results = await self.search(query, file_type)
        if results:
            return await self.open(query, file_type)
        
        # No match — return None to allow fallback
        return None


# ==============================================================================
# Singleton
# ==============================================================================
_file_opener_instance: Optional[FileOpener] = None


def get_file_opener() -> FileOpener:
    global _file_opener_instance
    if _file_opener_instance is None:
        _file_opener_instance = FileOpener()
        logger.info("FileOpener instance created")
    return _file_opener_instance


# ==============================================================================
# Helper function for quick folder check (sync, for use in processors)
# ==============================================================================
def is_file_or_folder_request(query: str) -> bool:
    """
    Quick check if a query might be for file_opener (not an app).
    Used by AppRequestProcessor to route correctly.
    """
    query_lower = query.lower()
    
    # Check folder aliases
    for alias in FOLDER_ALIASES:
        if alias in query_lower:
            return True
    
    # Check for file type keywords
    file_keywords = [
        "song", "music", "video", "movie", "photo", "picture", "image",
        "document", "pdf", "file", "folder", "gaana", "gana",
        ".mp3", ".mp4", ".pdf", ".jpg", ".png", ".mkv", ".avi",
    ]
    if any(kw in query_lower for kw in file_keywords):
        return True
    
    return False


def is_browser_selection_command(query: str) -> tuple[bool, int]:
    """
    Check if query is a browser selection command like 'play 1', 'open 2', 'open one', etc.
    Returns: (is_selection_command, number)
    """
    import re
    
    query_lower = query.lower().strip()
    
    # Remove punctuation for matching
    query_clean = re.sub(r'[.,!?]', '', query_lower)
    
    # Patterns for selection commands with numbers
    patterns = [
        r"^(?:play|open|select|chala|khol|bajao)\s*(?:number\s*)?(\d+)$",
        r"^(?:number\s*)?(\d+)\s*(?:play|open|chala|khol|bajao)?$",
        r"^(\d+)(?:st|nd|rd|th)?\s*(?:one|wala|wali)?$",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query_clean)
        if match:
            try:
                num = int(match.group(1))
                if 1 <= num <= 100:  # Reasonable range
                    return True, num
            except (ValueError, IndexError):
                pass
    
    # Word numbers mapping
    word_numbers = {
        "one": 1, "first": 1, "pehla": 1, "ek": 1,
        "two": 2, "second": 2, "doosra": 2, "do": 2,
        "three": 3, "third": 3, "teesra": 3, "teen": 3,
        "four": 4, "fourth": 4, "chautha": 4, "char": 4,
        "five": 5, "fifth": 5, "panchwa": 5, "paanch": 5,
        "six": 6, "sixth": 6, "chhata": 6, "chhe": 6,
        "seven": 7, "seventh": 7, "saatwa": 7, "saat": 7,
        "eight": 8, "eighth": 8, "aathwa": 8, "aath": 8,
        "nine": 9, "ninth": 9, "nauwa": 9, "nau": 9,
        "ten": 10, "tenth": 10, "daswa": 10, "das": 10,
    }
    
    # Selection action keywords
    selection_keywords = ["play", "open", "select", "chala", "khol", "bajao", "number"]
    
    # Check if any selection keyword is present
    has_selection_keyword = any(kw in query_clean for kw in selection_keywords)
    
    # Check for word numbers
    for word, num in word_numbers.items():
        # Match whole word
        if re.search(rf'\b{word}\b', query_clean):
            # If selection keyword present, or phrase is very short (like "open one")
            if has_selection_keyword or len(query_clean.split()) <= 3:
                return True, num
    
    return False, 0


def is_browser_navigation_command(query: str) -> Optional[str]:
    """
    Check if query is a browser navigation command.
    Returns: 'back', 'close', or None
    """
    query_lower = query.lower().strip()
    
    back_keywords = ["go back", "back", "peeche", "previous folder", "parent folder", "wapas"]
    close_keywords = ["close browser", "close folder", "exit browser", "band karo browser", "close list"]
    
    if any(kw in query_lower for kw in close_keywords):
        return "close"
    if any(kw in query_lower for kw in back_keywords):
        return "back"
    
    return None


def get_browser_state() -> dict:
    """Get current browser state (for external access)."""
    return {
        "active": _browser_state.active,
        "folder": _browser_state.folder_name,
        "item_count": len(_browser_state.current_items),
        "current_folder": _browser_state.current_folder,
    }


# ==============================================================================
# Quick Test
# ==============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-20s %(levelname)-8s %(message)s",
    )

    async def test():
        opener = get_file_opener()

        print("=" * 50)
        print("Testing File Opener Module")
        print("=" * 50)
        
        print("\n1. Indexing files...")
        stats = await opener.refresh_index()
        print(f"   Index stats: {stats}")
        
        print("\n2. Testing folder detection...")
        test_queries = [
            "music folder",
            "open music",
            "documents",
            "downloads folder",
            "videos",
        ]
        for q in test_queries:
            is_folder = opener._is_folder_request(q)
            path = opener._get_folder_path(q)
            print(f"   '{q}' → folder={is_folder}, path={path}")
        
        print("\n3. Testing smart_open for 'music folder'...")
        result = await opener.smart_open("music folder")
        print(f"   Result: {result}")
        
        print("\n4. Testing search for documents...")
        results = await opener.search("readme", file_type="document")
        if results:
            for entry, score in results[:3]:
                print(f"   [{score}] {entry.name}")
        else:
            print("   No documents found matching 'readme'")
        
        print("\n5. Testing is_file_or_folder_request helper...")
        test_targets = ["notepad", "music folder", "my song.mp3", "chrome", "photos"]
        for t in test_targets:
            result = is_file_or_folder_request(t)
            print(f"   '{t}' → {result}")
        
        print("\n6. Testing browse_folder for 'music'...")
        msg, items = await opener.browse_folder("music")
        print(f"   TTS: {msg}")
        print(f"   Items found: {len(items)}")
        
        print("\n7. Testing selection command detection...")
        test_commands = ["play 1", "open 2", "3", "number 5", "pehla chala", "open notepad"]
        for cmd in test_commands:
            is_sel, num = is_browser_selection_command(cmd)
            print(f"   '{cmd}' → selection={is_sel}, number={num}")
        
        print("\n8. Testing navigation command detection...")
        test_nav = ["go back", "close browser", "peeche", "open chrome"]
        for cmd in test_nav:
            result = is_browser_navigation_command(cmd)
            print(f"   '{cmd}' → {result}")
        
        print("\n" + "=" * 50)
        print("Done!")

    asyncio.run(test())
