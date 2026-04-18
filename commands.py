"""
Command System — Tool Capabilities for the Voice Assistant
==========================================================
Provides tools that the assistant can execute:
  • open_application(name)      — Launch ANY app via Windows Search
  • close_application(name)     — Close running applications by window title
  • open_website(url)           — Open websites in browser
  • type_text(text)             — Type text in active window
  • run_terminal_command(cmd)   — Execute shell commands
  • create_file(path, content)  — Create new files
  • modify_file(path, content)  — Overwrite existing files
  • read_file(path)             — Read file contents
  • search_files(directory, pattern) — Search for files
  • list_running_apps()         — List all visible windows

All commands are sandboxed with basic safety checks.
"""

import os
import subprocess
import logging
import shutil
import time
import webbrowser

logger = logging.getLogger(__name__)

# ==============================================================================
# Optional Dependencies
# ==============================================================================

# pyautogui for Windows Search method and typing
try:
    import pyautogui
    PYAUTOGUI_AVAILABLE = True
    pyautogui.FAILSAFE = True  # Move mouse to corner to abort
except ImportError:
    PYAUTOGUI_AVAILABLE = False
    logger.warning("pyautogui not available - install with: pip install pyautogui")

# win32gui for window control (close by title)
try:
    import win32gui
    import win32con
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False
    logger.warning("win32gui not available - install with: pip install pywin32")

# psutil for process listing
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - install with: pip install psutil")

# fuzzywuzzy for fuzzy matching
try:
    from fuzzywuzzy import process as fuzz_process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    logger.warning("fuzzywuzzy not available - install with: pip install fuzzywuzzy python-Levenshtein")


# ==============================================================================
# Website Mappings — Common sites with their URLs
# ==============================================================================
WEBSITE_MAPPINGS = {
    # Shopping
    "amazon": "https://www.amazon.in",
    "flipkart": "https://www.flipkart.com",
    "myntra": "https://www.myntra.com",
    "meesho": "https://www.meesho.com",
    
    # Social Media
    "youtube": "https://www.youtube.com",
    "facebook": "https://www.facebook.com",
    "instagram": "https://www.instagram.com",
    "twitter": "https://www.twitter.com",
    "x": "https://www.x.com",
    "linkedin": "https://www.linkedin.com",
    "reddit": "https://www.reddit.com",
    "pinterest": "https://www.pinterest.com",
    
    # Entertainment
    "netflix": "https://www.netflix.com",
    "hotstar": "https://www.hotstar.com",
    "prime video": "https://www.primevideo.com",
    "spotify": "https://open.spotify.com",
    "twitch": "https://www.twitch.tv",
    
    # Productivity
    "google": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "google drive": "https://drive.google.com",
    "google docs": "https://docs.google.com",
    "google sheets": "https://sheets.google.com",
    "notion": "https://www.notion.so",
    "trello": "https://trello.com",
    
    # Development
    "github": "https://github.com",
    "stackoverflow": "https://stackoverflow.com",
    "stack overflow": "https://stackoverflow.com",
    "chatgpt": "https://chat.openai.com",
    "chat gpt": "https://chat.openai.com",
    "claude": "https://claude.ai",
    "huggingface": "https://huggingface.co",
    "kaggle": "https://www.kaggle.com",
    
    # News
    "news": "https://news.google.com",
    "bbc": "https://www.bbc.com",
    
    # Others
    "whatsapp": "https://web.whatsapp.com",
    "telegram": "https://web.telegram.org",
    "wikipedia": "https://www.wikipedia.org",
    "maps": "https://maps.google.com",
    "google maps": "https://maps.google.com",
}

# ==============================================================================
# Fallback App Aliases (used when pyautogui not available)
# ==============================================================================
APP_ALIASES = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "paint": "mspaint.exe",
    "explorer": "explorer.exe",
    "file explorer": "explorer.exe",
    "browser": "start msedge",
    "edge": "start msedge",
    "chrome": "start chrome",
    "firefox": "start firefox",
    "terminal": "start cmd",
    "cmd": "start cmd",
    "powershell": "start powershell",
    "task manager": "taskmgr.exe",
    "settings": "start ms-settings:",
    "vscode": "code",
    "code": "code",
    "vs code": "code",
}


class CommandExecutor:
    """Executes system commands and file operations for the voice assistant."""

    def __init__(self):
        logger.info("Command executor initialized")
        # Check available features
        self.can_open_any_app = PYAUTOGUI_AVAILABLE
        self.can_close_by_title = WIN32_AVAILABLE
        self.can_type = PYAUTOGUI_AVAILABLE
        if self.can_open_any_app:
            print("✅  App control ready (Windows Search method)")
        if self.can_close_by_title:
            print("✅  Window close ready (win32gui)")
        if self.can_type:
            print("✅  Typing control ready (pyautogui)")

    # ------------------------------------------------------------------
    # Application launcher — Windows Search Method (works for ANY app)
    # ------------------------------------------------------------------
    def open_application(self, name: str) -> str:
        """
        Open ANY application using Windows Search emulation.
        Works for all apps, games, and software — not just predefined ones.
        
        Method: Press Win key → Type app name → Press Enter
        """
        name = name.strip()
        if not name:
            return "Boss, please tell me which app to open."

        # Try Windows Search method first (works for ANY app)
        if PYAUTOGUI_AVAILABLE:
            try:
                logger.info("Opening app via Windows Search: %s", name)
                
                # Press Windows key to open Start menu / Search
                pyautogui.press('win')
                time.sleep(1.5)  # Wait for search to open
                
                # Type the app name
                pyautogui.write(name, interval=0.05)
                time.sleep(0.8)  # Wait for search results
                
                # Press Enter to launch
                pyautogui.press('enter')
                
                msg = f"Opening {name} for you, boss."
                logger.info("Opened via Windows Search: %s", name)
                return msg
                
            except Exception as e:
                logger.warning("Windows Search failed: %s, trying fallback", e)
                # Fall through to fallback method
        
        # Fallback: Use predefined aliases
        return self._open_via_alias(name)

    def _open_via_alias(self, name: str) -> str:
        """Fallback method using predefined app aliases."""
        name_lower = name.lower().strip()
        
        # Try exact match first
        cmd = APP_ALIASES.get(name_lower)
        
        # Try fuzzy match if exact not found
        if not cmd and FUZZY_AVAILABLE:
            matches = fuzz_process.extractBests(name_lower, APP_ALIASES.keys(), score_cutoff=70)
            if matches:
                best_match = matches[0][0]
                cmd = APP_ALIASES.get(best_match)
                logger.info("Fuzzy matched '%s' to '%s'", name, best_match)
        
        if cmd:
            try:
                subprocess.Popen(cmd, shell=True)
                msg = f"Opening {name} for you, boss."
                logger.info("Opened application: %s -> %s", name, cmd)
                return msg
            except Exception as e:
                msg = f"Couldn't open {name}: {e}"
                logger.error("Failed to open %s: %s", name, e)
                return msg
        else:
            # Last resort: try direct launch
            try:
                subprocess.Popen(f"start {name_lower}", shell=True)
                msg = f"Launching {name}..."
                logger.info("Launched application: %s", name)
                return msg
            except Exception as e:
                msg = f"Sorry boss, I couldn't find '{name}'. Make sure it's installed."
                logger.error("Unknown application: %s", name)
                return msg

    # ------------------------------------------------------------------
    # Application closer — Close by window title (fuzzy matching)
    # ------------------------------------------------------------------
    def close_application(self, name: str) -> str:
        """
        Close an application by matching its window title.
        Uses fuzzy matching to find the right window.
        
        Args:
            name: App name or window title keyword (e.g., "chrome", "notepad")
        
        Returns:
            Status message
        """
        name = name.strip()
        if not name:
            return "Boss, please tell me which app to close."
        
        target = name.lower()
        closed_count = 0
        
        if WIN32_AVAILABLE:
            try:
                def enum_handler(hwnd, results):
                    """Callback to enumerate all windows."""
                    if win32gui.IsWindowVisible(hwnd):
                        title = win32gui.GetWindowText(hwnd)
                        if title and target in title.lower():
                            results.append((hwnd, title))
                
                windows_to_close = []
                win32gui.EnumWindows(enum_handler, windows_to_close)
                
                for hwnd, title in windows_to_close:
                    try:
                        win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
                        closed_count += 1
                        logger.info("Closed window: %s", title)
                    except Exception as e:
                        logger.warning("Failed to close window '%s': %s", title, e)
                
                if closed_count > 0:
                    return f"Closed {closed_count} window(s) matching '{name}', boss."
                else:
                    return f"No open window found matching '{name}', boss."
                    
            except Exception as e:
                logger.error("Window close error: %s", e)
                # Fall through to taskkill
        
        # Fallback: Use taskkill (process name based)
        return self._close_via_taskkill(name)

    def _close_via_taskkill(self, name: str) -> str:
        """Fallback: close app using taskkill command."""
        # Common process name mappings
        process_names = {
            "chrome": "chrome.exe",
            "firefox": "firefox.exe",
            "edge": "msedge.exe",
            "notepad": "notepad.exe",
            "calculator": "calculator.exe",
            "calc": "calculator.exe",
            "paint": "mspaint.exe",
            "vscode": "code.exe",
            "vs code": "code.exe",
            "code": "code.exe",
            "word": "winword.exe",
            "excel": "excel.exe",
            "spotify": "spotify.exe",
            "discord": "discord.exe",
            "steam": "steam.exe",
            "vlc": "vlc.exe",
        }
        
        name_lower = name.lower().strip()
        process = process_names.get(name_lower, f"{name_lower}.exe")
        
        try:
            result = subprocess.run(
                f"taskkill /IM {process} /F",
                shell=True,
                capture_output=True,
                text=True,
            )
            
            if result.returncode == 0:
                return f"Closed {name} for you, boss."
            else:
                return f"Couldn't find {name} running, boss."
                
        except Exception as e:
            logger.error("Taskkill failed: %s", e)
            return f"Failed to close {name}: {e}"

    # ------------------------------------------------------------------
    # List running applications
    # ------------------------------------------------------------------
    def list_running_apps(self) -> str:
        """List all visible windows / running applications."""
        apps = []
        
        if WIN32_AVAILABLE:
            def enum_handler(hwnd, results):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if title and len(title) > 1:
                        results.append(title)
            
            win32gui.EnumWindows(enum_handler, apps)
            
            if apps:
                # Remove duplicates and limit
                apps = list(dict.fromkeys(apps))[:15]
                result = "Currently open windows:\n"
                result += "\n".join(f"  • {app}" for app in apps)
                return result
            else:
                return "No visible windows found, boss."
        
        elif PSUTIL_AVAILABLE:
            # Fallback: use psutil to list processes
            seen = set()
            for proc in psutil.process_iter(['name']):
                try:
                    name = proc.info['name']
                    if name and name not in seen:
                        seen.add(name)
                        apps.append(name)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            apps = apps[:20]
            result = "Running processes:\n"
            result += "\n".join(f"  • {app}" for app in apps)
            return result
        
        else:
            return "Cannot list apps — win32gui and psutil not installed."

    # ------------------------------------------------------------------
    # Type text in active window
    # ------------------------------------------------------------------
    def type_text(self, text: str) -> str:
        """
        Type text in the currently active window.
        Uses pyautogui to simulate keyboard input.
        
        Args:
            text: The text to type
        
        Returns:
            Status message
        """
        if not text:
            return "Boss, please tell me what to type."
        
        if not PYAUTOGUI_AVAILABLE:
            return "Sorry boss, typing feature requires pyautogui. Install with: pip install pyautogui"
        
        try:
            # Small delay to ensure focus is on target window
            time.sleep(0.3)
            
            # Type the text with realistic speed
            pyautogui.write(text, interval=0.02)
            
            logger.info("Typed text: %s", text[:50] + "..." if len(text) > 50 else text)
            return f"Done boss, typed: {text[:50]}{'...' if len(text) > 50 else ''}"
            
        except Exception as e:
            logger.error("Failed to type text: %s", e)
            return f"Sorry boss, couldn't type that: {e}"
    
    def press_key(self, key: str) -> str:
        """
        Press a special key like Enter, Tab, Escape, etc.
        
        Args:
            key: Key name (enter, tab, escape, backspace, etc.)
        
        Returns:
            Status message
        """
        if not PYAUTOGUI_AVAILABLE:
            return "Sorry boss, key press requires pyautogui."
        
        key = key.lower().strip()
        
        # Map common key names
        key_map = {
            "enter": "enter",
            "return": "enter",
            "tab": "tab",
            "escape": "escape",
            "esc": "escape",
            "backspace": "backspace",
            "delete": "delete",
            "space": "space",
            "up": "up",
            "down": "down",
            "left": "left",
            "right": "right",
            "home": "home",
            "end": "end",
            "pageup": "pageup",
            "pagedown": "pagedown",
            "ctrl+a": ["ctrl", "a"],
            "ctrl+c": ["ctrl", "c"],
            "ctrl+v": ["ctrl", "v"],
            "ctrl+s": ["ctrl", "s"],
            "ctrl+z": ["ctrl", "z"],
            "alt+tab": ["alt", "tab"],
            "alt+f4": ["alt", "f4"],
        }
        
        try:
            mapped = key_map.get(key, key)
            
            if isinstance(mapped, list):
                pyautogui.hotkey(*mapped)
            else:
                pyautogui.press(mapped)
            
            logger.info("Pressed key: %s", key)
            return f"Pressed {key}, boss."
            
        except Exception as e:
            logger.error("Failed to press key: %s", e)
            return f"Couldn't press {key}: {e}"

    # ------------------------------------------------------------------
    # Website opener — Open URLs in default browser
    # ------------------------------------------------------------------
    def open_website(self, site: str) -> str:
        """
        Open a website in the default browser.
        Supports common site names (amazon, youtube, etc.) or full URLs.
        
        Args:
            site: Website name or URL
        
        Returns:
            Status message
        """
        site = site.strip().lower()
        
        if not site:
            return "Boss, please tell me which website to open."
        
        # Check if it's a known site
        url = WEBSITE_MAPPINGS.get(site)
        
        if not url:
            # Maybe it's already a URL
            if site.startswith(("http://", "https://", "www.")):
                url = site if site.startswith("http") else f"https://{site}"
            else:
                # Check for partial matches
                for key in WEBSITE_MAPPINGS:
                    if key in site or site in key:
                        url = WEBSITE_MAPPINGS[key]
                        break
                
                # If still no match, construct URL
                if not url:
                    # Remove spaces and construct URL
                    clean_name = site.replace(" ", "")
                    url = f"https://www.{clean_name}.com"
        
        try:
            webbrowser.open(url)
            logger.info("Opened website: %s -> %s", site, url)
            return f"Opening {site} in your browser, boss."
        except Exception as e:
            logger.error("Failed to open website: %s", e)
            return f"Sorry boss, couldn't open {site}: {e}"
    
    def search_web(self, query: str) -> str:
        """
        Search Google for a query.
        
        Args:
            query: Search query
        
        Returns:
            Status message
        """
        if not query:
            return "Boss, what should I search for?"
        
        try:
            search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
            webbrowser.open(search_url)
            logger.info("Searching Google for: %s", query)
            return f"Searching Google for '{query}', boss."
        except Exception as e:
            logger.error("Failed to search: %s", e)
            return f"Sorry boss, couldn't search: {e}"

    def search_on_website(self, site: str, query: str) -> str:
        """
        Search inside a specific website instead of opening Google in a new tab.

        Example:
            site="amazon", query="1000 rupees watch"
        """
        site = (site or "").strip().lower()
        query = (query or "").strip()
        if not site or not query:
            return self.search_web(query or site)

        q = query.replace(" ", "+")

        site_search_urls = {
            "amazon": f"https://www.amazon.in/s?k={q}",
            "flipkart": f"https://www.flipkart.com/search?q={q}",
            "youtube": f"https://www.youtube.com/results?search_query={q}",
            "github": f"https://github.com/search?q={q}",
            "wikipedia": f"https://en.wikipedia.org/w/index.php?search={q}",
            "myntra": f"https://www.myntra.com/{q}",
        }

        url = site_search_urls.get(site)
        if not url:
            base = WEBSITE_MAPPINGS.get(site, "")
            if base:
                # Generic fallback using site filter in Google (still one intent)
                domain = base.replace("https://", "").replace("http://", "").strip("/")
                return self.search_web(f"site:{domain} {query}")
            return self.search_web(query)

        try:
            webbrowser.open(url)
            logger.info("Searching on %s for: %s", site, query)
            return f"Searching {site} for '{query}', boss."
        except Exception as e:
            logger.error("Failed site search on %s: %s", site, e)
            return f"Sorry boss, couldn't search on {site}: {e}"

    # ------------------------------------------------------------------
    # Terminal command execution
    # ------------------------------------------------------------------
    def run_terminal_command(self, cmd: str, timeout: int = 30) -> str:
        """Run a terminal command and return the output."""
        # Basic safety: block dangerous commands
        dangerous = ["format", "del /s", "rm -rf", "rmdir /s", "shutdown", "restart"]
        cmd_lower = cmd.lower().strip()

        for d in dangerous:
            if d in cmd_lower:
                msg = f"Whoa boss, I'm not running '{cmd}' — that looks dangerous."
                logger.warning("Blocked dangerous command: %s", cmd)
                return msg

        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd(),
            )

            output = result.stdout.strip()
            error = result.stderr.strip()

            if result.returncode == 0:
                response = f"Command executed successfully."
                if output:
                    response += f"\nOutput:\n{output[:500]}"
                logger.info("Command OK: %s", cmd)
            else:
                response = f"Command finished with errors."
                if error:
                    response += f"\nError:\n{error[:500]}"
                logger.warning("Command error: %s -> %s", cmd, error[:200])

            return response

        except subprocess.TimeoutExpired:
            msg = f"Command timed out after {timeout} seconds."
            logger.error("Command timeout: %s", cmd)
            return msg
        except Exception as e:
            msg = f"Failed to run command: {e}"
            logger.error("Command failed: %s -> %s", cmd, e)
            return msg

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------
    def create_file(self, path: str, content: str) -> str:
        """Create a new file with the given content."""
        try:
            # Create parent directories if needed
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

            if os.path.exists(path):
                return f"File already exists: {path}. Use modify_file to overwrite."

            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            msg = f"Created file: {path}"
            logger.info("Created file: %s", path)
            return msg
        except Exception as e:
            msg = f"Failed to create file: {e}"
            logger.error("File create failed: %s -> %s", path, e)
            return msg

    def modify_file(self, path: str, content: str) -> str:
        """Overwrite an existing file with new content."""
        try:
            if not os.path.exists(path):
                return f"File not found: {path}. Use create_file for new files."

            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            msg = f"Modified file: {path}"
            logger.info("Modified file: %s", path)
            return msg
        except Exception as e:
            msg = f"Failed to modify file: {e}"
            logger.error("File modify failed: %s -> %s", path, e)
            return msg

    def read_file(self, path: str) -> str:
        """Read and return file contents."""
        try:
            if not os.path.exists(path):
                return f"File not found: {path}"

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            logger.info("Read file: %s (%d chars)", path, len(content))
            return content[:2000]  # Limit for voice output
        except Exception as e:
            msg = f"Failed to read file: {e}"
            logger.error("File read failed: %s -> %s", path, e)
            return msg

    def search_files(self, directory: str = ".", pattern: str = "*") -> str:
        """Search for files matching a pattern in a directory."""
        try:
            import glob
            matches = glob.glob(os.path.join(directory, "**", pattern), recursive=True)
            matches = matches[:20]  # Limit results

            if matches:
                result = f"Found {len(matches)} file(s):\n"
                result += "\n".join(f"  • {m}" for m in matches)
            else:
                result = f"No files matching '{pattern}' in {directory}"

            logger.info("File search: %s/%s -> %d results", directory, pattern, len(matches))
            return result
        except Exception as e:
            msg = f"Search failed: {e}"
            logger.error("File search failed: %s", e)
            return msg


# ── quick test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    cmd = CommandExecutor()
    
    print("\n" + "="*50)
    print("  App Open/Close Test Suite")
    print("="*50)
    
    print("\n📋 Feature Status:")
    print(f"  • Windows Search (open any app): {'✅' if PYAUTOGUI_AVAILABLE else '❌'}")
    print(f"  • Window Close (by title):       {'✅' if WIN32_AVAILABLE else '❌'}")
    print(f"  • Process Listing:               {'✅' if PSUTIL_AVAILABLE else '❌'}")
    print(f"  • Fuzzy Matching:                {'✅' if FUZZY_AVAILABLE else '❌'}")
    
    print("\n" + "-"*50)
    choice = input("Test kya karna hai?\n  1. Open app\n  2. Close app\n  3. List running apps\n  4. Skip\nChoice [1-4]: ").strip()
    
    if choice == "1":
        app = input("App name enter karo: ").strip()
        print(cmd.open_application(app))
    elif choice == "2":
        app = input("Close karna hai (window title): ").strip()
        print(cmd.close_application(app))
    elif choice == "3":
        print(cmd.list_running_apps())
    else:
        print("Test skipped.")
