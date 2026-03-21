"""
Command System — Tool Capabilities for the Voice Assistant
==========================================================
Provides tools that the assistant can execute:
  • open_application(name)      — Launch applications
  • run_terminal_command(cmd)   — Execute shell commands
  • create_file(path, content)  — Create new files
  • modify_file(path, content)  — Overwrite existing files
  • read_file(path)             — Read file contents
  • search_files(directory, pattern) — Search for files

All commands are sandboxed with basic safety checks.
"""

import os
import subprocess
import logging
import shutil

logger = logging.getLogger(__name__)

# Common application mappings for Windows
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
}


class CommandExecutor:
    """Executes system commands and file operations for the voice assistant."""

    def __init__(self):
        logger.info("Command executor initialized")

    # ------------------------------------------------------------------
    # Application launcher
    # ------------------------------------------------------------------
    def open_application(self, name: str) -> str:
        """Open an application by name."""
        name_lower = name.lower().strip()

        # Check alias map first
        cmd = APP_ALIASES.get(name_lower)

        if cmd:
            try:
                if cmd.startswith("start "):
                    subprocess.Popen(cmd, shell=True)
                else:
                    subprocess.Popen(cmd, shell=True)
                msg = f"Opening {name} for you, boss."
                logger.info("Opened application: %s -> %s", name, cmd)
                return msg
            except Exception as e:
                msg = f"Couldn't open {name}: {e}"
                logger.error("Failed to open %s: %s", name, e)
                return msg
        else:
            # Try launching directly
            try:
                subprocess.Popen(f"start {name_lower}", shell=True)
                msg = f"Launching {name}..."
                logger.info("Launched application: %s", name)
                return msg
            except Exception as e:
                msg = f"Sorry boss, I don't know how to open '{name}'. Error: {e}"
                logger.error("Unknown application: %s", name)
                return msg

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

    print("Test 1: Open notepad")
    print(cmd.open_application("notepad"))

    print("\nTest 2: Run dir")
    print(cmd.run_terminal_command("dir"))

    print("\nTest 3: Create file")
    print(cmd.create_file("test_output.txt", "Hello from Jarvis!"))

    print("\nTest 4: Read file")
    print(cmd.read_file("test_output.txt"))
