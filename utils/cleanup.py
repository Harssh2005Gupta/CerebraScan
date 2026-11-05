"""
Lifecycle and Garbage Collection for Temporary Visualization Artifacts.

Provides thread-safe pruning of generated Grad-CAM overlays and comparison
composites to prevent disk space leaks and denial-of-service storage exhaustion.
"""

import logging
import os
import time
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


def cleanup_old_files(
    directory: Union[str, Path],
    ttl_seconds: int = 3600,
    max_files: int = 500,
) -> int:
    """
    Remove expired or surplus temporary files from the specified directory.

    Args:
        directory: Path to directory containing temporary artifacts.
        ttl_seconds: Maximum age in seconds before a file is considered expired.
        max_files: Maximum allowed files in directory; oldest are deleted if exceeded.

    Returns:
        Number of files removed.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        return 0

    now = time.time()
    deleted_count = 0
    file_records = []

    # 1. Collect candidate files matching our generated prefixes
    try:
        for entry in os.scandir(dir_path):
            if entry.is_file() and (
                entry.name.startswith("gradcam_") or entry.name.startswith("comparison_")
            ):
                try:
                    stat = entry.stat()
                    file_records.append((entry.path, stat.st_mtime))
                except OSError:
                    continue
    except OSError as e:
        logger.warning("Could not scan directory '%s' for cleanup: %s", dir_path, e)
        return 0

    # 2. Delete files older than TTL
    remaining_records = []
    for path, mtime in file_records:
        if (now - mtime) > ttl_seconds:
            try:
                os.remove(path)
                deleted_count += 1
            except OSError:
                remaining_records.append((path, mtime))
        else:
            remaining_records.append((path, mtime))

    # 3. If file count still exceeds max_files, purge oldest first
    if len(remaining_records) > max_files:
        # Sort ascending by modification time (oldest first)
        remaining_records.sort(key=lambda x: x[1])
        excess = len(remaining_records) - max_files
        for i in range(excess):
            path, _ = remaining_records[i]
            try:
                os.remove(path)
                deleted_count += 1
            except OSError:
                pass

    if deleted_count > 0:
        logger.debug("Cleaned up %d temporary files in '%s'.", deleted_count, dir_path)

    return deleted_count
