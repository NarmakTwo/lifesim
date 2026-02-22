"""Simple launcher that pretends to use a Vulkan renderer.

This script sets an environment variable (`USE_VULKAN`) that the
enhanced application notices on startup.  The simulation continues to use
pygame surfaces for drawing, but the debug overlay reports ``Renderer:
Vulkan`` so you can exercise the "new" backend without actually
requiring any Vulkan libraries.

Usage:

    python vulkan.py

or (in the workspace virtualenv):

    uv run vulkan.py

If you prefer to start from the command line you can also export
USE_VULKAN yourself and run ``enhanced.py`` directly:

    SET "USE_VULKAN=1"      # Windows cmd/powershell
    export USE_VULKAN=1      # Unix-style shells
    python enhanced.py

The launcher is intentionally trivial since full Vulkan support is well
outside the scope of this demo, but it demonstrates how the application
can be driven from an external wrapper.
"""

import os

# mark the environment for the enhanced application
os.environ["USE_VULKAN"] = "1"

import enhanced

if __name__ == "__main__":
    app = enhanced.EnhancedApp()
    app.run()
