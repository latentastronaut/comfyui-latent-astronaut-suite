# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ComfyUI custom nodes suite focused on batch processing and iteration workflows. The nodes are registered via `__init__.py` which exports `NODE_CLASS_MAPPINGS`, `NODE_DISPLAY_NAME_MAPPINGS`, and `WEB_DIRECTORY`.

## Development

**Installation**: Clone into ComfyUI's `custom_nodes/` directory and restart ComfyUI.

**Testing**: No automated tests. Manual testing in ComfyUI by loading workflows from `workflows/`.

**Dependencies**: Uses ComfyUI's built-in `torch`, `numpy`, and `comfy` modules. No external dependencies.

## Architecture

### Node Structure

Each node module exports:
- `NODE_CLASS_MAPPINGS` - dict mapping internal names to classes
- `NODE_DISPLAY_NAME_MAPPINGS` - dict mapping internal names to display names

Node classes require:
- `INPUT_TYPES()` classmethod returning input spec dict
- `RETURN_TYPES` tuple of output types
- `FUNCTION` string naming the execution method
- `CATEGORY` string for node menu placement

### Dynamic Typing Pattern

The codebase uses a custom `AlwaysEqualProxy` class to implement ComfyUI's "any type" (`*`) pattern:

```python
class AlwaysEqualProxy(str):
    def __eq__(self, _): return True
    def __ne__(self, _): return False

any_type = AlwaysEqualProxy("*")
```

`ByPassTypeTuple` extends this for dynamic output counts by returning the first element for any index access.

### Loop Implementation (Graph Expansion)

For loops use ComfyUI's graph expansion API (`comfy_execution.graph_utils.GraphBuilder`). The `ForLoopEnd.end_loop()` method recursively expands the graph for remaining iterations by:
1. Walking upstream dependencies to find contained nodes
2. Cloning the subgraph with updated iteration index
3. Returning `{"result": ..., "expand": graph.finalize()}`

### Frontend JavaScript

`js/dynamicInputs.js` provides dynamic slot management using LiteGraph's `addInput/removeInput` APIs. Registers via `app.registerExtension()` and hooks into `beforeRegisterNodeDef`.

Key patterns:
- `stabilizeSlots()` - adds/removes slots based on connections
- `renumberValueSlots()` - keeps slot indices sequential after removals
- Widget hiding via `widget.type = "hidden"` for progressive disclosure

## Categories

- `looping/latent-astronaut` - For Loop Start/End nodes
- `utils/latent-astronaut` - String list utilities
- `loaders/latent-astronaut` - LoRA selector nodes
- `experimental/latent-astronaut` - Experimental nodes (in `experimental/` folder, not loaded by default)
