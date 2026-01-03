"""
For Loop nodes for ComfyUI.
Simple for loop that iterates a fixed number of times with dynamic value slots.
"""

from comfy_execution.graph_utils import GraphBuilder, is_link

MAX_FLOW_NUM = 10


class AlwaysEqualProxy(str):
    """A string that equals any other value - used for dynamic typing."""
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


class TautologyStr(str):
    """A string that is never not-equal to anything."""
    def __ne__(self, _):
        return False


class ByPassTypeTuple(tuple):
    """Tuple that returns first item for any index access - for dynamic outputs."""
    def __getitem__(self, index):
        if index > 0:
            index = 0
        item = super().__getitem__(index)
        if isinstance(item, str):
            return TautologyStr(item)
        return item


any_type = AlwaysEqualProxy("*")


class ForLoopStart:
    """
    Start a for loop that iterates `count` times.
    Outputs the current index (0 to count-1) and optional pass-through values.
    Value slots appear dynamically as you connect them.
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "count": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 10000,
                    "tooltip": "Number of iterations"
                }),
            },
            "optional": {},
            "hidden": {
                "_current_index": ("INT", {"default": 0}),
                "unique_id": "UNIQUE_ID",
            }
        }
        # Add dynamic value inputs (JS will show/hide these dynamically)
        for i in range(MAX_FLOW_NUM):
            inputs["optional"][f"initial_value{i}"] = (any_type,)

        return inputs

    RETURN_TYPES = ByPassTypeTuple(tuple(["FOR_LOOP_FLOW", "INT"] + [any_type] * MAX_FLOW_NUM))
    RETURN_NAMES = ByPassTypeTuple(tuple(["flow", "index"] + [f"value{i}" for i in range(MAX_FLOW_NUM)]))

    FUNCTION = "start_loop"
    CATEGORY = "looping/latent-astronaut"

    def start_loop(self, count, _current_index=0, unique_id=None, **kwargs):
        values = [kwargs.get(f"initial_value{i}", None) for i in range(MAX_FLOW_NUM)]
        # Flow carries node_id and resolved count so ForLoopEnd can access both
        flow_data = (str(unique_id), count)
        return tuple([flow_data, _current_index] + values)


class ForLoopEnd:
    """
    End a for loop. Continues iterating until count is reached.
    Pass values back to continue accumulating across iterations.
    Value slots appear dynamically as you connect them.
    """

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "flow": ("FOR_LOOP_FLOW", {
                    "tooltip": "Connect from ForLoopStart's flow"
                }),
            },
            "optional": {},
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            }
        }
        # Add dynamic value inputs (JS will show/hide these dynamically)
        for i in range(MAX_FLOW_NUM):
            inputs["optional"][f"value{i}"] = (any_type,)

        return inputs

    RETURN_TYPES = ByPassTypeTuple(tuple([any_type] * MAX_FLOW_NUM))
    RETURN_NAMES = ByPassTypeTuple(tuple([f"value{i}" for i in range(MAX_FLOW_NUM)]))

    FUNCTION = "end_loop"
    CATEGORY = "looping/latent-astronaut"

    def explore_dependencies(self, node_id, dynprompt, upstream):
        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return
        for k, v in node_info["inputs"].items():
            if is_link(v):
                parent_id = v[0]
                if parent_id not in upstream:
                    upstream[parent_id] = []
                    self.explore_dependencies(parent_id, dynprompt, upstream)
                upstream[parent_id].append(node_id)

    def collect_contained(self, node_id, upstream, contained):
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                self.collect_contained(child_id, upstream, contained)

    def end_loop(self, flow, dynprompt=None, unique_id=None, **kwargs):
        # Collect values
        values = [kwargs.get(f"value{i}", None) for i in range(MAX_FLOW_NUM)]

        # flow is (start_node_id, count) tuple from ForLoopStart
        open_node_id = flow[0]
        count = flow[1]

        # Get current index from start node
        start_node = dynprompt.get_node(open_node_id)
        current_index = start_node["inputs"].get("_current_index", 0)

        # Check if we should continue looping
        next_index = current_index + 1
        if next_index >= count:
            # Done - return final values
            return tuple(values)

        # More iterations needed - expand graph
        upstream = {}
        self.explore_dependencies(unique_id, dynprompt, upstream)

        contained = {}
        self.collect_contained(open_node_id, upstream, contained)
        contained[unique_id] = True
        contained[open_node_id] = True

        # Build the recursive graph
        graph = GraphBuilder()
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.node(
                original_node["class_type"],
                "Recurse" if node_id == unique_id else node_id
            )
            node.set_override_display_id(node_id)

        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.lookup_node("Recurse" if node_id == unique_id else node_id)
            for k, v in original_node["inputs"].items():
                if is_link(v) and v[0] in contained:
                    parent = graph.lookup_node(v[0])
                    node.set_input(k, parent.out(v[1]))
                else:
                    node.set_input(k, v)

        # Update the start node with next index and pass values
        new_open = graph.lookup_node(open_node_id)
        new_open.set_input("_current_index", next_index)
        for i in range(MAX_FLOW_NUM):
            new_open.set_input(f"initial_value{i}", values[i])

        my_clone = graph.lookup_node("Recurse")
        result = tuple(my_clone.out(i) for i in range(MAX_FLOW_NUM))

        return {
            "result": result,
            "expand": graph.finalize(),
        }


NODE_CLASS_MAPPINGS = {
    "ForLoopStart": ForLoopStart,
    "ForLoopEnd": ForLoopEnd,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ForLoopStart": "For Loop Start",
    "ForLoopEnd": "For Loop End",
}
