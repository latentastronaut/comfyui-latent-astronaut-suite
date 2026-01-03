import { app } from "../../scripts/app.js";

/**
 * Dynamic inputs for Latent Astronaut nodes.
 * Uses LiteGraph's addInput/removeInput for truly dynamic slots.
 */

app.registerExtension({
    name: "latentAstronaut.dynamicInputs",

    beforeRegisterNodeDef(nodeType, nodeData, app) {

        // For Loop Start - dynamic value inputs and outputs
        if (nodeData.name === "ForLoopStart") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated?.apply(this, arguments);

                // Remove all predefined value inputs/outputs first
                // Python defines initial_value0-9, we want to start with just one
                for (let i = this.inputs.length - 1; i >= 0; i--) {
                    if (this.inputs[i].name.startsWith("initial_value")) {
                        this.removeInput(i);
                    }
                }
                for (let i = this.outputs.length - 1; i >= 0; i--) {
                    if (this.outputs[i].name.startsWith("value")) {
                        this.removeOutput(i);
                    }
                }

                // Add one empty value slot
                this.addInput("initial_value0", "*");
                this.addOutput("value0", "*");
            };

            nodeType.prototype.stabilizeSlots = function() {
                if (!this.inputs || !this.outputs) return;

                // Find value inputs (initial_value*)
                const valueInputs = this.inputs.filter(i => i.name.startsWith("initial_value"));

                // Check if last value input is connected
                const lastValueInput = valueInputs[valueInputs.length - 1];
                const hasEmptyValueInput = lastValueInput && !lastValueInput.link;

                if (!hasEmptyValueInput && valueInputs.length < 10) {
                    // Add new value input/output pair
                    const nextIndex = valueInputs.length;
                    this.addInput(`initial_value${nextIndex}`, "*");
                    this.addOutput(`value${nextIndex}`, "*");
                }

                // Remove unconnected inputs from middle (keep last empty one)
                const currentValueInputs = this.inputs.filter(i => i.name.startsWith("initial_value"));
                for (let i = currentValueInputs.length - 2; i >= 0; i--) {
                    const input = currentValueInputs[i];
                    if (!input.link) {
                        const inputIndex = this.inputs.indexOf(input);
                        if (inputIndex !== -1) {
                            this.removeInput(inputIndex);
                            // Find and remove corresponding output
                            const outputName = input.name.replace("initial_", "");
                            const outputIndex = this.outputs.findIndex(o => o.name === outputName);
                            if (outputIndex !== -1) {
                                this.removeOutput(outputIndex);
                            }
                        }
                    }
                }

                // Renumber remaining slots
                this.renumberValueSlots();
                app.graph?.setDirtyCanvas(true, true);
            };

            nodeType.prototype.renumberValueSlots = function() {
                let valueIndex = 0;
                for (const input of this.inputs) {
                    if (input.name.startsWith("initial_value")) {
                        input.name = `initial_value${valueIndex}`;
                        valueIndex++;
                    }
                }
                valueIndex = 0;
                for (const output of this.outputs) {
                    if (output.name.startsWith("value")) {
                        output.name = `value${valueIndex}`;
                        valueIndex++;
                    }
                }
            };

            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function(type, index, connected, linkInfo, ioSlot) {
                onConnectionsChange?.apply(this, arguments);
                setTimeout(() => this.stabilizeSlots(), 10);
            };
        }

        // For Loop End - dynamic value inputs and outputs
        if (nodeData.name === "ForLoopEnd") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated?.apply(this, arguments);

                // Remove all predefined value inputs/outputs first
                for (let i = this.inputs.length - 1; i >= 0; i--) {
                    if (this.inputs[i].name.startsWith("value") && this.inputs[i].name !== "flow") {
                        this.removeInput(i);
                    }
                }
                for (let i = this.outputs.length - 1; i >= 0; i--) {
                    if (this.outputs[i].name.startsWith("value")) {
                        this.removeOutput(i);
                    }
                }

                // Add one empty value slot
                this.addInput("value0", "*");
                this.addOutput("value0", "*");
            };

            nodeType.prototype.stabilizeSlots = function() {
                if (!this.inputs || !this.outputs) return;

                // Find value inputs (value* but not flow)
                const valueInputs = this.inputs.filter(i => i.name.startsWith("value") && i.name !== "flow");

                // Check if last value input is connected
                const lastValueInput = valueInputs[valueInputs.length - 1];
                const hasEmptyValueInput = lastValueInput && !lastValueInput.link;

                if (!hasEmptyValueInput && valueInputs.length < 10) {
                    const nextIndex = valueInputs.length;
                    this.addInput(`value${nextIndex}`, "*");
                    this.addOutput(`value${nextIndex}`, "*");
                }

                // Remove unconnected inputs from middle
                const currentValueInputs = this.inputs.filter(i => i.name.startsWith("value") && i.name !== "flow");
                for (let i = currentValueInputs.length - 2; i >= 0; i--) {
                    const input = currentValueInputs[i];
                    if (!input.link) {
                        const inputIndex = this.inputs.indexOf(input);
                        if (inputIndex !== -1) {
                            this.removeInput(inputIndex);
                            const outputIndex = this.outputs.findIndex(o => o.name === input.name);
                            if (outputIndex !== -1) {
                                this.removeOutput(outputIndex);
                            }
                        }
                    }
                }

                this.renumberValueSlots();
                app.graph?.setDirtyCanvas(true, true);
            };

            nodeType.prototype.renumberValueSlots = function() {
                let valueIndex = 0;
                for (const input of this.inputs) {
                    if (input.name.startsWith("value") && input.name !== "flow") {
                        input.name = `value${valueIndex}`;
                        valueIndex++;
                    }
                }
                valueIndex = 0;
                for (const output of this.outputs) {
                    if (output.name.startsWith("value")) {
                        output.name = `value${valueIndex}`;
                        valueIndex++;
                    }
                }
            };

            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function(type, index, connected, linkInfo, ioSlot) {
                onConnectionsChange?.apply(this, arguments);
                setTimeout(() => this.stabilizeSlots(), 10);
            };
        }

        // LoRA Selector nodes - dynamic lora slots with Reset All button
        if (nodeData.name === "LoraLoaderModelOnlySelector" || nodeData.name === "LoraLoaderSelector") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated?.apply(this, arguments);

                // Add Reset All button
                this.addWidget("button", "Reset All", null, () => {
                    for (const widget of this.widgets) {
                        if (widget.name.startsWith("lora_")) {
                            widget.value = "None";
                        }
                    }
                    this.stabilizeLoraSlots();
                });

                this.stabilizeLoraSlots();
            };

            nodeType.prototype.stabilizeLoraSlots = function() {
                if (!this.widgets) return;

                // Find lora widgets
                const loraWidgets = this.widgets.filter(w => w.name.startsWith("lora_"));

                // Count how many are set to non-None, show that many + 1
                let lastNonNoneIndex = -1;
                for (let i = 0; i < loraWidgets.length; i++) {
                    if (loraWidgets[i].value !== "None") {
                        lastNonNoneIndex = i;
                    }
                }

                // Show up to lastNonNoneIndex + 2 (the used ones + 1 empty)
                const numToShow = Math.min(lastNonNoneIndex + 2, 16);

                for (let i = 0; i < loraWidgets.length; i++) {
                    const widget = loraWidgets[i];
                    if (i < numToShow) {
                        widget.type = widget._origType || widget.type;
                        if (widget._origType) delete widget._origType;
                    } else {
                        if (widget.type !== "hidden") {
                            widget._origType = widget.type;
                            widget.type = "hidden";
                        }
                    }
                }

                // Resize node
                requestAnimationFrame(() => {
                    this.setSize([this.size[0], this.computeSize()[1]]);
                    app.graph?.setDirtyCanvas(true, false);
                });
            };

            // Watch for widget value changes
            const onWidgetValueChange = nodeType.prototype.onWidgetValueChange;
            nodeType.prototype.onWidgetValueChange = function(name, value) {
                onWidgetValueChange?.apply(this, arguments);
                if (name.startsWith("lora_")) {
                    this.stabilizeLoraSlots();
                }
            };

            // Also hook into the widget callback
            const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(canvas, options) {
                origGetExtraMenuOptions?.apply(this, arguments);

                // Setup widget callbacks if not already done
                if (!this._loraCallbacksSetup) {
                    this._loraCallbacksSetup = true;
                    for (const widget of this.widgets) {
                        if (widget.name.startsWith("lora_")) {
                            const origCallback = widget.callback;
                            widget.callback = (value) => {
                                origCallback?.call(widget, value);
                                setTimeout(() => this.stabilizeLoraSlots(), 10);
                            };
                        }
                    }
                }
            };
        }
    }
});
