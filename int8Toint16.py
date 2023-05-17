import onnx
from onnx import numpy_helper
import numpy as np

def parse_onnx_model(model_path):
    # Load the ONNX model
    model = onnx.load(model_path)

    # Print the model information
    print("ONNX Model Information:")
    print(f"Model IR Version: {model.ir_version}")
    print(f"Producer Name: {model.producer_name}")
    print(f"Producer Version: {model.producer_version}")
    print(f"Model Domain: {model.domain}")
    print(f"Model Description: {model.doc_string}")
    print(f"Number of Inputs: {len(model.graph.input)}")
    print(f"Number of Outputs: {len(model.graph.output)}")
    print(f"Number of Nodes: {len(model.graph.node)}")

    # Print the input information
    print("\nInput Information:")
    for i, input in enumerate(model.graph.input):
        print(f"Input {i}: {input.name} (Type: {input.type})")

    # Print the output information
    print("\nOutput Information:")
    for i, output in enumerate(model.graph.output):
        print(f"Output {i}: {output.name} (Type: {output.type})")

    # Print the node information
    print("\nNode Information:")
    for i, node in enumerate(model.graph.node):
        print(f"Node {i}: {node.op_type} (Name: {node.name})")
        print("  Inputs:")
        for input in node.input:
            print(f"    {input}")
        print("  Outputs:")
        for output in node.output:
            print(f"    {output}")
        print()

    # You can perform additional operations on the parsed model here

    # For example, you can access the model's metadata properties
    metadata_props = model.metadata_props
    if metadata_props:
        print("\nModel Metadata Properties:")
        for prop in metadata_props:
            print(f"{prop.key}: {prop.value}")

    # Or you can access the model's initializer tensors
    initializers = model.graph.initializer
    if initializers:
        print("\nInitializer Tensors:")
        for initializer in initializers:
            print(f"{initializer.name} (Type: {initializer.data_type}, Shape: {initializer.dims})")


def int8Toint16(model_name, model_path, model_quant_path):
    # Load the ONNX model
    model = onnx.load(model_path)
    model_quant = onnx.load(model_quant_path)

    # Store weights and biases in a dictionary
    model_dic = {}
    model_quant_dic = {}

    # access the float32 model's initializer tensors
    initializers = model.graph.initializer
    for initializer in initializers:
        model_dic[initializer.name] = numpy_helper.to_array(initializer)
    
    # access the INT8 model's initializer tensors
    initializers = model_quant.graph.initializer
    for initializer in initializers:
        if initializer.name[-6:] == "_scale":
            model_quant_dic[initializer.name] = numpy_helper.to_array(initializer)

    # convert INT8 model to INT16
    initializers = model_quant.graph.initializer
    for initializer in initializers:
        if initializer.name[-6:] == "_scale":
            float_scale = np.array(initializer.float_data) / 256
            INT16_scale = np.array(float_scale, dtype=np.int16)
            initializer.CopyFrom(numpy_helper.from_array(INT16_scale))
        elif initializer.name[-10:] == "_quantized":
            float_data = model_dic[initializer.name[:-10]]
            if((initializer.name[:-10] + "_scale") in model_quant_dic):
                scale = model_quant_dic[initializer.name[:-10] + "_scale"] # weight case
            else:
                scale = model_quant_dic[initializer.name[:-10] + "_quantized_scale"] # bias case
            float_data = float_data * 256 #/ scale
            INT16_data = np.array(float_data, dtype=np.int16)
            initializer.CopyFrom(numpy_helper.from_array(INT16_data))

    # save the model
    onnx.save(model_quant, model_name + "_s16s16_ptq.onnx")

# Provide the path to your ONNX model files
model_name = "yolov7"
model_path = "yolov7.onnx"
model_quant_path = "yolov7_s8s8_ptq.onnx"

# Parse and convert quanted IN8 ONNX model to INT16, save as "model_s16s16_ptq.onnx"
int8Toint16(model_name, model_path, model_quant_path)
parse_onnx_model(model_name + "_s16s16_ptq.onnx")
