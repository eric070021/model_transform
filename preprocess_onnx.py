import onnx

onnx_model = onnx.load('touch_pen_0511_original.onnx')

onnx_model.graph.input[0].type.CopyFrom(
    onnx.helper.make_tensor_type_proto(1, [1, 1, 11, 11])
)
onnx_model.graph.output[0].type.CopyFrom(
    onnx.helper.make_tensor_type_proto(1, [1, 2, 1, 1])
)

for i in reversed(range(len(onnx_model.graph.node))):
    nd = onnx_model.graph.node[i]
    print(nd.name)
    if nd.name == '/conv/conv.0/Conv':
        nd.input[0] = 'input'
        nd.attribute[3].CopyFrom(
            onnx.helper.make_attribute('pads', [0, 0, 0, 0])
        )
    if nd.name == '/conv/conv.8/Conv':
        nd.output[0] = 'output'
    if nd.name in ['/Constant', '/Reshape', '/Constant_1', '/Reshape_1', '/Clip']:
        del onnx_model.graph.node[i]

onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

onnx.save(onnx_model, 'touch_pen_0511.onnx')
