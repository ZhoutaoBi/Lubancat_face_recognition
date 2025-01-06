import onnx
# 使用onnx模块
model = onnx.load('new_yunet1.onnx')
onnx.checker.check_model(model) # 验证Onnx模型是否准确
print(model.graph)

print(model.graph.input)
for input_node in model.graph.input:
    print(input_node)
for input_node in model.graph.input:
    print(input_node.type)
for input_node in model.graph.input:
    print(input_node.type.tensor_type.shape)




