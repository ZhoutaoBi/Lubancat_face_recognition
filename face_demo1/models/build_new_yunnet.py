import onnx
# 使用onnx模块
model = onnx.load('yunet.onnx')
onnx.checker.check_model(model) # 验证Onnx模型是否准确
print(model.graph)

print(model.graph.input)
for input_node in model.graph.input:
    print(input_node)
for input_node in model.graph.input:
    print(input_node.type)
for input_node in model.graph.input:
    print(input_node.type.tensor_type.shape)

for input_node in model.graph.input:
	print(input_node.type.tensor_type.shape)
	input_node.type.tensor_type.shape.dim[2].dim_value = 320
	input_node.type.tensor_type.shape.dim[3].dim_value = 320
	print(input_node.type.tensor_type.shape)

onnx.checker.check_model(model)
onnx.save(model, 'new_yunet1.onnx')


