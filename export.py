import torch
import onnx
import onnxruntime
import numpy as np


def test():


    model_GSTF_onnx = onnxruntime.InferenceSession("model_gstf.onnx")
    model_local_onnx = onnxruntime.InferenceSession("model_local.onnx")


    torch.manual_seed(0)
    input_data_gstf = torch.randn(32,250,60)
    input_data_local = torch.randn(32,15000)


    # np.save("input_data_gstf.npy", input_data_gstf.numpy())
    # np.save("input_data_local.npy", input_data_local.numpy())
    
    # data_reshape_gstf = input_data_gstf.numpy().reshape(-1,60)
    # data_reshape_gstf = data_reshape_gstf.tofile("data_reshape_gstf.bin")

    data_reshape_local = input_data_local.numpy()
    data_reshape_local = data_reshape_local.tofile("data_reshape_local.bin")


    # inputs_gstf = {model_GSTF_onnx.get_inputs()[0].name: input_data_gstf.numpy()}
    # inputs_local = {model_local_onnx.get_inputs()[0].name: input_data_local.numpy()}


    # output_gstf = model_GSTF_onnx.run(None, inputs_gstf)[0]
    # output_local = model_local_onnx.run(None, inputs_local)[0]
    # print("-------------input_gstf-------------")
    # print(input_data_gstf)
    # print("------------output_gstf-------------")
    # print(output_gstf.shape)
    # print(output_gstf)
    # print("-------------------------------")

    # print("-------------input_local-------------")
    # print(input_data_local)
    # print("------------output_local-------------")
    # print(output_local.shape)
    # print(output_local)
    # print("-------------------------------")


if __name__ == '__main__':
    test()


