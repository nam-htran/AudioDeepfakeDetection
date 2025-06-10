# notebooks/cnn_trainer.py
import torch
import torch.nn as nn
# import torch.nn.functional as F # Không cần cho lớp CNN_Audio này

class CNN_Audio(nn.Module):
    def __init__(self, img_size: int, in_channels: int, num_classes: int,
                 linear_output_units_1st_fc: int,
                 cnn_conv_channels: list[int], cnn_pool_after_conv: list[bool], 
                 dropout: float = 0.3):
        super(CNN_Audio, self).__init__()
        self.in_channels = in_channels
        self.cnn_conv_channels = cnn_conv_channels
        self.cnn_pool_after_conv = cnn_pool_after_conv
        # img_size được truyền vào __init__ từ notebook, có thể được dùng để tính toán kích thước động
        # nếu cần, hoặc chỉ để nhất quán với các config khác.
        # Trong trường hợp này, nó dùng để tính self.pooled_size qua dummy_input
        self.expected_input_img_size = img_size 
        self.dropout = dropout
        self.num_classes = num_classes

        conv_layers_list = []
        current_in_channels = self.in_channels
        
        # Biến tạm để theo dõi kích thước không gian, nếu cần tính toán phức tạp hơn
        # current_h, current_w = self.expected_input_img_size, self.expected_input_img_size

        for i, out_channels in enumerate(self.cnn_conv_channels):
            conv_layers_list.append(nn.Conv2d(current_in_channels, out_channels, kernel_size=3, padding=1, bias=False))
            conv_layers_list.append(nn.BatchNorm2d(out_channels))
            conv_layers_list.append(nn.ReLU(inplace=True))
            
            if self.cnn_pool_after_conv[i]:
                conv_layers_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
                # current_h //= 2
                # current_w //= 2
            
            # Dropout2d được áp dụng sau mỗi block conv+bn+relu+pool (nếu có)
            conv_layers_list.append(nn.Dropout2d(self.dropout)) 
            current_in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers_list)

        # Tính toán self.pooled_size dựa trên output của conv_layers và adaptive_pool
        # AdaptiveAvgPool2d((4, 4)) sẽ tạo ra output (N, current_in_channels, 4, 4)
        # current_in_channels ở đây là số kênh đầu ra của lớp conv cuối cùng
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.pooled_size = current_in_channels * 4 * 4

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.pooled_size, linear_output_units_1st_fc),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout), # Dropout 1D (mặc định của nn.Dropout)
            nn.Linear(linear_output_units_1st_fc, linear_output_units_1st_fc // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(linear_output_units_1st_fc // 2, num_classes)
        )

    def forward(self, x):
        # x đầu vào đây dự kiến là (batch, 1, self.expected_input_img_size, self.expected_input_img_size)
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1) # Flatten
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    # Test code (tùy chọn, có thể xóa khi deploy)
    # Sử dụng các giá trị từ Config của notebook CNN_Small
    config_params_small = {
        "img_size": 224, 
        "in_channels": 1, 
        "num_classes": 2,
        "linear_output_units_1st_fc": 192,
        "cnn_conv_channels": [32, 64, 128],
        "cnn_pool_after_conv": [True, True, True],
        "dropout": 0.3
    }
    model_small = CNN_Audio(**config_params_small)
    print("CNN_Audio (Small) initialized.")
    print(model_small)
    dummy_input = torch.randn(2, 1, 224, 224) # Batch 2, 1 channel, 224x224
    output = model_small(dummy_input)
    print("Output shape (Small):", output.shape) # Expected: (2, 2)

    config_params_large = {
        "img_size": 224, 
        "in_channels": 1, 
        "num_classes": 2,
        "linear_output_units_1st_fc": 192, # Theo notebook là 192
        "cnn_conv_channels": [64, 128, 256, 512, 512],
        "cnn_pool_after_conv": [True, True, True, True, False],
        "dropout": 0.3
    }
    model_large = CNN_Audio(**config_params_large)
    print("\nCNN_Audio (Large) initialized.")
    print(model_large)
    output_large = model_large(dummy_input)
    print("Output shape (Large):", output_large.shape) # Expected: (2, 2)