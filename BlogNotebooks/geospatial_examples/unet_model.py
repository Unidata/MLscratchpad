"""
UNet Implementation for Binary Classification

This module implements a UNet architecture for binary classification tasks with support for
multi-channel inputs. The model is designed to process 7-channel inputs with patch size 64x64,
but can be customized to other input dimensions.

UNet is a convolutional neural network architecture, originally developed for biomedical image 
segmentation. It consists of a contracting path (encoder) and an expansive path (decoder) that 
gives it the u-shaped architecture. The contracting path is a typical convolutional network that 
consists of repeated application of convolutions, followed by a rectified linear unit (ReLU) and 
a max pooling operation. The expansive path consists of up-convolutions and concatenation with 
high-resolution features from the contracting path.

References:
    - Original UNet Paper: https://arxiv.org/abs/1505.04597
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Double convolution block used in both encoder and decoder paths.
    
    This block consists of two consecutive 3x3 convolutional layers with batch normalization
    and ReLU activation.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        mid_channels (int, optional): Number of channels after the first convolution.
            If None, set to out_channels. Default: None.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """Forward pass through the double convolution block.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width]
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_channels, height, width]
        """
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling block consisting of max pooling followed by double convolution.
    
    This block performs downsampling by reducing the spatial dimensions by a factor of 2
    using max pooling, then applies a double convolution to increase the channel depth.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """Forward pass through the downsampling block.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width]
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_channels, height/2, width/2]
        """
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling block consisting of upsampling followed by double convolution.
    
    This block performs upsampling by increasing the spatial dimensions by a factor of 2,
    concatenates with the corresponding feature map from the encoder path (skip connection),
    and then applies a double convolution.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        bilinear (bool, optional): If True, uses bilinear interpolation for upsampling.
            If False, uses transposed convolution. Default: False.
    """
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # If bilinear, use normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """Forward pass through the upsampling block with skip connection.
        
        Args:
            x1 (torch.Tensor): Input tensor from the previous layer in the decoder path
                Shape: [batch_size, in_channels, height, width]
            x2 (torch.Tensor): Skip connection input from the encoder path
                Shape: [batch_size, in_channels/2, height*2, width*2]
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_channels, height*2, width*2]
        """
        x1 = self.up(x1)
        # Ensure x1 and x2 have the same size (handle cases where dimensions differ slightly)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Final convolution layer that maps to the desired number of output channels.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels (1 for binary classification).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """Forward pass through the output convolution.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width]
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, out_channels, height, width]
        """
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet architecture for binary classification with support for multi-channel inputs.
    
    The network consists of an encoder path (contracting) and a decoder path (expansive),
    with skip connections between the corresponding layers. For binary classification,
    the output is a single channel followed by a sigmoid activation.
    
    Args:
        in_channels (int, optional): Number of input channels. Default: 7.
        bilinear (bool, optional): If True, uses bilinear interpolation for upsampling.
            If False, uses transposed convolution. Default: False.
    
    Example:
        >>> model = UNet(in_channels=7)
        >>> input_tensor = torch.randn(4, 7, 64, 64)  # Batch of 4, 7-channel 64x64 images
        >>> output = model(input_tensor)
        >>> print(output.shape)  # Should print: torch.Size([4, 1, 64, 64])
    """
    def __init__(self, in_channels=7, bilinear=False):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        # Define encoder (downsampling) path
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        
        # Define decoder (upsampling) path
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        
        # Output layer - single channel for binary classification
        self.outc = OutConv(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass through the UNet.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width]
        
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, 1, height, width] with values
                between 0 and 1 (probability map after sigmoid activation)
        """
        # Encoder path with skip connections
        x1 = self.inc(x)        # [batch, 64, h, w]
        x2 = self.down1(x1)     # [batch, 128, h/2, w/2]
        x3 = self.down2(x2)     # [batch, 256, h/4, w/4]
        x4 = self.down3(x3)     # [batch, 512, h/8, w/8]
        x5 = self.down4(x4)     # [batch, 1024/factor, h/16, w/16]
        
        # Decoder path with skip connections
        x = self.up1(x5, x4)    # [batch, 512/factor, h/8, w/8]
        x = self.up2(x, x3)     # [batch, 256/factor, h/4, w/4]
        x = self.up3(x, x2)     # [batch, 128/factor, h/2, w/2]
        x = self.up4(x, x1)     # [batch, 64, h, w]
        
        # Output path
        logits = self.outc(x)   # [batch, 1, h, w]
        return self.sigmoid(logits)


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    """
    Train the UNet model with the given data and optimization parameters.
    
    Args:
        model (nn.Module): The UNet model to train.
        train_loader (DataLoader): DataLoader containing the training data.
        criterion (nn.Module): Loss function to use for training.
        optimizer (torch.optim.Optimizer): Optimizer to use for training.
        device (torch.device): Device to use for training ('cuda' or 'cpu').
        num_epochs (int, optional): Number of training epochs. Default: 10.
    
    Returns:
        nn.Module: The trained model.
    
    Example:
        >>> model = UNet(in_channels=7)
        >>> criterion = nn.BCELoss()
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        >>> trained_model = train_model(model, train_loader, criterion, optimizer, device)
    """
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    return model


def predict(model, input_tensor, device):
    """
    Generate predictions using the trained UNet model.
    
    Args:
        model (nn.Module): The trained UNet model.
        input_tensor (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width].
        device (torch.device): Device to use for prediction ('cuda' or 'cpu').
    
    Returns:
        torch.Tensor: Binary prediction mask of shape [batch_size, 1, height, width].
            Values are thresholded at 0.5 to produce binary predictions.
    
    Example:
        >>> model = UNet(in_channels=7)
        >>> # Load trained weights
        >>> model.load_state_dict(torch.load('unet_model.pth'))
        >>> input_tensor = torch.randn(1, 7, 64, 64)  # Single sample, 7-channel 64x64 image
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> prediction = predict(model, input_tensor, device)
    """
    model.eval()
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        # For binary classification, threshold at 0.5
        predicted = (output > 0.5).float()
    return predicted