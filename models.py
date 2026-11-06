"""
Module: models.py

This module defines the LRCN (Long-term Recurrent Convolutional Network) model for video
classification. The LRCN model combines a 2D CNN backbone (e.g., ResNet) for spatial feature
extraction from individual frames with an LSTM to capture temporal dynamics across frames.
An additional fully-connected layer is used to output the final class predictions.

Classes:
    Identity: A helper module that returns the input unchanged. It is used to replace the fully-connected
              layer of the ResNet backbone.
    LRCN: The main model that integrates the CNN backbone, an LSTM, dropout regularization, and a final
          fully-connected layer to produce class logits.
"""

import torch
import torch.nn as nn
from torchvision import models

class Identity(nn.Module):
    """
    A placeholder identity operator that is argument-insensitive.
    
    This module is used to replace the fully-connected (fc) layer in the ResNet backbone,
    effectively making the backbone output the raw features before classification.
    
    Example:
        >>> identity = Identity()
        >>> output = identity(input_tensor)
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """
        Forward pass that returns the input as is.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: The same tensor x.
        """
        return x

class LRCN(nn.Module):
    """Classify a video using an LSTM over ordered visual features.

    In frame mode, the ResNet extracts one feature vector per frame.
    In embedding mode, the caller supplies the feature vectors directly.
    Both modes use the same LSTM, temporal pooling, dropout, and classifier.

    Args:
        hidden_size: Number of features in the LSTM hidden state.
        n_layers: Number of recurrent layers.
        dropout_rate: Dropout probability before the classifier.
        n_classes: Number of output classes.
        pretrained: Whether to load pretrained weights for the frame encoder.
        cnn_model: ResNet backbone to use in frame mode.
        temporal_pool: Use the last LSTM output ("last") or average the
            outputs across time ("mean").
        embedding_size: Width of the supplied feature vectors. Use None
            to build the frame encoder, or 512 for R(2+1)D-18 embeddings.
    """
    def __init__(
        self,
        hidden_size,
        n_layers,
        dropout_rate,
        n_classes,
        pretrained=True,
        cnn_model="resnet34",
        temporal_pool="last",
        embedding_size=None,
    ):
        super().__init__()
        if temporal_pool not in ("last", "mean"):
            raise ValueError("temporal_pool must be 'last' or 'mean'.")
        self.temporal_pool = temporal_pool

        self.base_model = None
        if embedding_size is None:
            self.base_model = self.build_frame_encoder(cnn_model, pretrained)
            embedding_size = self.base_model.fc.in_features
            self.base_model.fc = Identity()

        self.rnn = nn.LSTM(
            embedding_size, hidden_size, n_layers, batch_first=True
        )
        
        # Define dropout for regularization.
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final fully-connected layer to produce logits for each class.
        self.fc = nn.Linear(hidden_size, n_classes)

    @staticmethod
    def build_frame_encoder(cnn_model, pretrained):
        """Build the ResNet used by the frame-based LRCN."""
        backbones = {
            "resnet18": models.resnet18,
            "resnet34": models.resnet34,
            "resnet50": models.resnet50,
            "resnet101": models.resnet101,
            "resnet152": models.resnet34,
        }
        if cnn_model not in backbones:
            raise ValueError(f"Unsupported CNN backbone: {cnn_model}")
        return backbones[cnn_model](pretrained=pretrained)

    def forward(self, inputs):
        """Return class logits with shape (batch, classes).

        In frame mode, inputs have shape (batch, time, channels, height, width).
        In embedding mode, inputs have shape (batch, time, embedding_size).
        In both modes, the time axis must be in chronological order.
        """
        if self.base_model is not None:
            inputs = torch.stack(
                [self.base_model(frame) for frame in inputs.unbind(dim=1)],
                dim=1,
            )

        outputs, _ = self.rnn(inputs)
        if self.temporal_pool == "mean":
            clip_features = outputs.mean(dim=1)
        else:
            clip_features = outputs[:, -1]

        return self.fc(self.dropout(clip_features))
