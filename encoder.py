import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module): 
    def __init__(self, pretrained=True, latent_dim=128): 
        super().__init__()
        
        model_base = models.resnet18(pretrained=pretrained)
        self.encoder = nn.Sequential(*list(model_base.children())[:-1]) # remove head
        self.fc = nn.Linear(512, latent_dim)

    def forward(self, x): 
        return self.fc(self.encoder(x).squeeze(-1).squeeze(-1))

class CURL(nn.Module): 
    def __init__(self, pretrained=True, latent_dim=128):
        super().__init__()

        self.pretrained = pretrained
        self.encoder = Encoder(pretrained=pretrained, latent_dim=latent_dim)

        if not self.pretrained: 
            self.key_encoder = Encoder(pretrained=pretrained, latent_dim=latent_dim)
            
            for param in self.key_encoder.parameters(): 
                param.requires_grad = False

    # Only gets called if training encoder alongside SAC 
    # TODO check if q gets updated or k here
    def momentum(self, tau=0.05): 
        for q, k in zip(self.encoder.parameters(), self.key_encoder.parameters()): 
            k.data = tau * q.data + (1 - tau) * k.data

    def forward(self, x): 
        return self.encoder(x)
