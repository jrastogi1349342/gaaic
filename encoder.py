import torch.nn as nn
import torchvision.models as models
import utils

class Encoder(nn.Module): 
    def __init__(self, pretrained=True, latent_dim=128, device="cuda"): 
        super().__init__()
        
        # TODO fix code for if weights are not pretrained
        # 166 MB VRAM
        model_base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(model_base.children())[:-1]).half().to(device) # remove head
        self.fc = nn.Linear(512, latent_dim).half().to(device)

    def forward(self, x): 
        return self.fc(self.encoder(x).squeeze(-1).squeeze(-1))

class CURL(nn.Module): 
    def __init__(self, pretrained=True, latent_dim=128, device="cuda"):
        super().__init__()

        self.pretrained = pretrained
        self.device = device
        self.encoder = Encoder(pretrained=pretrained, latent_dim=latent_dim, device=device)

        if not self.pretrained: 
            self.key_encoder = Encoder(pretrained=pretrained, latent_dim=latent_dim).to(device)
            
            for param in self.key_encoder.parameters(): 
                param.requires_grad = False

    # Only gets called if training encoder alongside SAC 
    # TODO check if q gets updated or k here
    def momentum(self, tau=0.05): 
        for q, k in zip(self.encoder.parameters(), self.key_encoder.parameters()): 
            k.data = tau * q.data + (1 - tau) * k.data

    def forward(self, x): 
        return self.encoder(x)

# curl = CURL(pretrained=True, latent_dim=256, device="cuda")
# print(curl.parameters)
# print(utils.get_mem_used())
