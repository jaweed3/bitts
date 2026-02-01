import torch
import json
import os
import sys
from models_gan import Generator

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Vocoder:
    def __init__(self, checkpoint_path, config_path, device='cpu'):
        print(f"Loading hifi gan... from {checkpoint_path}")

        with open(config_path) as f:
            data = f.read()
            json_config = json.loads(data)
            self.h = AttrDict(json_config)

        self.device = device
        self.generator = Generator(self.h).to(device)

        state_dict_g = torch.load(checkpoint_path, map_location=device)

        if 'generator' in state_dict_g:
            self.generator.load_state_dict(state_dict_g['generator'])
        else:
            self.generator.load_state_dict(state_dict_g)

        self.generator.eval()
        self.generator.remove_weight_norm()
        print("Vocder ready!")

    def infer(self, mel):
        """
        input: mel spectrogram [batch, 80, time] atau [1, 80, time]
        output: audio wavefrom [1, timeAudio]
        """

        with torch.no_grad():
            if mel.device != self.device:
                mel = mel.to(self.device)

            if mel.size(1) != 80 and mel.size(2) == 80:
                print("transporting mel for HIFI GAN ([B, T, 80] -> [B, 80, T])")
                mel = mel.transpose(1, 2)

            audio = self.generator(mel)

            audio = audio.squeeze()

        return audio
