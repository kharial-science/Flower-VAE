"""
The generator part of the gan is exactly the same as the decoder part of the vae.
"""

import src.models.vae.parts.decoder as Decoder

Generator = Decoder.Decoder
